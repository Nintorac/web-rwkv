#!/usr/bin/env python3
"""Decode SQTT data from RADV .rgp files using AMD's rocprofiler decoder.

Extracts per-wave execution timing from hardware thread traces.

Usage: python3 scripts/rgp_decode_sqtt.py /tmp/foo.rgp
"""
import struct, sys, ctypes, sqlite3, os
from tinygrad.runtime.autogen import rocprof

# ── RGP file parsing ───────────────────────────────────────────────────
SQTT_FILE_MAGIC = 0x50303042

def parse_rgp_sqtt(path: str):
    """Extract SQTT data blobs and ASIC info from an .rgp file."""
    with open(path, 'rb') as f:
        data = f.read()

    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == SQTT_FILE_MAGIC, f"Bad magic: {hex(magic)}"
    chunk_offset = struct.unpack_from('<I', data, 16)[0]

    # Parse ASIC info for clock frequency
    gpu_clock_hz = 0
    sqtt_blobs = []  # (se_index, blob_bytes)
    sqtt_version = 0

    pos = chunk_offset
    while pos + 16 <= len(data):
        raw_id, minor, major, size = struct.unpack_from('<I2Hi', data, pos)
        ctype = raw_id & 0xFF
        cindex = (raw_id >> 8) & 0xFF

        if size <= 0 or pos + size > len(data):
            break

        doff = pos + 16  # data starts after 16-byte chunk header

        if ctype == 0:  # ASIC_INFO
            if doff + 24 <= len(data):
                gpu_clock_hz = struct.unpack_from('<Q', data, doff + 8)[0]

        elif ctype == 1:  # SQTT_DESC
            if doff + 16 <= len(data):
                se_idx = struct.unpack_from('<I', data, doff)[0]  # shader_engine_index
                sqtt_version = struct.unpack_from('<I', data, doff + 4)[0]

        elif ctype == 2:  # SQTT_DATA
            # Sub-header: absolute_offset(u32), data_size(u32)
            if doff + 8 <= len(data):
                abs_offset, actual_sz = struct.unpack_from('<II', data, doff)
                if abs_offset + actual_sz <= len(data) and actual_sz > 0:
                    blob = data[abs_offset:abs_offset + actual_sz]
                    sqtt_blobs.append((cindex, blob))
                else:
                    # Fallback: data starts right after sub-header
                    blob = data[doff + 8:pos + size]
                    if len(blob) > 0:
                        sqtt_blobs.append((cindex, blob))

        pos += size

    return gpu_clock_hz, sqtt_blobs, sqtt_version


def decode_sqtt_blobs(blobs, gpu_clock_hz):
    """Use AMD's rocprofiler decoder to parse SQTT token streams."""
    waves = []
    occ_events = []
    realtime_pairs = []

    blob_iter = iter(blobs)

    @rocprof.rocprof_trace_decoder_se_data_callback_t
    def copy_cb(buf, buf_size, _):
        try:
            se_idx, blob = next(blob_iter)
        except StopIteration:
            return 0
        arr = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        buf[0] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_ubyte))
        buf_size[0] = len(blob)
        copy_cb._refs.append(arr)
        return len(blob)
    copy_cb._refs = []

    @rocprof.rocprof_trace_decoder_trace_callback_t
    def trace_cb(record_type, events_ptr, n, _):
        if record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
            for ev in (rocprof.rocprofiler_thread_trace_decoder_wave_t * n).from_address(events_ptr):
                waves.append({
                    'wave_id': ev.wave_id,
                    'cu': ev.cu,
                    'simd': ev.simd,
                    'begin_time': ev.begin_time,
                    'end_time': ev.end_time,
                    'duration_ticks': ev.end_time - ev.begin_time if ev.end_time > ev.begin_time else 0,
                    'instructions': ev.instructions_size,
                })
        elif record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
            for ev in (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr):
                occ_events.append({
                    'wave_id': ev.wave_id,
                    'cu': ev.cu,
                    'simd': ev.simd,
                    'time': ev.time,
                    'start': ev.start,
                })
        elif record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME:
            for ev in (rocprof.rocprofiler_thread_trace_decoder_realtime_t * n).from_address(events_ptr):
                realtime_pairs.append((ev.shader_clock, ev.realtime_clock))
        return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

    @rocprof.rocprof_trace_decoder_isa_callback_t
    def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, _):
        name = b"s_nop"
        if size_ptr[0] > 0:
            sz = min(len(name), size_ptr[0] - 1)
            ctypes.memmove(instr_ptr, name, sz)
            size_ptr[0] = sz
        else:
            size_ptr[0] = 0
        mem_size_ptr[0] = 4  # assume 4-byte instruction
        return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

    # Single-call API: parse_data drives copy_cb → trace_cb callbacks
    import threading
    exc = None
    def worker():
        nonlocal exc
        try:
            rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)
        except Exception as e:
            exc = e
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=30)
    if exc is not None:
        print(f"Decoder error: {exc}")
    if t.is_alive():
        print("Decoder timed out (30s)")

    # Convert ticks to nanoseconds
    if gpu_clock_hz > 0:
        tick_ns = 1e9 / gpu_clock_hz
        for w in waves:
            w['duration_ns'] = int(w['duration_ticks'] * tick_ns)
            w['duration_us'] = w['duration_ticks'] * tick_ns / 1000

    return waves, occ_events, realtime_pairs


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.rgp> [-o output.db]")
        sys.exit(1)

    rgp_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 3 and sys.argv[2] == '-o' else rgp_path.replace('.rgp', '_sqtt.db')
    if '-o' in sys.argv:
        db_path = sys.argv[sys.argv.index('-o') + 1]

    print(f"Parsing {rgp_path}...")
    gpu_clock_hz, blobs, sqtt_ver = parse_rgp_sqtt(rgp_path)
    print(f"GPU clock: {gpu_clock_hz/1e6:.0f} MHz")
    print(f"SQTT blobs: {len(blobs)} (shader engines)")
    for i, (se, blob) in enumerate(blobs):
        print(f"  SE{se}: {len(blob)} bytes")

    if not blobs:
        print("No SQTT data found!")
        sys.exit(1)

    print("\nDecoding SQTT tokens...")
    waves, occ, rt = decode_sqtt_blobs(blobs, gpu_clock_hz)
    print(f"Decoded: {len(waves)} wave executions, {len(occ)} occupancy events, {len(rt)} realtime pairs")

    if waves:
        durations = [w['duration_us'] for w in waves if w.get('duration_us', 0) > 0]
        if durations:
            print(f"\nWave execution timing:")
            print(f"  Count: {len(durations)}")
            print(f"  Mean:  {sum(durations)/len(durations):.2f} μs")
            print(f"  Min:   {min(durations):.2f} μs")
            print(f"  Max:   {max(durations):.2f} μs")
            print(f"  Total: {sum(durations)/1000:.2f} ms")

    # Write to SQLite
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS wave_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wave_id INTEGER, cu INTEGER, simd INTEGER,
        begin_time INTEGER, end_time INTEGER,
        duration_ticks INTEGER, duration_ns INTEGER, duration_us REAL,
        instructions INTEGER
    )""")
    for w in waves:
        c.execute("INSERT INTO wave_executions (wave_id, cu, simd, begin_time, end_time, duration_ticks, duration_ns, duration_us, instructions) VALUES (?,?,?,?,?,?,?,?,?)",
                  (w['wave_id'], w['cu'], w['simd'], w['begin_time'], w['end_time'],
                   w['duration_ticks'], w.get('duration_ns', 0), w.get('duration_us', 0),
                   w['instructions']))

    c.execute("""CREATE TABLE IF NOT EXISTS occupancy_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wave_id INTEGER, cu INTEGER, simd INTEGER,
        time INTEGER, start INTEGER
    )""")
    for o in occ:
        c.execute("INSERT INTO occupancy_events (wave_id, cu, simd, time, start) VALUES (?,?,?,?,?)",
                  (o['wave_id'], o['cu'], o['simd'], o['time'], o['start']))

    conn.commit()
    conn.close()
    print(f"\nSQLite written: {db_path}")


if __name__ == "__main__":
    main()
