#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["sqlite3"]
# ///
"""Parse RADV-generated .rgp files (SQTT format) and dump to SQLite.

Usage: python3 scripts/rgp2sqlite.py /tmp/foo.rgp [-o /tmp/foo.db]
"""
import argparse, struct, sqlite3, sys, os
from dataclasses import dataclass

# ── constants from sqtt.h ──────────────────────────────────────────────
SQTT_FILE_MAGIC = 0x50303042  # "B00P"

CHUNK_ASIC_INFO              = 0
CHUNK_SQTT_DESC              = 1
CHUNK_SQTT_DATA              = 2
CHUNK_API_INFO               = 3
CHUNK_QUEUE_EVENT_TIMINGS    = 5
CHUNK_CLOCK_CALIBRATION      = 6
CHUNK_CPU_INFO               = 7
CHUNK_SPM_DB                 = 8
CHUNK_CODE_OBJECT_DATABASE   = 9
CHUNK_CODE_OBJECT_LOADER     = 10
CHUNK_PSO_CORRELATION        = 11

CHUNK_NAMES = {
    0: "ASIC_INFO", 1: "SQTT_DESC", 2: "SQTT_DATA", 3: "API_INFO",
    4: "RESERVED", 5: "QUEUE_EVENT_TIMINGS", 6: "CLOCK_CALIBRATION",
    7: "CPU_INFO", 8: "SPM_DB", 9: "CODE_OBJECT_DATABASE",
    10: "CODE_OBJECT_LOADER_EVENTS", 11: "PSO_CORRELATION",
    12: "INSTRUMENTATION_TABLE",
}

# Queue event types (from RADV source)
QUEUE_EVENT_NAMES = {
    0: "CmdDraw", 1: "CmdDrawIndexed", 2: "CmdDrawIndirect",
    3: "CmdDrawIndexedIndirect", 4: "CmdDrawIndirectCountAMD",
    5: "CmdDrawIndexedIndirectCountAMD", 6: "CmdDispatch",
    7: "CmdDispatchIndirect", 8: "CmdCopyBuffer",
    9: "CmdCopyImage", 10: "CmdBlitImage", 11: "CmdCopyBufferToImage",
    12: "CmdCopyImageToBuffer", 13: "CmdUpdateBuffer",
    14: "CmdFillBuffer", 15: "CmdClearColorImage",
    16: "CmdClearDepthStencilImage", 17: "CmdClearAttachments",
    18: "CmdResolveImage", 19: "CmdWaitEvents", 20: "CmdPipelineBarrier",
    21: "CmdResetQueryPool", 22: "CmdCopyQueryPoolResults",
    23: "InternalUnknown",
}

# ── file header: 56 bytes ──────────────────────────────────────────────
# uint32 magic, version_major, version_minor, flags, chunk_offset
# int32  second, minute, hour, day_in_month, month, year, day_in_week, day_in_year, is_daylight
FILE_HDR = struct.Struct("<5I9i")

# ── chunk header: 16 bytes ─────────────────────────────────────────────
# chunk_id (uint32: type:8 | index:8 | pad:16), minor:u16, major:u16, size:i32, pad:i32
CHUNK_HDR = struct.Struct("<I2Hi2x")  # 14 bytes + 2 pad = 16


@dataclass
class ChunkInfo:
    chunk_type: int
    index: int
    major: int
    minor: int
    size: int
    offset: int  # offset of chunk header in file
    data_offset: int  # offset of data after header


def parse_file_header(data: bytes):
    vals = FILE_HDR.unpack_from(data, 0)
    magic = vals[0]
    if magic != SQTT_FILE_MAGIC:
        raise ValueError(f"Bad magic: {hex(magic)} (expected {hex(SQTT_FILE_MAGIC)})")
    return {
        "magic": magic,
        "version_major": vals[1],
        "version_minor": vals[2],
        "flags": vals[3],
        "chunk_offset": vals[4],
        "timestamp": f"{vals[9]+1900}-{vals[8]+1:02d}-{vals[7]:02d} {vals[6]:02d}:{vals[5]:02d}:{vals[5]:02d}",
    }


def parse_chunks(data: bytes, chunk_offset: int) -> list[ChunkInfo]:
    """Walk the chunk list starting at chunk_offset."""
    chunks = []
    pos = chunk_offset
    while pos + CHUNK_HDR.size <= len(data):
        raw_id, minor, major, size = CHUNK_HDR.unpack_from(data, pos)
        ctype = raw_id & 0xFF
        cindex = (raw_id >> 8) & 0xFF

        if size <= 0 or pos + size > len(data):
            break

        chunks.append(ChunkInfo(
            chunk_type=ctype, index=cindex, major=major, minor=minor,
            size=size, offset=pos, data_offset=pos + CHUNK_HDR.size,
        ))
        pos += size
    return chunks


def parse_queue_event_timings(data: bytes, chunk: ChunkInfo) -> list[dict]:
    """Parse QUEUE_EVENT_TIMINGS chunk.

    Layout after chunk header (16 bytes):
      uint32 queue_info_table_record_count
      uint32 queue_info_table_size
      uint32 queue_event_table_record_count
      uint32 queue_event_table_size
    Then: queue_info records, then queue_event records.

    Queue event record (32 bytes on RADV):
      uint32 event_type (enum)
      uint32 sqtt_cb_id
      uint64 frame_index
      uint32 queue_info_index
      uint32 submit_index
      uint64 cpu_timestamp
      uint64 (reserved/padding -- sometimes pre/post timestamps)

    This varies by RGP version. We'll try a few layouts.
    """
    events = []
    off = chunk.data_offset

    qi_count, qi_size, qe_count, qe_size = struct.unpack_from("<4I", data, off)
    off += 16

    # Skip queue info table
    off += qi_size

    # Parse event table
    # Try to infer record size
    if qe_count > 0 and qe_size > 0:
        rec_size = qe_size // qe_count if qe_count else 0
    else:
        return events

    for i in range(qe_count):
        rec_off = off + i * rec_size
        if rec_off + 24 > len(data):
            break
        # Minimal parse: event_type(u32), sqtt_cb_id(u32), frame_index(u64)
        etype, cb_id, frame_idx = struct.unpack_from("<2IQ", data, rec_off)
        ev = {
            "index": i,
            "event_type": etype,
            "event_name": QUEUE_EVENT_NAMES.get(etype, f"Unknown({etype})"),
            "cb_id": cb_id,
            "frame_index": frame_idx,
        }
        # If record is big enough, grab more fields
        if rec_size >= 32:
            qi_idx, sub_idx = struct.unpack_from("<2I", data, rec_off + 16)
            ev["queue_info_index"] = qi_idx
            ev["submit_index"] = sub_idx
        if rec_size >= 40:
            cpu_ts, = struct.unpack_from("<Q", data, rec_off + 24)
            ev["cpu_timestamp"] = cpu_ts
        if rec_size >= 56 and rec_off + 56 <= len(data):
            pre_ts, post_ts = struct.unpack_from("<2Q", data, rec_off + 40)
            ev["pre_timestamp"] = pre_ts
            ev["post_timestamp"] = post_ts
            if pre_ts > 0 and post_ts > pre_ts:
                ev["duration_ns"] = post_ts - pre_ts
        events.append(ev)
    return events


def parse_clock_calibration(data: bytes, chunk: ChunkInfo) -> dict:
    """Parse CLOCK_CALIBRATION chunk for GPU↔CPU clock mapping."""
    off = chunk.data_offset
    # cpu_timestamp(u64), gpu_timestamp(u64)
    if off + 16 <= len(data):
        cpu_ts, gpu_ts = struct.unpack_from("<2Q", data, off)
        return {"cpu_timestamp": cpu_ts, "gpu_timestamp": gpu_ts}
    return {}


def parse_asic_info(data: bytes, chunk: ChunkInfo) -> dict:
    """Parse ASIC_INFO — extract key GPU info."""
    off = chunk.data_offset
    info = {}
    if off + 8 <= len(data):
        flags, trace_shader_core_clock = struct.unpack_from("<2I", data, off)
        info["flags"] = flags
        info["trace_shader_core_clock_mhz"] = trace_shader_core_clock
    if off + 16 <= len(data):
        trace_mem_clock, = struct.unpack_from("<I", data, off + 8)
        info["trace_memory_clock_mhz"] = trace_mem_clock
    # Device ID at offset 16
    if off + 20 <= len(data):
        dev_id, dev_rev = struct.unpack_from("<2I", data, off + 12)
        info["device_id"] = hex(dev_id)
        info["device_revision"] = dev_rev
    return info


def parse_code_object_db(data: bytes, chunk: ChunkInfo) -> list[dict]:
    """Parse CODE_OBJECT_DATABASE — shader ELF blobs with hashes."""
    records = []
    off = chunk.data_offset
    if off + 8 > len(data):
        return records
    record_offset, record_count = struct.unpack_from("<2I", data, off)
    # Records follow after the chunk-specific header
    # Each record: uint32 size, then ELF blob
    rec_off = chunk.offset + record_offset
    for i in range(record_count):
        if rec_off + 4 > len(data):
            break
        rec_size, = struct.unpack_from("<I", data, rec_off)
        blob_start = rec_off + 4
        blob_end = rec_off + 4 + rec_size
        if blob_end > len(data):
            break
        # Check for ELF magic
        is_elf = data[blob_start:blob_start+4] == b'\x7fELF' if blob_start + 4 <= len(data) else False
        records.append({
            "index": i,
            "size": rec_size,
            "is_elf": is_elf,
            "offset": blob_start,
        })
        rec_off = blob_end
        # Align to 4 bytes
        rec_off = (rec_off + 3) & ~3
    return records


def write_sqlite(db_path: str, file_hdr: dict, chunks: list[ChunkInfo],
                 events: list[dict], clocks: list[dict], asic: dict,
                 code_objects: list[dict], raw_data: bytes):
    """Write parsed data to SQLite."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS file_info (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    for k, v in file_hdr.items():
        c.execute("INSERT INTO file_info VALUES (?, ?)", (k, str(v)))
    for k, v in asic.items():
        c.execute("INSERT OR REPLACE INTO file_info VALUES (?, ?)", (f"asic_{k}", str(v)))

    c.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        chunk_type INTEGER,
        chunk_name TEXT,
        chunk_index INTEGER,
        major INTEGER, minor INTEGER,
        size INTEGER,
        offset INTEGER
    )""")
    for i, ch in enumerate(chunks):
        c.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                  (i, ch.chunk_type, CHUNK_NAMES.get(ch.chunk_type, "UNKNOWN"),
                   ch.index, ch.major, ch.minor, ch.size, ch.offset))

    c.execute("""CREATE TABLE IF NOT EXISTS queue_events (
        id INTEGER PRIMARY KEY,
        event_type INTEGER,
        event_name TEXT,
        cb_id INTEGER,
        frame_index INTEGER,
        queue_info_index INTEGER,
        submit_index INTEGER,
        cpu_timestamp INTEGER,
        pre_timestamp INTEGER,
        post_timestamp INTEGER,
        duration_ns INTEGER
    )""")
    for ev in events:
        c.execute("INSERT INTO queue_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (ev["index"], ev["event_type"], ev["event_name"],
                   ev.get("cb_id"), ev.get("frame_index"),
                   ev.get("queue_info_index"), ev.get("submit_index"),
                   ev.get("cpu_timestamp"),
                   ev.get("pre_timestamp"), ev.get("post_timestamp"),
                   ev.get("duration_ns")))

    c.execute("""CREATE TABLE IF NOT EXISTS clock_calibration (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cpu_timestamp INTEGER,
        gpu_timestamp INTEGER
    )""")
    for cl in clocks:
        c.execute("INSERT INTO clock_calibration (cpu_timestamp, gpu_timestamp) VALUES (?,?)",
                  (cl.get("cpu_timestamp"), cl.get("gpu_timestamp")))

    c.execute("""CREATE TABLE IF NOT EXISTS code_objects (
        id INTEGER PRIMARY KEY,
        size INTEGER,
        is_elf INTEGER,
        blob_offset INTEGER
    )""")
    for co in code_objects:
        c.execute("INSERT INTO code_objects VALUES (?,?,?,?)",
                  (co["index"], co["size"], int(co["is_elf"]), co["offset"]))

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Parse .rgp (SQTT) file to SQLite")
    parser.add_argument("rgp_file", help="Path to .rgp file")
    parser.add_argument("-o", "--output", help="Output SQLite path (default: <input>.db)")
    parser.add_argument("--dump-chunks", action="store_true", help="Print chunk summary and exit")
    args = parser.parse_args()

    with open(args.rgp_file, "rb") as f:
        data = f.read()

    print(f"File: {args.rgp_file} ({len(data)} bytes)")

    file_hdr = parse_file_header(data)
    print(f"Version: {file_hdr['version_major']}.{file_hdr['version_minor']}")
    print(f"Chunk offset: {file_hdr['chunk_offset']}")

    chunks = parse_chunks(data, file_hdr["chunk_offset"])
    print(f"Chunks found: {len(chunks)}")

    for ch in chunks:
        name = CHUNK_NAMES.get(ch.chunk_type, f"UNKNOWN({ch.chunk_type})")
        print(f"  [{ch.index:2d}] {name:30s} v{ch.major}.{ch.minor}  size={ch.size:>8d}  off={ch.offset}")

    if args.dump_chunks:
        return

    # Parse specific chunk types
    events = []
    clocks = []
    asic = {}
    code_objects = []

    for ch in chunks:
        if ch.chunk_type == CHUNK_QUEUE_EVENT_TIMINGS:
            events.extend(parse_queue_event_timings(data, ch))
        elif ch.chunk_type == CHUNK_CLOCK_CALIBRATION:
            cl = parse_clock_calibration(data, ch)
            if cl:
                clocks.append(cl)
        elif ch.chunk_type == CHUNK_ASIC_INFO:
            asic = parse_asic_info(data, ch)
        elif ch.chunk_type == CHUNK_CODE_OBJECT_DATABASE:
            code_objects.extend(parse_code_object_db(data, ch))

    print(f"\nQueue events: {len(events)}")
    print(f"Clock calibrations: {len(clocks)}")
    print(f"Code objects: {len(code_objects)}")

    if events:
        # Summary
        by_type = {}
        for ev in events:
            n = ev["event_name"]
            by_type.setdefault(n, []).append(ev)
        print("\nEvent summary:")
        for name, evs in sorted(by_type.items(), key=lambda x: -len(x[1])):
            durations = [e["duration_ns"] for e in evs if e.get("duration_ns")]
            if durations:
                avg = sum(durations) / len(durations)
                total = sum(durations)
                print(f"  {name:30s}: {len(evs):5d} events, avg={avg/1000:.1f}μs, total={total/1e6:.2f}ms")
            else:
                print(f"  {name:30s}: {len(evs):5d} events (no timing)")

    # Write SQLite
    db_path = args.output or args.rgp_file.replace(".rgp", ".db")
    write_sqlite(db_path, file_hdr, chunks, events, clocks, asic, code_objects, data)
    print(f"\nSQLite written: {db_path}")


if __name__ == "__main__":
    main()
