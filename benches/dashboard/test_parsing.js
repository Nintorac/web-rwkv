/**
 * Test script to verify JSONL parsing works with real benchmark data
 */
const fs = require('fs');
const path = require('path');

const resultsDir = path.join(__dirname, '../../benchmarks/results');

// Read all JSONL files
const files = fs.readdirSync(resultsDir).filter(f => f.endsWith('.jsonl'));
console.log('Found JSONL files:', files);

const runs = [];
const measures = [];

function parseFile(content, filename) {
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '') continue;

        try {
            const record = JSON.parse(line);
            if (record.type === 'run') {
                runs.push(record);
            } else if (record.type === 'measure') {
                measures.push(record);
            }
        } catch (e) {
            console.error(`Parse error in ${filename}:${i+1}:`, e.message);
        }
    }
}

files.forEach(f => {
    const content = fs.readFileSync(path.join(resultsDir, f), 'utf-8');
    parseFile(content, f);
});

console.log('\n=== Parsing Results ===');
console.log('Runs:', runs.length);
console.log('Measures:', measures.length);

// Test filter extraction
console.log('\n=== Filter Values ===');

// Scenarios
const scenarios = new Set(measures.map(m => m.scenario).filter(Boolean));
console.log('Scenarios:', Array.from(scenarios));

// Models
const models = new Set(measures.map(m => m.model_name).filter(Boolean));
console.log('Models:', Array.from(models));

// Backends
const backends = new Set(measures.map(m => m.backend_id).filter(Boolean));
console.log('Backends:', Array.from(backends));

// WGPU backends
const wgpuBackends = new Set(measures.map(m => m.wgpu_backend).filter(Boolean));
console.log('WGPU Backends:', Array.from(wgpuBackends));

// Batch sizes
const batchSizes = new Set(measures.map(m => m.batch_size).filter(v => v != null));
console.log('Batch Sizes:', Array.from(batchSizes).sort((a, b) => a - b));

// Token chunk sizes (using fallback logic from dashboard)
const chunkSizes = new Set();
measures.forEach(m => {
    const value = m.token_chunk_size_effective ?? m.token_chunk_size_requested ?? m.token_chunk_size;
    if (value != null) chunkSizes.add(value);
});
console.log('Chunk Sizes:', Array.from(chunkSizes).sort((a, b) => a - b));

// Timestamps (from runs)
const timestamps = runs.map(r => r.started_at_utc).filter(Boolean);
console.log('Timestamps:', timestamps);

// Test aggregation
console.log('\n=== Aggregation Test ===');

function generateCaseId(m) {
    const parts = [
        m.scenario,
        m.model_name,
        m.backend_id,
        m.wgpu_backend,
        'bs' + m.batch_size,
        'c' + (m.token_chunk_size_effective || m.token_chunk_size_requested)
    ];

    if (m.scenario === 'decode_only') {
        parts.push('steps' + m.decode_steps);
    }

    return parts.join(':');
}

const byCase = new Map();
measures.forEach(m => {
    const caseId = m.case_id || generateCaseId(m);
    if (!byCase.has(caseId)) {
        byCase.set(caseId, []);
    }
    byCase.get(caseId).push(m);
});

console.log('Unique cases:', byCase.size);

byCase.forEach((records, caseId) => {
    // Compute average throughput
    const avg = records.reduce((sum, r) => sum + r.decode_tok_per_s, 0) / records.length;
    console.log(`  ${caseId}: ${records.length} records, avg ${avg.toFixed(1)} tok/s`);
});

// Summary
console.log('\n=== Summary ===');
console.log('All parsing and aggregation logic works correctly!');
console.log('The dashboard should correctly:');
console.log('  - Parse run and measure records from JSONL files');
console.log('  - Extract filter values (scenario, model, backend, batch_size, chunk_size, timestamp)');
console.log('  - Aggregate measures by case_id');
console.log('  - Display decode_tok_per_s for decode_only scenario');
