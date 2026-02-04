/**
 * web-rwkv Benchmark Dashboard
 * Main application logic
 */

(function() {
    'use strict';

    // Application state
    const state = {
        data: {
            runs: [],      // Run header records (keyed by run_id)
            measures: [],  // Measure records
            loaded: false,
            // Track source files for each record
            sourceFiles: new Map()  // run_id -> filename
        },
        // Multi-select filter state: each key maps to Set of selected values (null = all)
        filters: {
            scenario: null,
            model_name: null,
            model_size: null,
            backend_id: null,
            wgpu_backend: null,
            batch_size: null,
            token_chunk_size: null,
            seq_len: null,
            mixed_case_id: null,
            run_id: null,
            git_sha: null,
            timestamp: null  // Will store {min, max} range if set
        },
        // Available values for each filter dimension (extracted from data)
        filterOptions: {
            scenario: [],
            model_name: [],
            model_size: [],
            backend_id: [],
            wgpu_backend: [],
            batch_size: [],
            token_chunk_size: [],
            seq_len: [],
            mixed_case_id: [],
            run_id: [],
            git_sha: [],
            timestamp: []
        },
        currentView: 'table',
        loading: false,
        errors: []  // Parse errors from loading
    };

    // Configuration
    const config = {
        // If true, merge new data with existing; if false, replace
        mergeOnLoad: true
    };

    // DOM element references
    let elements = {};

    /**
     * Initialize the dashboard application
     */
    function init() {
        console.log('[dashboard] Initializing web-rwkv benchmark dashboard v0.1');

        // Cache DOM elements
        cacheElements();

        // Set up event listeners
        setupDropZone();
        setupTabs();
        setupExportButton();
        setupServerFiles();

        console.log('[dashboard] Initialization complete');
    }

    /**
     * Cache frequently accessed DOM elements
     */
    function cacheElements() {
        elements = {
            dropZone: document.getElementById('drop-zone'),
            fileInput: document.getElementById('file-input'),
            filterGroups: document.getElementById('filter-groups'),
            viewContainer: document.getElementById('view-container'),
            viewTabs: document.getElementById('view-tabs'),
            stats: {
                runs: document.getElementById('stat-runs'),
                cases: document.getElementById('stat-cases'),
                backends: document.getElementById('stat-backends'),
                models: document.getElementById('stat-models')
            },
            exportBtn: document.getElementById('export-btn'),
            // Server files section
            serverFilesSection: document.getElementById('server-files-section'),
            serverFilesContent: document.getElementById('server-files-content'),
            serverFilesStatus: document.getElementById('server-files-status'),
            serverRefreshBtn: document.getElementById('server-refresh-btn')
        };
    }

    /**
     * Set up drag-and-drop file loading
     */
    function setupDropZone() {
        const dropZone = elements.dropZone;
        const fileInput = elements.fileInput;

        // Click to browse
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFiles(e.target.files);
            }
        });

        // Drag events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');

            if (e.dataTransfer.files.length > 0) {
                handleFiles(e.dataTransfer.files);
            }
        });

        console.log('[dashboard] Drop zone configured');
    }

    /**
     * Set up view tab switching
     */
    function setupTabs() {
        const tabs = elements.viewTabs.querySelectorAll('.tab-button');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const view = tab.dataset.view;
                switchView(view);

                // Update active state
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
            });
        });

        console.log('[dashboard] View tabs configured');
    }

    /**
     * Set up export button
     */
    function setupExportButton() {
        const exportBtn = elements.exportBtn;
        if (!exportBtn) {
            console.warn('[dashboard] Export button not found');
            return;
        }

        exportBtn.addEventListener('click', () => {
            exportToJsonl();
        });

        console.log('[dashboard] Export button configured');
    }

    // =========================================================================
    // Server Files Loading
    // =========================================================================

    /**
     * Server files state
     */
    const serverFilesState = {
        files: [],           // Array of file objects from manifest
        loading: false,      // Whether currently loading manifest
        error: null,         // Last error message
        manifestPath: '/data/index.json'  // Default manifest path
    };

    /**
     * Set up server files section
     */
    function setupServerFiles() {
        const refreshBtn = elements.serverRefreshBtn;
        if (!refreshBtn) {
            console.warn('[dashboard] Server refresh button not found');
            return;
        }

        refreshBtn.addEventListener('click', () => {
            fetchServerFiles();
        });

        // Try to auto-load manifest on init (gracefully fail if not available)
        fetchServerFiles();

        console.log('[dashboard] Server files section configured');
    }

    /**
     * Fetch the server manifest listing available JSONL files
     */
    async function fetchServerFiles() {
        if (serverFilesState.loading) return;

        console.log('[dashboard] Fetching server file manifest');
        serverFilesState.loading = true;
        serverFilesState.error = null;
        updateServerFilesUI();

        try {
            const response = await fetch(serverFilesState.manifestPath);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const manifest = await response.json();

            // Validate manifest structure
            if (!manifest.files || !Array.isArray(manifest.files)) {
                throw new Error('Invalid manifest: missing "files" array');
            }

            serverFilesState.files = manifest.files;
            serverFilesState.error = null;

            console.log(`[dashboard] Loaded manifest: ${manifest.files.length} file(s) available`);

        } catch (err) {
            console.warn('[dashboard] Failed to fetch server manifest:', err.message);
            serverFilesState.files = [];
            serverFilesState.error = err.message;
        } finally {
            serverFilesState.loading = false;
            updateServerFilesUI();
        }
    }

    /**
     * Load a single file from the server
     * @param {string} filename - Name of the file to load
     */
    async function loadServerFile(filename) {
        console.log(`[dashboard] Loading server file: ${filename}`);

        // Reset errors for this load session
        state.errors = [];

        // Show loading state
        setLoadingState(true);

        // If not merging, clear existing data before loading
        if (!config.mergeOnLoad) {
            clearData();
        }

        try {
            // Construct the full path (files are in /data/ directory)
            const filePath = `/data/${filename}`;
            const response = await fetch(filePath);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const content = await response.text();
            const parseResult = parseJsonlContent(content, filename);

            // Merge results into state
            let runsAdded = 0;
            let measuresAdded = 0;

            parseResult.runs.forEach(run => {
                // Check for duplicate run_id
                const existingIdx = state.data.runs.findIndex(r => r.run_id === run.run_id);
                if (existingIdx === -1) {
                    state.data.runs.push(run);
                    state.data.sourceFiles.set(run.run_id, filename);
                    runsAdded++;
                } else {
                    console.log(`[dashboard] Skipping duplicate run: ${run.run_id}`);
                }
            });

            parseResult.measures.forEach(measure => {
                measure._sourceFile = filename;
                state.data.measures.push(measure);
                measuresAdded++;
            });

            state.errors.push(...parseResult.errors);

            console.log(`[dashboard] Loaded ${filename}: ${runsAdded} runs, ${measuresAdded} measures`);

            // Mark data as loaded if we have any records
            if (state.data.runs.length > 0 || state.data.measures.length > 0) {
                state.data.loaded = true;
            }

            // Show error summary if there were parsing errors
            if (parseResult.errors.length > 0) {
                showErrorSummary(parseResult.errors);
            }

        } catch (err) {
            const error = {
                file: filename,
                line: null,
                message: `Failed to load: ${err.message}`
            };
            state.errors.push(error);
            console.error(`[dashboard] Error loading ${filename}:`, err);
            showErrorSummary([error]);
        } finally {
            setLoadingState(false);
            updateFilters();
            render();
        }
    }

    /**
     * Update the server files UI based on current state
     */
    function updateServerFilesUI() {
        const content = elements.serverFilesContent;
        const status = elements.serverFilesStatus;
        if (!content) return;

        if (serverFilesState.loading) {
            content.innerHTML = '<p class="server-files-status">Loading...</p>';
            return;
        }

        if (serverFilesState.error) {
            // Show error with helpful message
            let errorHtml = `<p class="server-files-status server-files-error">`;
            if (serverFilesState.error.includes('404')) {
                errorHtml += `No manifest found. Create <code>data/index.json</code> to enable server loading.`;
            } else if (serverFilesState.error.includes('Failed to fetch')) {
                errorHtml += `Server not available. Use drag-and-drop instead.`;
            } else {
                errorHtml += `Error: ${escapeHtml(serverFilesState.error)}`;
            }
            errorHtml += `</p>`;
            content.innerHTML = errorHtml;
            return;
        }

        if (serverFilesState.files.length === 0) {
            content.innerHTML = '<p class="server-files-status">No files available</p>';
            return;
        }

        // Build file list
        let html = '<ul class="server-files-list">';
        serverFilesState.files.forEach(file => {
            const name = file.name || 'unknown';
            const size = file.size ? formatBytes(file.size) : '';
            const modified = file.modified ? formatDate(file.modified) : '';

            html += `
                <li class="server-file-item" data-filename="${escapeHtml(name)}">
                    <button class="server-file-btn" title="Click to load">
                        <span class="server-file-name">${escapeHtml(name)}</span>
                        <span class="server-file-meta">${size}${modified ? ' - ' + modified : ''}</span>
                    </button>
                </li>
            `;
        });
        html += '</ul>';

        content.innerHTML = html;

        // Attach click handlers
        content.querySelectorAll('.server-file-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const item = btn.closest('.server-file-item');
                const filename = item.dataset.filename;
                loadServerFile(filename);
            });
        });
    }

    /**
     * Format a date string for display
     * @param {string} dateStr - ISO date string
     * @returns {string} Formatted date
     */
    function formatDate(dateStr) {
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch {
            return dateStr;
        }
    }

    /**
     * Export filtered data as JSONL file
     * Format: run headers first (deduplicated), then measure records
     */
    function exportToJsonl() {
        console.log('[dashboard] Exporting filtered data to JSONL');

        // Get filtered measures
        const filteredMeasures = applyFilters();

        if (filteredMeasures.length === 0) {
            console.warn('[dashboard] No data to export');
            alert('No data to export. Adjust filters or load benchmark data first.');
            return;
        }

        // Collect unique run_ids from filtered measures
        const runIds = new Set(filteredMeasures.map(m => m.run_id).filter(Boolean));

        // Get corresponding run headers (deduplicated)
        const runHeaders = state.data.runs.filter(run => runIds.has(run.run_id));

        // Build JSONL content: run headers first, then measures
        const lines = [];

        // Add run headers (without internal tracking fields)
        runHeaders.forEach(run => {
            // Create a clean copy without internal fields
            const cleanRun = { ...run };
            lines.push(JSON.stringify(cleanRun));
        });

        // Add measure records (without internal tracking fields)
        filteredMeasures.forEach(measure => {
            // Create a clean copy without internal fields (e.g., _sourceFile)
            const cleanMeasure = {};
            for (const [key, value] of Object.entries(measure)) {
                if (!key.startsWith('_')) {
                    cleanMeasure[key] = value;
                }
            }
            lines.push(JSON.stringify(cleanMeasure));
        });

        const jsonlContent = lines.join('\n') + '\n';

        // Generate filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = `export-${timestamp}.jsonl`;

        // Trigger download
        downloadFile(jsonlContent, filename, 'application/x-ndjson');

        console.log(`[dashboard] Exported ${runHeaders.length} run headers and ${filteredMeasures.length} measures to ${filename}`);
    }

    /**
     * Trigger browser download of a file
     * @param {string} content - File content
     * @param {string} filename - Suggested filename
     * @param {string} mimeType - MIME type of the file
     */
    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';

        document.body.appendChild(a);
        a.click();

        // Cleanup
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Handle dropped/selected files
     * @param {FileList} files - Files to process
     */
    async function handleFiles(files) {
        const fileArray = Array.from(files);
        console.log(`[dashboard] Processing ${fileArray.length} file(s)`);

        if (fileArray.length === 0) return;

        // Reset errors for this load session
        state.errors = [];

        // Show loading state
        setLoadingState(true);

        // If not merging, clear existing data before loading
        if (!config.mergeOnLoad) {
            clearData();
        }

        // Process all files
        const results = await Promise.all(
            fileArray.map(file => loadJsonlFile(file))
        );

        // Aggregate results
        let totalRuns = 0;
        let totalMeasures = 0;
        let totalErrors = 0;

        results.forEach(result => {
            totalRuns += result.runsAdded;
            totalMeasures += result.measuresAdded;
            totalErrors += result.errors.length;
        });

        console.log(`[dashboard] Load complete: ${totalRuns} runs, ${totalMeasures} measures, ${totalErrors} errors`);

        // Mark data as loaded if we have any records
        if (state.data.runs.length > 0 || state.data.measures.length > 0) {
            state.data.loaded = true;
        }

        // Update UI
        setLoadingState(false);
        updateFilters();  // Update filter options and UI when data changes
        render();

        // Show error summary if there were parsing errors
        if (totalErrors > 0) {
            showErrorSummary(state.errors);
        }
    }

    /**
     * Load and parse a single JSONL file
     * @param {File} file - File object to load
     * @returns {Promise<{runsAdded: number, measuresAdded: number, errors: Array}>}
     */
    async function loadJsonlFile(file) {
        console.log(`[dashboard] Loading file: ${file.name} (${formatBytes(file.size)})`);

        const result = {
            runsAdded: 0,
            measuresAdded: 0,
            errors: []
        };

        try {
            const content = await readFileAsText(file);
            const parseResult = parseJsonlContent(content, file.name);

            // Merge results into state
            parseResult.runs.forEach(run => {
                // Check for duplicate run_id
                const existingIdx = state.data.runs.findIndex(r => r.run_id === run.run_id);
                if (existingIdx === -1) {
                    state.data.runs.push(run);
                    state.data.sourceFiles.set(run.run_id, file.name);
                    result.runsAdded++;
                } else {
                    console.log(`[dashboard] Skipping duplicate run: ${run.run_id}`);
                }
            });

            parseResult.measures.forEach(measure => {
                // For measures, we allow duplicates from different files
                // but track them with their source
                measure._sourceFile = file.name;
                state.data.measures.push(measure);
                result.measuresAdded++;
            });

            result.errors = parseResult.errors;
            state.errors.push(...parseResult.errors);

        } catch (err) {
            const error = {
                file: file.name,
                line: null,
                message: `Failed to read file: ${err.message}`
            };
            result.errors.push(error);
            state.errors.push(error);
            console.error(`[dashboard] Error loading ${file.name}:`, err);
        }

        return result;
    }

    /**
     * Read file as text using FileReader API
     * @param {File} file - File to read
     * @returns {Promise<string>}
     */
    function readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsText(file);
        });
    }

    /**
     * Parse JSONL content line-by-line
     * @param {string} content - Raw JSONL content
     * @param {string} filename - Source filename for error reporting
     * @returns {{runs: Array, measures: Array, errors: Array}}
     */
    function parseJsonlContent(content, filename) {
        const runs = [];
        const measures = [];
        const errors = [];

        // Split by newlines and process line-by-line
        const lines = content.split('\n');

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            const lineNum = i + 1;

            // Skip empty lines
            if (line === '') continue;

            try {
                const record = JSON.parse(line);

                // Validate record has required fields
                if (!record.type) {
                    errors.push({
                        file: filename,
                        line: lineNum,
                        message: 'Missing "type" field'
                    });
                    continue;
                }

                if (!record.schema_version) {
                    errors.push({
                        file: filename,
                        line: lineNum,
                        message: 'Missing "schema_version" field'
                    });
                    continue;
                }

                // Route to appropriate collection based on type
                if (record.type === 'run') {
                    if (!record.run_id) {
                        errors.push({
                            file: filename,
                            line: lineNum,
                            message: 'Run record missing "run_id" field'
                        });
                        continue;
                    }
                    runs.push(record);
                } else if (record.type === 'measure') {
                    if (!record.run_id) {
                        errors.push({
                            file: filename,
                            line: lineNum,
                            message: 'Measure record missing "run_id" field'
                        });
                        continue;
                    }
                    measures.push(record);
                } else {
                    // Unknown type - log warning but don't fail
                    console.warn(`[dashboard] Unknown record type "${record.type}" at ${filename}:${lineNum}`);
                }

            } catch (parseErr) {
                // JSON parse error - continue processing other lines
                errors.push({
                    file: filename,
                    line: lineNum,
                    message: `Invalid JSON: ${parseErr.message}`
                });
            }
        }

        console.log(`[dashboard] Parsed ${filename}: ${runs.length} runs, ${measures.length} measures, ${errors.length} errors`);

        return { runs, measures, errors };
    }

    /**
     * Clear all loaded data
     */
    function clearData() {
        state.data.runs = [];
        state.data.measures = [];
        state.data.sourceFiles.clear();
        state.data.loaded = false;
        state.errors = [];

        // Reset all filter states
        for (const key of Object.keys(state.filters)) {
            state.filters[key] = null;
        }
        // Reset filter options
        for (const key of Object.keys(state.filterOptions)) {
            state.filterOptions[key] = [];
        }

        console.log('[dashboard] Data cleared');
    }

    /**
     * Set loading state and update UI
     * @param {boolean} loading - Whether currently loading
     */
    function setLoadingState(loading) {
        state.loading = loading;
        elements.dropZone.classList.toggle('loading', loading);

        if (loading) {
            elements.dropZone.querySelector('.drop-zone-text').textContent = 'Loading...';
        } else {
            elements.dropZone.querySelector('.drop-zone-text').textContent = 'Drop JSONL files here or click to browse';
        }
    }

    /**
     * Show error summary in the UI
     * @param {Array} errors - Array of error objects
     */
    function showErrorSummary(errors) {
        if (errors.length === 0) return;

        // Group errors by file
        const byFile = new Map();
        errors.forEach(err => {
            const key = err.file || 'unknown';
            if (!byFile.has(key)) byFile.set(key, []);
            byFile.get(key).push(err);
        });

        // Build summary message
        let summary = `Parsed with ${errors.length} error(s):\n`;
        byFile.forEach((fileErrors, filename) => {
            summary += `\n${filename}:\n`;
            // Show first 5 errors per file
            fileErrors.slice(0, 5).forEach(err => {
                const lineInfo = err.line ? `line ${err.line}: ` : '';
                summary += `  ${lineInfo}${err.message}\n`;
            });
            if (fileErrors.length > 5) {
                summary += `  ... and ${fileErrors.length - 5} more errors\n`;
            }
        });

        console.warn('[dashboard] Parse errors:', summary);

        // Update drop zone to show error count
        const hint = elements.dropZone.querySelector('.drop-zone-hint');
        hint.textContent = `Loaded with ${errors.length} parse error(s). Check console for details.`;
        hint.style.color = 'var(--warning)';

        // Reset hint after 5 seconds
        setTimeout(() => {
            hint.textContent = 'Supports multiple files';
            hint.style.color = '';
        }, 5000);
    }

    /**
     * Format bytes to human-readable string
     * @param {number} bytes - Number of bytes
     * @returns {string}
     */
    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    /**
     * Filter dimension configuration
     * Defines the order, labels, and data sources for each filter
     */
    const filterConfig = [
        { key: 'scenario', label: 'Scenario', source: 'measures' },
        { key: 'model_name', label: 'Model Name', source: 'measures' },
        { key: 'model_size', label: 'Model Size', source: 'measures' },
        { key: 'backend_id', label: 'Backend', source: 'measures' },
        { key: 'wgpu_backend', label: 'WGPU Backend', source: 'measures' },
        { key: 'batch_size', label: 'Batch Size', source: 'measures', numeric: true },
        { key: 'token_chunk_size', label: 'Chunk Size', source: 'measures', numeric: true },
        { key: 'seq_len', label: 'Seq Length', source: 'measures', numeric: true },
        { key: 'mixed_case_id', label: 'Mixed Case', source: 'measures' },
        { key: 'run_id', label: 'Run ID', source: 'runs' },
        { key: 'git_sha', label: 'Git SHA', source: 'runs' },
        { key: 'timestamp', label: 'Timestamp', source: 'runs' }
    ];

    /**
     * Extract unique values for all filter dimensions from loaded data
     */
    function extractFilterOptions() {
        const measures = state.data.measures;
        const runs = state.data.runs;

        // Reset filter options
        for (const key of Object.keys(state.filterOptions)) {
            state.filterOptions[key] = [];
        }

        // Extract from measures
        // Note: token_chunk_size uses token_chunk_size_effective (or token_chunk_size_requested as fallback)
        const measureFields = ['scenario', 'model_name', 'model_size', 'backend_id', 'wgpu_backend',
                               'batch_size', 'token_chunk_size', 'seq_len', 'mixed_case_id'];
        measureFields.forEach(field => {
            const values = new Set();
            measures.forEach(m => {
                // Special handling for token_chunk_size - check effective then requested
                let value;
                if (field === 'token_chunk_size') {
                    value = m.token_chunk_size_effective ?? m.token_chunk_size_requested ?? m.token_chunk_size;
                } else {
                    value = m[field];
                }
                if (value !== undefined && value !== null) {
                    values.add(value);
                }
            });
            state.filterOptions[field] = Array.from(values).sort((a, b) => {
                // Numeric sort for number values
                if (typeof a === 'number' && typeof b === 'number') {
                    return a - b;
                }
                return String(a).localeCompare(String(b));
            });
        });

        // Extract from runs
        // Note: timestamp filter uses started_at_utc field from the data
        const runFieldMappings = [
            { filterKey: 'run_id', dataKey: 'run_id' },
            { filterKey: 'git_sha', dataKey: 'git_sha' },
            { filterKey: 'timestamp', dataKey: 'started_at_utc' }  // Map timestamp filter to started_at_utc field
        ];
        runFieldMappings.forEach(({ filterKey, dataKey }) => {
            const values = new Set();
            runs.forEach(r => {
                if (r[dataKey] !== undefined && r[dataKey] !== null) {
                    values.add(r[dataKey]);
                }
            });
            state.filterOptions[filterKey] = Array.from(values).sort((a, b) => {
                // Reverse sort for timestamps (most recent first)
                if (filterKey === 'timestamp') {
                    return String(b).localeCompare(String(a));
                }
                return String(a).localeCompare(String(b));
            });
        });

        console.log('[dashboard] Filter options extracted:', Object.entries(state.filterOptions)
            .map(([k, v]) => `${k}: ${v.length}`).join(', '));
    }

    /**
     * Update filter UI based on loaded data
     */
    function updateFilters() {
        // Extract unique values for filter options
        extractFilterOptions();

        // Build filter UI
        renderFilterUI();

        console.log('[dashboard] Filters updated');
    }

    /**
     * Render the filter UI based on available options
     */
    function renderFilterUI() {
        const container = elements.filterGroups;

        // Check if we have any data
        const hasData = state.data.loaded && (state.data.measures.length > 0 || state.data.runs.length > 0);

        if (!hasData) {
            container.innerHTML = '<p class="placeholder-text">Load benchmark data to enable filters</p>';
            return;
        }

        // Build filter groups HTML
        let html = '';

        filterConfig.forEach(config => {
            const options = state.filterOptions[config.key];
            const selectedValues = state.filters[config.key];

            // Skip filters with no options
            if (options.length === 0) {
                return;
            }

            html += `
                <div class="filter-group" data-filter="${config.key}">
                    <label>${config.label}</label>
                    <div class="multi-select" data-filter="${config.key}">
                        <div class="multi-select-header" tabindex="0">
                            <span class="multi-select-summary">${getFilterSummary(config.key, options, selectedValues)}</span>
                            <span class="multi-select-arrow">&#9662;</span>
                        </div>
                        <div class="multi-select-dropdown">
                            <div class="multi-select-controls">
                                <button type="button" class="select-all-btn" data-filter="${config.key}">All</button>
                                <button type="button" class="select-none-btn" data-filter="${config.key}">None</button>
                            </div>
                            <div class="multi-select-options">
                                ${options.map(opt => {
                                    const checked = selectedValues === null || selectedValues.has(opt);
                                    const displayValue = formatFilterValue(config.key, opt);
                                    return `
                                        <label class="multi-select-option">
                                            <input type="checkbox" value="${escapeHtml(String(opt))}" ${checked ? 'checked' : ''}>
                                            <span class="option-label">${escapeHtml(displayValue)}</span>
                                        </label>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        // Add clear all filters button
        html += `
            <div class="filter-actions">
                <button type="button" class="clear-filters-btn" id="clear-filters-btn">Clear All Filters</button>
            </div>
        `;

        container.innerHTML = html;

        // Attach event listeners for filter interactions
        attachFilterListeners();
    }

    /**
     * Get summary text for a filter (showing selection state)
     */
    function getFilterSummary(key, options, selectedValues) {
        if (selectedValues === null || selectedValues.size === options.length) {
            return 'All';
        }
        if (selectedValues.size === 0) {
            return 'None';
        }
        if (selectedValues.size === 1) {
            const value = Array.from(selectedValues)[0];
            return formatFilterValue(key, value);
        }
        return `${selectedValues.size} selected`;
    }

    /**
     * Format a filter value for display
     */
    function formatFilterValue(key, value) {
        if (value === null || value === undefined) {
            return '(empty)';
        }
        // Truncate long values
        const str = String(value);
        if (str.length > 20) {
            return str.substring(0, 17) + '...';
        }
        return str;
    }

    /**
     * Escape HTML special characters
     */
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Attach event listeners for filter UI interactions
     */
    function attachFilterListeners() {
        // Multi-select dropdown toggle
        document.querySelectorAll('.multi-select-header').forEach(header => {
            header.addEventListener('click', (e) => {
                const multiSelect = header.closest('.multi-select');
                toggleDropdown(multiSelect);
            });

            // Keyboard accessibility
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const multiSelect = header.closest('.multi-select');
                    toggleDropdown(multiSelect);
                }
            });
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.multi-select')) {
                closeAllDropdowns();
            }
        });

        // Checkbox change handlers
        document.querySelectorAll('.multi-select-option input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const multiSelect = checkbox.closest('.multi-select');
                const filterKey = multiSelect.dataset.filter;
                handleFilterChange(filterKey, multiSelect);
            });
        });

        // Select All buttons
        document.querySelectorAll('.select-all-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const filterKey = btn.dataset.filter;
                const multiSelect = btn.closest('.multi-select');
                selectAllOptions(filterKey, multiSelect);
            });
        });

        // Select None buttons
        document.querySelectorAll('.select-none-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const filterKey = btn.dataset.filter;
                const multiSelect = btn.closest('.multi-select');
                selectNoneOptions(filterKey, multiSelect);
            });
        });

        // Clear all filters button
        const clearBtn = document.getElementById('clear-filters-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', clearAllFilters);
        }
    }

    /**
     * Toggle dropdown visibility
     */
    function toggleDropdown(multiSelect) {
        const isOpen = multiSelect.classList.contains('open');

        // Close all other dropdowns first
        closeAllDropdowns();

        if (!isOpen) {
            multiSelect.classList.add('open');
        }
    }

    /**
     * Close all open dropdowns
     */
    function closeAllDropdowns() {
        document.querySelectorAll('.multi-select.open').forEach(ms => {
            ms.classList.remove('open');
        });
    }

    /**
     * Handle filter checkbox change
     */
    function handleFilterChange(filterKey, multiSelect) {
        const options = state.filterOptions[filterKey];
        const checkboxes = multiSelect.querySelectorAll('.multi-select-option input[type="checkbox"]');

        // Collect checked values
        const checkedValues = new Set();
        checkboxes.forEach(cb => {
            if (cb.checked) {
                // Parse value back to original type if numeric
                let value = cb.value;
                const config = filterConfig.find(c => c.key === filterKey);
                if (config && config.numeric) {
                    value = parseFloat(value);
                }
                checkedValues.add(value);
            }
        });

        // Update filter state
        if (checkedValues.size === options.length) {
            // All selected = no filter
            state.filters[filterKey] = null;
        } else {
            state.filters[filterKey] = checkedValues;
        }

        // Update summary text
        const summary = multiSelect.querySelector('.multi-select-summary');
        summary.textContent = getFilterSummary(filterKey, options, state.filters[filterKey]);

        // Trigger re-render with filtered data
        onFiltersChanged();
    }

    /**
     * Select all options for a filter
     */
    function selectAllOptions(filterKey, multiSelect) {
        const checkboxes = multiSelect.querySelectorAll('.multi-select-option input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = true);
        state.filters[filterKey] = null;

        // Update summary
        const options = state.filterOptions[filterKey];
        const summary = multiSelect.querySelector('.multi-select-summary');
        summary.textContent = getFilterSummary(filterKey, options, null);

        onFiltersChanged();
    }

    /**
     * Deselect all options for a filter
     */
    function selectNoneOptions(filterKey, multiSelect) {
        const checkboxes = multiSelect.querySelectorAll('.multi-select-option input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = false);
        state.filters[filterKey] = new Set();

        // Update summary
        const options = state.filterOptions[filterKey];
        const summary = multiSelect.querySelector('.multi-select-summary');
        summary.textContent = getFilterSummary(filterKey, options, state.filters[filterKey]);

        onFiltersChanged();
    }

    /**
     * Clear all filters (reset to "All")
     */
    function clearAllFilters() {
        // Reset all filter states
        for (const key of Object.keys(state.filters)) {
            state.filters[key] = null;
        }

        // Re-render filter UI to update checkboxes and summaries
        renderFilterUI();

        onFiltersChanged();

        console.log('[dashboard] All filters cleared');
    }

    /**
     * Called when filter state changes - triggers view update
     */
    function onFiltersChanged() {
        console.log('[dashboard] Filters changed:', getActiveFiltersDescription());
        render();
    }

    /**
     * Get description of active filters for logging
     */
    function getActiveFiltersDescription() {
        const active = [];
        for (const [key, value] of Object.entries(state.filters)) {
            if (value !== null) {
                if (value instanceof Set) {
                    active.push(`${key}: ${value.size} selected`);
                } else {
                    active.push(`${key}: ${JSON.stringify(value)}`);
                }
            }
        }
        return active.length > 0 ? active.join(', ') : '(none)';
    }

    /**
     * Apply current filters to data
     * @returns {Array} Filtered measure records
     */
    function applyFilters() {
        let filtered = state.data.measures;

        // Build a map of run_id -> run record for run-level filters
        const runsMap = new Map();
        state.data.runs.forEach(run => runsMap.set(run.run_id, run));

        // Attach run timestamp for display (non-persistent)
        filtered.forEach(m => {
            const run = runsMap.get(m.run_id);
            m._run_started_at = run?.started_at_utc || null;
        });

        // Apply each filter
        // Note: token_chunk_size uses token_chunk_size_effective (or token_chunk_size_requested as fallback)
        const measureFilters = ['scenario', 'model_name', 'model_size', 'backend_id', 'wgpu_backend',
                                'batch_size', 'token_chunk_size', 'seq_len', 'mixed_case_id'];

        // Map filter keys to run data field names
        const runFilterMappings = [
            { filterKey: 'run_id', dataKey: 'run_id' },
            { filterKey: 'git_sha', dataKey: 'git_sha' },
            { filterKey: 'timestamp', dataKey: 'started_at_utc' }  // Map timestamp filter to started_at_utc field
        ];

        // Apply measure-level filters
        measureFilters.forEach(key => {
            const filterValue = state.filters[key];
            if (filterValue !== null && filterValue instanceof Set) {
                filtered = filtered.filter(m => {
                    // Special handling for token_chunk_size - check effective then requested
                    let value;
                    if (key === 'token_chunk_size') {
                        value = m.token_chunk_size_effective ?? m.token_chunk_size_requested ?? m.token_chunk_size;
                    } else {
                        value = m[key];
                    }
                    if (value === undefined || value === null) {
                        return false;  // Exclude records without the field if filter is active
                    }
                    return filterValue.has(value);
                });
            }
        });

        // Apply run-level filters (filter measures by their associated run)
        runFilterMappings.forEach(({ filterKey, dataKey }) => {
            const filterValue = state.filters[filterKey];
            if (filterValue !== null && filterValue instanceof Set) {
                filtered = filtered.filter(m => {
                    const run = runsMap.get(m.run_id);
                    if (!run) {
                        // If we can't find the run, check if the measure has run_id directly
                        if (filterKey === 'run_id') {
                            return filterValue.has(m.run_id);
                        }
                        return false;
                    }
                    const value = run[dataKey];
                    if (value === undefined || value === null) {
                        return false;
                    }
                    return filterValue.has(value);
                });
            }
        });

        console.log(`[dashboard] applyFilters(): ${state.data.measures.length} -> ${filtered.length} records`);
        return filtered;
    }

    /**
     * Switch to a different view
     * @param {string} viewName - Name of the view to switch to
     */
    function switchView(viewName) {
        console.log(`[dashboard] Switching to view: ${viewName}`);
        state.currentView = viewName;
        render();
    }

    /**
     * Main render function
     */
    function render() {
        if (!state.data.loaded) {
            renderEmptyState();
            // Also reset filters UI to placeholder
            elements.filterGroups.innerHTML = '<p class="placeholder-text">Load benchmark data to enable filters</p>';
            return;
        }

        updateStats();

        switch (state.currentView) {
            case 'table':
                renderTable();
                break;
            case 'heatmap':
                renderHeatmap();
                break;
            case 'line':
                renderLineChart();
                break;
            case 'distribution':
                renderDistribution();
                break;
            case 'compare':
                renderCompare();
                break;
            default:
                renderEmptyState();
        }
    }

    /**
     * Render empty state placeholder
     */
    function renderEmptyState() {
        elements.viewContainer.innerHTML = `
            <div class="empty-state">
                <p>No data loaded</p>
                <p class="empty-hint">Drop a JSONL benchmark file above to get started</p>
            </div>
        `;
    }

    /**
     * Update stats bar with current data summary
     */
    function updateStats() {
        const runs = state.data.runs;
        const measures = state.data.measures;

        // Count unique values
        const uniqueBackends = new Set(measures.map(m => m.backend_id).filter(Boolean));
        const uniqueModels = new Set(measures.map(m => m.model_name).filter(Boolean));

        // Update stat values
        elements.stats.runs.textContent = runs.length;
        elements.stats.cases.textContent = measures.length;
        elements.stats.backends.textContent = uniqueBackends.size;
        elements.stats.models.textContent = uniqueModels.size;

        // Enable/disable export button based on data availability
        if (elements.exportBtn) {
            elements.exportBtn.disabled = measures.length === 0;
        }

        console.log(`[dashboard] Stats updated: ${runs.length} runs, ${measures.length} cases, ${uniqueBackends.size} backends, ${uniqueModels.size} models`);
    }

    // Table sorting state
    const tableState = {
        sortColumn: 'decode_tok_per_s',  // Default sort by throughput
        sortDirection: 'desc',            // desc = highest first
        selectedRow: null                 // Currently selected row for detail view
    };

    // Compare view state
    const compareState = {
        baseRun: null,         // run_id of the baseline run (older)
        compareRun: null,      // run_id of the run to compare (newer)
        threshold: 5,          // Regression threshold percentage (highlight regressions >= this)
        sortColumn: 'delta_pct',
        sortDirection: 'desc'  // Sort by largest regressions first
    };

    const lineChartState = {
        scenario: 'prefill_uniform',
        xAxis: {
            prefill_uniform: 'seq_len',
            prefill_mixed: 'mixed_case_id',
            decode_only: 'decode_steps'
        },
        logX: {
            prefill_uniform: false,
            prefill_mixed: false,
            decode_only: false
        }
    };

    /**
     * Render summary table view
     */
    function renderTable() {
        console.log('[dashboard] renderTable()');

        // Get filtered data
        const measures = applyFilters();

        if (measures.length === 0) {
            elements.viewContainer.innerHTML = `
                <div class="empty-state">
                    <p>No data matches the current filters</p>
                    <p class="empty-hint">Try adjusting your filter selections</p>
                </div>
            `;
            return;
        }

        // Aggregate measures by case_id (average across repeats)
        const aggregated = aggregateMeasures(measures);

        // Sort the data
        const sorted = sortTableData(aggregated, tableState.sortColumn, tableState.sortDirection);

        // Build table HTML
        const tableHtml = buildTableHtml(sorted);

        elements.viewContainer.innerHTML = tableHtml;

        // Attach event listeners for sorting and row selection
        attachTableListeners();

        console.log(`[dashboard] Table rendered with ${sorted.length} aggregated cases`);
    }

    /**
     * Aggregate measures by case_id, computing averages across repeats
     * @param {Array} measures - Raw measure records
     * @returns {Array} Aggregated records (one per case)
     */
    function aggregateMeasures(measures) {
        // Group by case_id
        const byCase = new Map();

        measures.forEach(m => {
            const caseId = m.case_id || generateCaseId(m);
            if (!byCase.has(caseId)) {
                byCase.set(caseId, []);
            }
            byCase.get(caseId).push(m);
        });

        // Aggregate each group
        const aggregated = [];

        byCase.forEach((records, caseId) => {
            // Use first record as template for non-metric fields
            const template = records[0];

            // Compute averages for metric fields
            const agg = {
                case_id: caseId,
                scenario: template.scenario,
                started_at_utc: template._run_started_at || null,
                model_name: template.model_name,
                model_size: template.model_size,
                backend_id: template.backend_id,
                wgpu_backend: template.wgpu_backend,
                batch_size: template.batch_size,
                token_chunk_size: template.token_chunk_size_effective || template.token_chunk_size_requested,
                seq_len: template.seq_len,
                mixed_case_id: template.mixed_case_id,
                decode_steps: template.decode_steps,
                status: template.status,
                repeat_count: records.length,
                // Store raw records for drilldown
                _records: records
            };

            // Filter to only successful records for metric averaging
            const okRecords = records.filter(r => r.status === 'ok');

            if (okRecords.length > 0) {
                // Decode metrics
                if (template.scenario === 'decode_only') {
                    agg.decode_total_ms = average(okRecords, 'decode_total_ms');
                    agg.decode_tokens = okRecords[0].decode_tokens;  // Same for all repeats
                    agg.decode_tok_per_s = average(okRecords, 'decode_tok_per_s');
                    agg.decode_step_ms_p50 = average(okRecords, 'decode_step_ms_p50');
                    agg.decode_step_ms_p95 = average(okRecords, 'decode_step_ms_p95');
                }

                // Prefill metrics
                if (template.scenario === 'prefill_uniform' || template.scenario === 'prefill_mixed') {
                    agg.prefill_total_ms = average(okRecords, 'prefill_total_ms');
                    agg.total_prompt_tokens = okRecords[0].total_prompt_tokens;
                    agg.prefill_tok_per_s = average(okRecords, 'prefill_tok_per_s');
                    agg.num_infer_calls = okRecords[0].num_infer_calls;
                    agg.ttft_min_ms = average(okRecords, 'ttft_min_ms');
                    agg.ttft_p50_ms = average(okRecords, 'ttft_p50_ms');
                    agg.ttft_max_ms = average(okRecords, 'ttft_max_ms');
                }
            }

            aggregated.push(agg);
        });

        return aggregated;
    }

    /**
     * Generate a case_id from record fields if not present
     */
    function generateCaseId(m) {
        const parts = [
            m.scenario,
            m.model_name,
            m.backend_id,
            m.wgpu_backend,
            `bs${m.batch_size}`,
            `c${m.token_chunk_size_effective || m.token_chunk_size_requested}`
        ];

        if (m.scenario === 'decode_only') {
            parts.push(`steps${m.decode_steps}`);
        } else if (m.scenario === 'prefill_uniform') {
            parts.push(`len${m.seq_len}`);
        } else if (m.scenario === 'prefill_mixed') {
            parts.push(m.mixed_case_id);
        }

        return parts.join(':');
    }

    /**
     * Compute average of a field across records
     */
    function average(records, field) {
        const values = records.map(r => r[field]).filter(v => v !== undefined && v !== null && !isNaN(v));
        if (values.length === 0) return null;
        return values.reduce((a, b) => a + b, 0) / values.length;
    }

    /**
     * Sort table data by column
     */
    function sortTableData(data, column, direction) {
        const numericColumns = new Set([
            'batch_size',
            'token_chunk_size',
            'decode_steps',
            'seq_len',
            'decode_tok_per_s',
            'prefill_tok_per_s',
            'ttft_p50_ms',
            'repeat_count',
            'decode_total_ms',
            'prefill_total_ms'
        ]);

        const sorted = [...data];

        sorted.sort((a, b) => {
            let aVal = a[column];
            let bVal = b[column];

            if (numericColumns.has(column)) {
                aVal = aVal !== null && aVal !== undefined ? Number(aVal) : NaN;
                bVal = bVal !== null && bVal !== undefined ? Number(bVal) : NaN;
            }

            // Handle nulls
            if (aVal === null || aVal === undefined || Number.isNaN(aVal)) aVal = -Infinity;
            if (bVal === null || bVal === undefined || Number.isNaN(bVal)) bVal = -Infinity;

            // Numeric comparison for numbers, string for others
            let cmp;
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                cmp = aVal - bVal;
            } else {
                cmp = String(aVal).localeCompare(String(bVal));
            }

            return direction === 'asc' ? cmp : -cmp;
        });

        return sorted;
    }

    /**
     * Build HTML for the summary table
     */
    function buildTableHtml(data) {
        // Define columns based on what data is available
        const hasDecodeData = data.some(d => d.scenario === 'decode_only');
        const hasPrefillData = data.some(d => d.scenario === 'prefill_uniform' || d.scenario === 'prefill_mixed');

        // Base columns always shown
        const columns = [
            { key: 'scenario', label: 'Scenario', sortable: true },
            { key: 'started_at_utc', label: 'Time', sortable: true },
            { key: 'model_name', label: 'Model', sortable: true },
            { key: 'model_size', label: 'Size', sortable: true },
            { key: 'backend_id', label: 'Backend', sortable: true },
            { key: 'batch_size', label: 'Batch', sortable: true, numeric: true },
            { key: 'token_chunk_size', label: 'Chunk', sortable: true, numeric: true }
        ];

        // Add scenario-specific columns
        if (hasDecodeData) {
            columns.push(
                { key: 'decode_steps', label: 'Steps', sortable: true, numeric: true },
                { key: 'decode_tok_per_s', label: 'Decode tok/s', sortable: true, numeric: true, metric: true }
            );
        }

        if (hasPrefillData) {
            columns.push(
                { key: 'seq_len', label: 'Seq Len', sortable: true, numeric: true },
                { key: 'prefill_tok_per_s', label: 'Prefill tok/s', sortable: true, numeric: true, metric: true },
                { key: 'ttft_p50_ms', label: 'TTFT p50 (ms)', sortable: true, numeric: true, metric: true }
            );
        }

        // Status and repeat count
        columns.push(
            { key: 'status', label: 'Status', sortable: true },
            { key: 'repeat_count', label: 'Reps', sortable: true, numeric: true }
        );

        // Build header
        let headerHtml = '<tr>';
        columns.forEach(col => {
            const sortClass = col.sortable ? 'sortable' : '';
            const activeClass = col.key === tableState.sortColumn ? 'sort-active' : '';
            const dirClass = col.key === tableState.sortColumn ? `sort-${tableState.sortDirection}` : '';
            const numericClass = col.numeric ? 'numeric' : '';
            const sortIndicator = col.key === tableState.sortColumn
                ? (tableState.sortDirection === 'asc' ? ' &#9650;' : ' &#9660;')
                : '';

            headerHtml += `<th class="${sortClass} ${activeClass} ${dirClass} ${numericClass}" data-column="${col.key}">${col.label}${sortIndicator}</th>`;
        });
        headerHtml += '</tr>';

        // Build rows
        let bodyHtml = '';
        data.forEach((row, idx) => {
            const statusClass = row.status === 'ok' ? '' : (row.status === 'error' ? 'status-error' : 'status-skipped');
            const selectedClass = tableState.selectedRow === idx ? 'selected' : '';

            bodyHtml += `<tr class="${statusClass} ${selectedClass} table-row-clickable" data-row-index="${idx}">`;

            columns.forEach(col => {
                const value = row[col.key];
                const numericClass = col.numeric ? 'numeric' : '';
                const formatted = formatCellValue(col.key, value, col.metric);

                bodyHtml += `<td class="${numericClass}">${formatted}</td>`;
            });

            bodyHtml += '</tr>';
        });

        // Build detail panel (shown below table when row selected)
        const detailHtml = tableState.selectedRow !== null
            ? buildDetailPanel(data[tableState.selectedRow])
            : '';

        return `
            <div class="table-wrapper">
                <table class="data-table summary-table">
                    <thead>${headerHtml}</thead>
                    <tbody>${bodyHtml}</tbody>
                </table>
            </div>
            ${detailHtml}
        `;
    }

    /**
     * Format a cell value for display
     */
    function formatCellValue(key, value, isMetric) {
        if (value === null || value === undefined) {
            return '<span class="cell-empty">--</span>';
        }

        if (key === 'started_at_utc') {
            return formatRunTimestamp(value);
        }

        // Format numbers
        if (typeof value === 'number') {
            if (isMetric) {
                // Metrics: show with appropriate precision
                if (value >= 1000) {
                    return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
                } else if (value >= 1) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(3);
                }
            } else {
                // Non-metric numbers (batch size, etc.)
                return value.toLocaleString();
            }
        }

        // Status badges
        if (key === 'status') {
            const statusClass = value === 'ok' ? 'status-ok' : (value === 'error' ? 'status-error' : 'status-skipped');
            return `<span class="status-badge ${statusClass}">${value}</span>`;
        }

        // Truncate long model names
        if (key === 'model_name' && value.length > 30) {
            return `<span title="${escapeHtml(value)}">${escapeHtml(value.substring(0, 27))}...</span>`;
        }

        return escapeHtml(String(value));
    }

    function formatRunTimestamp(value) {
        try {
            const date = new Date(value);
            return date.toLocaleString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return escapeHtml(String(value));
        }
    }

    /**
     * Build detail panel for selected row
     */
    function buildDetailPanel(row) {
        if (!row) return '';

        const records = row._records || [];

        let metricsHtml = '';

        // Show all metrics for the case
        if (row.scenario === 'decode_only') {
            metricsHtml = `
                <div class="detail-metrics">
                    <div class="detail-metric">
                        <span class="metric-label">Decode Steps</span>
                        <span class="metric-value">${row.decode_steps || '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Total Tokens</span>
                        <span class="metric-value">${row.decode_tokens || '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Throughput (tok/s)</span>
                        <span class="metric-value metric-primary">${row.decode_tok_per_s ? row.decode_tok_per_s.toFixed(1) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Total Time (ms)</span>
                        <span class="metric-value">${row.decode_total_ms ? row.decode_total_ms.toFixed(2) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Step p50 (ms)</span>
                        <span class="metric-value">${row.decode_step_ms_p50 ? row.decode_step_ms_p50.toFixed(3) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Step p95 (ms)</span>
                        <span class="metric-value">${row.decode_step_ms_p95 ? row.decode_step_ms_p95.toFixed(3) : '--'}</span>
                    </div>
                </div>
            `;
        } else if (row.scenario === 'prefill_uniform' || row.scenario === 'prefill_mixed') {
            metricsHtml = `
                <div class="detail-metrics">
                    <div class="detail-metric">
                        <span class="metric-label">Seq Length</span>
                        <span class="metric-value">${row.seq_len || row.mixed_case_id || '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Total Tokens</span>
                        <span class="metric-value">${row.total_prompt_tokens || '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Throughput (tok/s)</span>
                        <span class="metric-value metric-primary">${row.prefill_tok_per_s ? row.prefill_tok_per_s.toFixed(1) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Total Time (ms)</span>
                        <span class="metric-value">${row.prefill_total_ms ? row.prefill_total_ms.toFixed(2) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">TTFT Min (ms)</span>
                        <span class="metric-value">${row.ttft_min_ms ? row.ttft_min_ms.toFixed(2) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">TTFT p50 (ms)</span>
                        <span class="metric-value">${row.ttft_p50_ms ? row.ttft_p50_ms.toFixed(2) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">TTFT Max (ms)</span>
                        <span class="metric-value">${row.ttft_max_ms ? row.ttft_max_ms.toFixed(2) : '--'}</span>
                    </div>
                    <div class="detail-metric">
                        <span class="metric-label">Infer Calls</span>
                        <span class="metric-value">${row.num_infer_calls || '--'}</span>
                    </div>
                </div>
            `;
        }

        // Build repeat table if multiple repeats
        let repeatsHtml = '';
        if (records.length > 1) {
            const repeatRows = records.map((r, i) => {
                const throughput = r.decode_tok_per_s || r.prefill_tok_per_s;
                const time = r.decode_total_ms || r.prefill_total_ms;
                return `
                    <tr>
                        <td class="numeric">${r.repeat_index}</td>
                        <td class="numeric">${throughput ? throughput.toFixed(1) : '--'}</td>
                        <td class="numeric">${time ? time.toFixed(2) : '--'}</td>
                        <td>${r.status}</td>
                    </tr>
                `;
            }).join('');

            repeatsHtml = `
                <div class="detail-repeats">
                    <h4>Individual Repeats</h4>
                    <table class="data-table repeats-table">
                        <thead>
                            <tr>
                                <th class="numeric">#</th>
                                <th class="numeric">tok/s</th>
                                <th class="numeric">Time (ms)</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>${repeatRows}</tbody>
                    </table>
                </div>
            `;
        }

        return `
            <div class="detail-panel" id="detail-panel">
                <div class="detail-header">
                    <h3>Case Details</h3>
                    <button class="detail-close-btn" id="detail-close-btn" title="Close details">&#10005;</button>
                </div>
                <div class="detail-info">
                    <div class="detail-info-row">
                        <span class="info-label">Case ID:</span>
                        <span class="info-value monospace">${escapeHtml(row.case_id || '--')}</span>
                    </div>
                    <div class="detail-info-row">
                        <span class="info-label">Model:</span>
                        <span class="info-value">${escapeHtml(row.model_name || '--')}</span>
                    </div>
                    <div class="detail-info-row">
                        <span class="info-label">Backend:</span>
                        <span class="info-value">${row.backend_id} / ${row.wgpu_backend}</span>
                    </div>
                    <div class="detail-info-row">
                        <span class="info-label">Configuration:</span>
                        <span class="info-value">batch=${row.batch_size}, chunk=${row.token_chunk_size}</span>
                    </div>
                </div>
                ${metricsHtml}
                ${repeatsHtml}
            </div>
        `;
    }

    /**
     * Attach event listeners for table interactions
     */
    function attachTableListeners() {
        // Sortable column headers
        document.querySelectorAll('.summary-table th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const column = th.dataset.column;

                // Toggle direction if same column, otherwise default to desc for metrics
                if (tableState.sortColumn === column) {
                    tableState.sortDirection = tableState.sortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    tableState.sortColumn = column;
                    // Default to desc for numeric/metric columns, asc for text
                    const isNumeric = ['batch_size', 'token_chunk_size', 'decode_steps', 'seq_len',
                                      'decode_tok_per_s', 'prefill_tok_per_s', 'ttft_p50_ms',
                                      'repeat_count', 'decode_total_ms', 'prefill_total_ms'].includes(column);
                    tableState.sortDirection = isNumeric ? 'desc' : 'asc';
                }

                renderTable();
            });
        });

        // Row click for detail view
        document.querySelectorAll('.summary-table tbody tr').forEach(tr => {
            tr.addEventListener('click', () => {
                const rowIndex = parseInt(tr.dataset.rowIndex, 10);

                // Toggle selection
                if (tableState.selectedRow === rowIndex) {
                    tableState.selectedRow = null;
                } else {
                    tableState.selectedRow = rowIndex;
                }

                renderTable();
            });
        });

        // Close button for detail panel
        const closeBtn = document.getElementById('detail-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                tableState.selectedRow = null;
                renderTable();
            });
        }
    }

    /**
     * Render heatmap view using D3
     * Displays batch_size x seq_len colored by tok/s for prefill-uniform data
     */
    function renderHeatmap() {
        console.log('[dashboard] renderHeatmap()');

        // Get filtered data and filter to prefill_uniform scenario only
        const allFiltered = applyFilters();
        const filtered = allFiltered.filter(m => m.scenario === 'prefill_uniform');

        if (filtered.length === 0) {
            elements.viewContainer.innerHTML = `
                <div class="empty-state">
                    <p>No prefill-uniform data available</p>
                    <p class="empty-hint">Load benchmark data with prefill_uniform scenario or adjust filters</p>
                </div>
            `;
            return;
        }

        // Extract unique batch_sizes and seq_lens, sorted numerically
        const batchSizes = [...new Set(filtered.map(m => m.batch_size))].filter(v => v != null).sort((a, b) => a - b);
        const seqLens = [...new Set(filtered.map(m => m.seq_len))].filter(v => v != null).sort((a, b) => a - b);

        if (batchSizes.length === 0 || seqLens.length === 0) {
            elements.viewContainer.innerHTML = `
                <div class="empty-state">
                    <p>Insufficient data for heatmap</p>
                    <p class="empty-hint">Need batch_size and seq_len values in data</p>
                </div>
            `;
            return;
        }

        // Build a lookup map: (batch_size, seq_len) -> aggregated tok/s
        // For cells with multiple measures (repeats), we average
        const dataMap = new Map();
        filtered.forEach(m => {
            if (m.batch_size == null || m.seq_len == null || m.prefill_tok_per_s == null) return;
            const key = `${m.batch_size}_${m.seq_len}`;
            if (!dataMap.has(key)) {
                dataMap.set(key, { values: [], batch_size: m.batch_size, seq_len: m.seq_len });
            }
            dataMap.get(key).values.push(m.prefill_tok_per_s);
        });

        // Compute averages
        const heatmapData = [];
        dataMap.forEach((entry, key) => {
            const avg = entry.values.reduce((a, b) => a + b, 0) / entry.values.length;
            heatmapData.push({
                batch_size: entry.batch_size,
                seq_len: entry.seq_len,
                value: avg,
                count: entry.values.length
            });
        });

        // Find value range for color scale
        const values = heatmapData.map(d => d.value);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);

        // Render the heatmap
        renderHeatmapSVG(heatmapData, batchSizes, seqLens, minValue, maxValue);
    }

    /**
     * Render heatmap SVG using D3
     */
    function renderHeatmapSVG(data, batchSizes, seqLens, minValue, maxValue) {
        // Clear container
        elements.viewContainer.innerHTML = '';

        // Create wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'heatmap-wrapper';
        elements.viewContainer.appendChild(wrapper);

        // Dimensions
        const margin = { top: 40, right: 120, bottom: 60, left: 80 };
        const containerWidth = elements.viewContainer.clientWidth || 800;
        const width = Math.min(containerWidth - margin.left - margin.right, 800);
        const height = Math.max(300, Math.min(500, seqLens.length * 35));

        const cellWidth = width / seqLens.length;
        const cellHeight = height / batchSizes.length;

        // Create SVG
        const svg = d3.select(wrapper)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .attr('class', 'heatmap-svg');

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Color scale (blues - darker = higher throughput)
        const colorScale = d3.scaleSequential()
            .domain([minValue, maxValue])
            .interpolator(d3.interpolateBlues);

        // Build a quick lookup for cell data
        const dataLookup = new Map();
        data.forEach(d => {
            dataLookup.set(`${d.batch_size}_${d.seq_len}`, d);
        });

        // X scale (seq_len)
        const xScale = d3.scaleBand()
            .domain(seqLens.map(String))
            .range([0, width])
            .padding(0.05);

        // Y scale (batch_size)
        const yScale = d3.scaleBand()
            .domain(batchSizes.map(String))
            .range([0, height])
            .padding(0.05);

        // Draw cells for each combination
        batchSizes.forEach(bs => {
            seqLens.forEach(sl => {
                const cellData = dataLookup.get(`${bs}_${sl}`);
                const x = xScale(String(sl));
                const y = yScale(String(bs));

                if (cellData) {
                    // Cell with data
                    g.append('rect')
                        .attr('x', x)
                        .attr('y', y)
                        .attr('width', xScale.bandwidth())
                        .attr('height', yScale.bandwidth())
                        .attr('fill', colorScale(cellData.value))
                        .attr('class', 'heatmap-cell')
                        .on('mouseover', function(event) {
                            showHeatmapTooltip(event, cellData);
                        })
                        .on('mouseout', hideHeatmapTooltip);
                } else {
                    // Missing cell - show as gray with diagonal pattern
                    g.append('rect')
                        .attr('x', x)
                        .attr('y', y)
                        .attr('width', xScale.bandwidth())
                        .attr('height', yScale.bandwidth())
                        .attr('fill', 'var(--bg-tertiary)')
                        .attr('stroke', 'var(--border-subtle)')
                        .attr('stroke-width', 1)
                        .attr('class', 'heatmap-cell heatmap-cell-missing');
                }
            });
        });

        // X axis (seq_len)
        const xAxis = g.append('g')
            .attr('class', 'axis x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale));

        // X axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('x', width / 2)
            .attr('y', height + 45)
            .attr('text-anchor', 'middle')
            .text('Sequence Length');

        // Y axis (batch_size)
        const yAxis = g.append('g')
            .attr('class', 'axis y-axis')
            .call(d3.axisLeft(yScale));

        // Y axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -50)
            .attr('text-anchor', 'middle')
            .text('Batch Size');

        // Color legend
        renderHeatmapLegend(svg, colorScale, minValue, maxValue, width + margin.left + 20, margin.top, height);

        // Title
        svg.append('text')
            .attr('class', 'chart-title')
            .attr('x', margin.left + width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .text('Prefill Throughput (tok/s) by Batch Size and Sequence Length');

        console.log(`[dashboard] Heatmap rendered: ${batchSizes.length} x ${seqLens.length} grid, ${data.length} cells with data`);
    }

    /**
     * Render color legend for heatmap
     */
    function renderHeatmapLegend(svg, colorScale, minValue, maxValue, x, y, height) {
        const legendWidth = 20;
        const legendHeight = Math.min(height, 200);

        const legendGroup = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${x},${y})`);

        // Create gradient
        const gradientId = 'heatmap-gradient-' + Date.now();
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', gradientId)
            .attr('x1', '0%')
            .attr('y1', '100%')
            .attr('x2', '0%')
            .attr('y2', '0%');

        // Add gradient stops
        const nStops = 10;
        for (let i = 0; i <= nStops; i++) {
            const t = i / nStops;
            const value = minValue + t * (maxValue - minValue);
            gradient.append('stop')
                .attr('offset', `${t * 100}%`)
                .attr('stop-color', colorScale(value));
        }

        // Draw legend rect
        legendGroup.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .attr('fill', `url(#${gradientId})`)
            .attr('stroke', 'var(--border-color)')
            .attr('stroke-width', 1);

        // Legend scale
        const legendScale = d3.scaleLinear()
            .domain([minValue, maxValue])
            .range([legendHeight, 0]);

        // Legend axis
        const legendAxis = d3.axisRight(legendScale)
            .ticks(5)
            .tickFormat(d => formatThroughput(d));

        legendGroup.append('g')
            .attr('class', 'legend-axis')
            .attr('transform', `translate(${legendWidth},0)`)
            .call(legendAxis);

        // Legend title
        legendGroup.append('text')
            .attr('class', 'legend-title')
            .attr('x', legendWidth / 2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .text('tok/s');
    }

    /**
     * Format throughput value for display
     */
    function formatThroughput(value) {
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        } else {
            return value.toFixed(0);
        }
    }

    /**
     * Show tooltip for heatmap cell
     */
    function showHeatmapTooltip(event, cellData) {
        // Remove existing tooltip
        hideHeatmapTooltip();

        const tooltip = document.createElement('div');
        tooltip.className = 'heatmap-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-row"><strong>Batch Size:</strong> ${cellData.batch_size}</div>
            <div class="tooltip-row"><strong>Seq Length:</strong> ${cellData.seq_len}</div>
            <div class="tooltip-row"><strong>Throughput:</strong> ${formatThroughput(cellData.value)} tok/s</div>
            <div class="tooltip-row"><strong>Samples:</strong> ${cellData.count}</div>
        `;

        document.body.appendChild(tooltip);

        // Position tooltip
        const tooltipRect = tooltip.getBoundingClientRect();
        let left = event.pageX + 10;
        let top = event.pageY + 10;

        // Keep tooltip in viewport
        if (left + tooltipRect.width > window.innerWidth) {
            left = event.pageX - tooltipRect.width - 10;
        }
        if (top + tooltipRect.height > window.innerHeight) {
            top = event.pageY - tooltipRect.height - 10;
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    /**
     * Hide heatmap tooltip
     */
    function hideHeatmapTooltip() {
        const existing = document.querySelector('.heatmap-tooltip');
        if (existing) {
            existing.remove();
        }
    }

    /**
     * Render line chart view using D3
     * Displays scenario-specific line charts with series grouped by:
     * batch_size + model_name + backend_id + token_chunk_size
     */
    function renderLineChart() {
        console.log('[dashboard] renderLineChart()');

        elements.viewContainer.innerHTML = '';

        const container = document.createElement('div');
        container.className = 'line-chart-container';

        const controls = document.createElement('div');
        controls.className = 'line-chart-controls';

        const tabs = document.createElement('div');
        tabs.className = 'line-subtabs';

        const scenarios = [
            { id: 'prefill_uniform', label: 'Prefill' },
            { id: 'prefill_mixed', label: 'Prefill Mixed' },
            { id: 'decode_only', label: 'Decode' }
        ];

        scenarios.forEach(scenario => {
            const btn = document.createElement('button');
            btn.className = `line-subtab-btn ${lineChartState.scenario === scenario.id ? 'active' : ''}`;
            btn.textContent = scenario.label;
            btn.addEventListener('click', () => {
                if (lineChartState.scenario === scenario.id) return;
                lineChartState.scenario = scenario.id;
                renderLineChart();
            });
            tabs.appendChild(btn);
        });

        const axisPicker = document.createElement('div');
        axisPicker.className = 'line-axis-picker';
        axisPicker.innerHTML = `
            <label for="line-axis-select">X-axis</label>
            <select id="line-axis-select" class="line-axis-select"></select>
        `;
        const logToggle = document.createElement('div');
        logToggle.className = 'line-axis-log';
        logToggle.innerHTML = `
            <label class="line-axis-log-label">
                <input type="checkbox" id="line-axis-log" />
                Log X
            </label>
        `;
        const axisControls = document.createElement('div');
        axisControls.className = 'line-axis-controls';
        axisControls.appendChild(axisPicker);
        axisControls.appendChild(logToggle);

        controls.appendChild(tabs);
        controls.appendChild(axisControls);

        const chartContainer = document.createElement('div');
        chartContainer.className = 'line-chart-view';

        container.appendChild(controls);
        container.appendChild(chartContainer);
        elements.viewContainer.appendChild(container);

        updateLineAxisOptions();
        renderLineChartPlot(chartContainer, lineChartState.scenario);
    }

    function getLineChartAxisOptions(scenario) {
        switch (scenario) {
            case 'decode_only':
                return [
                    { key: 'decode_steps', label: 'Decode Steps', type: 'numeric' },
                    { key: 'batch_size', label: 'Batch Size', type: 'numeric' },
                    { key: 'token_chunk_size', label: 'Chunk Size', type: 'numeric' }
                ];
            case 'prefill_mixed':
                return [
                    { key: 'mixed_case_id', label: 'Mixed Case', type: 'categorical' },
                    { key: 'batch_size', label: 'Batch Size', type: 'numeric' },
                    { key: 'token_chunk_size', label: 'Chunk Size', type: 'numeric' }
                ];
            default:
                return [
                    { key: 'seq_len', label: 'Seq Length', type: 'numeric' },
                    { key: 'batch_size', label: 'Batch Size', type: 'numeric' },
                    { key: 'token_chunk_size', label: 'Chunk Size', type: 'numeric' }
                ];
        }
    }

    function updateLineAxisOptions() {
        const select = document.getElementById('line-axis-select');
        const logToggle = document.getElementById('line-axis-log');
        if (!select) return;

        const scenario = lineChartState.scenario;
        const options = getLineChartAxisOptions(scenario);
        const preferred = lineChartState.xAxis[scenario];
        const selectedKey = options.some(o => o.key === preferred) ? preferred : options[0].key;

        select.innerHTML = '';
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.key;
            opt.textContent = option.label;
            if (option.key === selectedKey) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });

        lineChartState.xAxis[scenario] = selectedKey;

        select.onchange = () => {
            lineChartState.xAxis[scenario] = select.value;
            renderLineChart();
        };

        if (logToggle) {
            const axisType = options.find(opt => opt.key === selectedKey)?.type || 'numeric';
            if (axisType !== 'numeric') {
                lineChartState.logX[scenario] = false;
                logToggle.checked = false;
                logToggle.disabled = true;
            } else {
                logToggle.disabled = false;
                logToggle.checked = !!lineChartState.logX[scenario];
            }
            logToggle.onchange = () => {
                if (axisType !== 'numeric') return;
                lineChartState.logX[scenario] = logToggle.checked;
                renderLineChart();
            };
        }
    }

    function getLineChartConfig(scenario) {
        const axisKey = lineChartState.xAxis[scenario] || getLineChartAxisOptions(scenario)[0].key;
        const axisOptions = getLineChartAxisOptions(scenario);
        const axis = axisOptions.find(opt => opt.key === axisKey) || axisOptions[0];
        const logX = axis.type === 'numeric' ? !!lineChartState.logX[scenario] : false;

        switch (scenario) {
            case 'decode_only':
                return {
                    scenario,
                    title: 'Decode Throughput vs Steps',
                    xKey: axis.key,
                    xLabel: axis.label,
                    xType: axis.type,
                    logX,
                    yKey: 'decode_tok_per_s',
                    yLabel: 'Decode tok/s',
                    yUnit: 'tok/s'
                };
            case 'prefill_mixed':
                return {
                    scenario,
                    title: 'Prefill Mixed Throughput by Case',
                    xKey: axis.key,
                    xLabel: axis.label,
                    xType: axis.type,
                    logX,
                    yKey: 'prefill_tok_per_s',
                    yLabel: 'Prefill tok/s',
                    yUnit: 'tok/s'
                };
            default:
                return {
                    scenario: 'prefill_uniform',
                    title: 'Prefill Throughput vs Sequence Length',
                    xKey: axis.key,
                    xLabel: axis.label,
                    xType: axis.type,
                    logX,
                    yKey: 'prefill_tok_per_s',
                    yLabel: 'Prefill tok/s',
                    yUnit: 'tok/s',
                    xFormatter: axis.key === 'seq_len' ? formatSeqLen : null
                };
        }
    }

    function renderLineChartPlot(container, scenario) {
        const config = getLineChartConfig(scenario);

        // Get filtered data and filter to target scenario only
        const allFiltered = applyFilters();
        const filtered = allFiltered.filter(m => m.scenario === config.scenario);

        if (filtered.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No ${config.scenario.replace('_', ' ')} data available</p>
                    <p class="empty-hint">Load benchmark data for this scenario or adjust filters</p>
                </div>
            `;
            return;
        }

        const xValues = [...new Set(filtered.map(m => m[config.xKey]).filter(v => v != null))];
        if (xValues.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>Insufficient data for line chart</p>
                    <p class="empty-hint">Need ${config.xKey} values in data</p>
                </div>
            `;
            return;
        }

        // Group data by series key: batch_size + model_name + backend_id + chunk_size
        const seriesMap = new Map();
        filtered.forEach(m => {
            const xValue = m[config.xKey];
            const yValue = m[config.yKey];
            if (xValue == null || yValue == null) return;

            const chunkSize = m.token_chunk_size_effective || m.token_chunk_size_requested || m.token_chunk_size;
            const seriesKeyParts = [
                config.xKey !== 'batch_size' ? (m.batch_size || '?') : null,
                m.model_name || '?',
                m.backend_id || '?',
                config.xKey !== 'token_chunk_size' ? (chunkSize || '?') : null,
                m.run_id || '?'
            ].filter(part => part !== null);
            const seriesKey = seriesKeyParts.join('|');

            if (!seriesMap.has(seriesKey)) {
                seriesMap.set(seriesKey, {
                    key: seriesKey,
                    batch_size: m.batch_size,
                    model_name: m.model_name,
                    model_size: m.model_size,
                    backend_id: m.backend_id,
                    token_chunk_size: chunkSize,
                    run_id: m.run_id,
                    run_started_at: m._run_started_at || null,
                    xKey: config.xKey,
                    points: new Map()  // xKey -> [values]
                });
            }

            const series = seriesMap.get(seriesKey);
            if (!series.points.has(xValue)) {
                series.points.set(xValue, []);
            }
            series.points.get(xValue).push(yValue);
        });

        // Convert to array format with averaged points
        const seriesData = [];
        seriesMap.forEach((series, key) => {
            const points = [];
            series.points.forEach((values, xValue) => {
                const avg = values.reduce((a, b) => a + b, 0) / values.length;
                points.push({ x: xValue, value: avg, count: values.length });
            });

            // Sort points for proper line drawing
            if (config.xType === 'numeric') {
                points.sort((a, b) => a.x - b.x);
            } else {
                points.sort((a, b) => String(a.x).localeCompare(String(b.x)));
            }

            if (points.length > 0) {
                seriesData.push({
                    key: key,
                    batch_size: series.batch_size,
                    model_name: series.model_name,
                    model_size: series.model_size,
                    backend_id: series.backend_id,
                    token_chunk_size: series.token_chunk_size,
                    run_id: series.run_id,
                    run_started_at: series.run_started_at,
                    xKey: series.xKey,
                    points: points
                });
            }
        });

        if (seriesData.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No valid data points for line chart</p>
                    <p class="empty-hint">Check that data has ${config.xKey} and ${config.yKey} values</p>
                </div>
            `;
            return;
        }

        const orderedX = config.xType === 'numeric'
            ? xValues.sort((a, b) => a - b)
            : xValues.sort((a, b) => String(a).localeCompare(String(b)));

        renderLineChartSVG(container, seriesData, orderedX, config);
    }

    /**
     * Render line chart SVG using D3
     */
    function renderLineChartSVG(container, seriesData, xValues, config) {
        // Clear container
        container.innerHTML = '';

        // Create wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'line-chart-wrapper';
        container.appendChild(wrapper);

        // Dimensions
        const margin = { top: 40, right: 200, bottom: 60, left: 80 };
        const containerWidth = container.clientWidth || 900;
        const width = Math.max(500, containerWidth - margin.left - margin.right);
        const height = 400;

        // Create SVG
        const svg = d3.select(wrapper)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .attr('class', 'line-chart-svg');

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Find data ranges
        let minSeqLen = Infinity, maxSeqLen = -Infinity;
        let minValue = Infinity, maxValue = -Infinity;

        seriesData.forEach(series => {
            series.points.forEach(pt => {
                if (config.xType === 'numeric') {
                    if (pt.x < minSeqLen) minSeqLen = pt.x;
                    if (pt.x > maxSeqLen) maxSeqLen = pt.x;
                }
                if (pt.value < minValue) minValue = pt.value;
                if (pt.value > maxValue) maxValue = pt.value;
            });
        });

        // Add some padding to value range
        const valuePadding = (maxValue - minValue) * 0.1 || maxValue * 0.1;
        minValue = Math.max(0, minValue - valuePadding);
        maxValue = maxValue + valuePadding;

        // X scale (seq_len) - linear or log scale
        let useLogX = config.logX && config.xType === 'numeric';
        let minPositive = Infinity;
        if (useLogX) {
            seriesData.forEach(series => {
                series.points.forEach(pt => {
                    if (pt.x > 0 && pt.x < minPositive) {
                        minPositive = pt.x;
                    }
                });
            });
            if (!Number.isFinite(minPositive)) {
                useLogX = false;
            }
        }

        const xScale = config.xType === 'numeric'
            ? (useLogX
                ? d3.scaleLog()
                    .domain([minPositive, maxSeqLen])
                    .range([0, width])
                : d3.scaleLinear()
                    .domain([minSeqLen, maxSeqLen])
                    .range([0, width]))
            : d3.scalePoint()
                .domain(xValues)
                .range([0, width])
                .padding(0.5);

        // Y scale (tok/s) - linear scale
        const yScale = d3.scaleLinear()
            .domain([minValue, maxValue])
            .range([height, 0]);

        // Color scale for different series
        const colorScale = d3.scaleOrdinal()
            .domain(seriesData.map(s => s.key))
            .range(d3.schemeTableau10);

        // Add grid lines
        const yGridLines = g.append('g')
            .attr('class', 'grid y-grid')
            .call(d3.axisLeft(yScale)
                .ticks(6)
                .tickSize(-width)
                .tickFormat('')
            );

        yGridLines.selectAll('line')
            .attr('stroke', 'var(--border-subtle)')
            .attr('stroke-dasharray', '3,3');

        yGridLines.select('.domain').remove();

        // Line generator
        const lineGenerator = d3.line()
            .x(d => xScale(d.x))
            .y(d => yScale(d.value))
            .defined(d => d.value != null);

        // Draw lines and points for each series
        seriesData.forEach((series, idx) => {
            const color = colorScale(series.key);

            // Draw line connecting points
            g.append('path')
                .datum(series.points)
                .attr('class', 'line-chart-line')
                .attr('fill', 'none')
                .attr('stroke', color)
                .attr('stroke-width', 2)
                .attr('d', lineGenerator);

            // Draw data points
            g.selectAll(`.line-chart-point-${idx}`)
                .data(series.points)
                .enter()
                .append('circle')
                .attr('class', `line-chart-point line-chart-point-${idx}`)
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.value))
                .attr('r', 4)
                .attr('fill', color)
                .attr('stroke', 'var(--bg-secondary)')
                .attr('stroke-width', 1.5)
                .on('mouseover', function(event, d) {
                    showLineChartTooltip(event, d, series, config);
                    d3.select(this).attr('r', 6);
                })
                .on('mouseout', function() {
                    hideLineChartTooltip();
                    d3.select(this).attr('r', 4);
                });
        });

        // X axis
        const xAxisBuilder = config.xType === 'numeric'
            ? (useLogX
                ? d3.axisBottom(xScale).ticks(6, '~g')
                : d3.axisBottom(xScale)
                    .tickValues(xValues)
                    .tickFormat(config.xFormatter || (d => d.toString())))
            : d3.axisBottom(xScale);

        const xAxis = g.append('g')
            .attr('class', 'axis x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxisBuilder);

        // X axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('x', width / 2)
            .attr('y', height + 45)
            .attr('text-anchor', 'middle')
            .text(config.xLabel);

        // Y axis
        const yAxis = g.append('g')
            .attr('class', 'axis y-axis')
            .call(d3.axisLeft(yScale)
                .ticks(6)
                .tickFormat(d => formatMetricValue(d, config))
            );

        // Y axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -55)
            .attr('text-anchor', 'middle')
            .text(config.yLabel);

        // Title
        svg.append('text')
            .attr('class', 'chart-title')
            .attr('x', margin.left + width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .text(config.title);

        // Legend
        renderLineChartLegend(svg, seriesData, colorScale, width + margin.left + 20, margin.top, config);

        console.log(`[dashboard] Line chart rendered with ${seriesData.length} series`);
    }

    /**
     * Format sequence length for display on axis
     */
    function formatSeqLen(value) {
        if (value >= 1000) {
            return (value / 1000).toFixed(value % 1000 === 0 ? 0 : 1) + 'K';
        }
        return value.toString();
    }

    function formatLineChartXValue(value, config) {
        if (config.xFormatter && config.xType === 'numeric') {
            return config.xFormatter(value);
        }
        return value != null ? value.toString() : '--';
    }

    function formatMetricValue(value, config) {
        if (value == null) return '--';
        if (config.yUnit === 'tok/s') {
            return formatThroughput(value);
        }
        if (config.yUnit === 'ms') {
            return `${value.toFixed(2)} ms`;
        }
        return value.toString();
    }

    /**
     * Render legend for line chart
     */
    function renderLineChartLegend(svg, seriesData, colorScale, x, y, config) {
        const legendGroup = svg.append('g')
            .attr('class', 'line-chart-legend')
            .attr('transform', `translate(${x},${y})`);

        const itemHeight = 22;
        const maxItems = 15;
        const displayData = seriesData.slice(0, maxItems);
        const hasMore = seriesData.length > maxItems;

        displayData.forEach((series, idx) => {
            const itemGroup = legendGroup.append('g')
                .attr('transform', `translate(0,${idx * itemHeight})`);

            // Color swatch
            itemGroup.append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', 14)
                .attr('height', 14)
                .attr('rx', 2)
                .attr('fill', colorScale(series.key));

            // Label - abbreviated for space
            const label = buildSeriesLabel(series, config);
            itemGroup.append('text')
                .attr('x', 20)
                .attr('y', 11)
                .attr('class', 'legend-text')
                .text(label.length > 25 ? label.substring(0, 22) + '...' : label)
                .append('title')
                .text(buildSeriesLabelFull(series, config));
        });

        // Show "and N more" if truncated
        if (hasMore) {
            legendGroup.append('text')
                .attr('x', 0)
                .attr('y', displayData.length * itemHeight + 12)
                .attr('class', 'legend-more-text')
                .text(`... and ${seriesData.length - maxItems} more`);
        }
    }

    /**
     * Build abbreviated series label for legend
     */
    function buildSeriesLabel(series, config) {
        const parts = [];
        if (series.model_size) parts.push(series.model_size);
        if (series.batch_size != null && config?.xKey !== 'batch_size') {
            parts.push(`bs=${series.batch_size}`);
        }
        if (series.backend_id) {
            const shortBackend = series.backend_id.length > 8
                ? series.backend_id.substring(0, 6) + '..'
                : series.backend_id;
            parts.push(shortBackend);
        }
        if (series.token_chunk_size != null && config?.xKey !== 'token_chunk_size') {
            parts.push(`c=${series.token_chunk_size}`);
        }
        if (series.run_started_at) {
            parts.push(formatRunTimestamp(series.run_started_at));
        }
        return parts.join(' ');
    }

    /**
     * Build full series label for tooltip
     */
    function buildSeriesLabelFull(series, config) {
        const parts = [];
        if (series.model_name) parts.push(`Model: ${series.model_name}`);
        if (series.model_size) parts.push(`Size: ${series.model_size}`);
        if (series.backend_id) parts.push(`Backend: ${series.backend_id}`);
        if (series.batch_size != null && config?.xKey !== 'batch_size') {
            parts.push(`Batch: ${series.batch_size}`);
        }
        if (series.token_chunk_size != null && config?.xKey !== 'token_chunk_size') {
            parts.push(`Chunk: ${series.token_chunk_size}`);
        }
        if (series.run_started_at) {
            parts.push(`Time: ${formatRunTimestamp(series.run_started_at)}`);
        }
        return parts.join(', ');
    }

    /**
     * Show tooltip for line chart point
     */
    function showLineChartTooltip(event, point, series, config) {
        // Remove existing tooltip
        hideLineChartTooltip();

        const tooltip = document.createElement('div');
        tooltip.className = 'line-chart-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header">${escapeHtml(buildSeriesLabelFull(series, config))}</div>
            <div class="tooltip-row"><strong>Scenario:</strong> ${escapeHtml(config.scenario)}</div>
            <div class="tooltip-row"><strong>${escapeHtml(config.xLabel)}:</strong> ${formatLineChartXValue(point.x, config)}</div>
            <div class="tooltip-row"><strong>${escapeHtml(config.yLabel)}:</strong> ${formatMetricValue(point.value, config)}</div>
            <div class="tooltip-row"><strong>Batch:</strong> ${series.batch_size ?? '--'}</div>
            <div class="tooltip-row"><strong>Chunk:</strong> ${series.token_chunk_size ?? '--'}</div>
            <div class="tooltip-row"><strong>Backend:</strong> ${escapeHtml(series.backend_id || '--')}</div>
            <div class="tooltip-row"><strong>Run ID:</strong> ${escapeHtml(series.run_id || '--')}</div>
            <div class="tooltip-row"><strong>Time:</strong> ${series.run_started_at ? formatRunTimestamp(series.run_started_at) : '--'}</div>
            <div class="tooltip-row"><strong>Samples:</strong> ${point.count}</div>
        `;

        document.body.appendChild(tooltip);

        // Position tooltip
        // Force a layout read after insertion to get accurate size
        const tooltipRect = tooltip.getBoundingClientRect();
        const mouseX = event.clientX;
        const mouseY = event.clientY;

        const offset = 12;
        let left = mouseX + offset;
        let top = mouseY + offset;

        // Decide above/below based on available space
        const viewportTop = 0;
        const viewportBottom = window.innerHeight;
        const spaceBelow = viewportBottom - (mouseY + offset);
        const spaceAbove = mouseY - (viewportTop + offset);
        if (spaceBelow < tooltipRect.height && spaceAbove >= tooltipRect.height) {
            top = mouseY - tooltipRect.height - offset;
        }

        // Clamp horizontally
        const maxLeft = window.innerWidth - tooltipRect.width - offset;
        if (left > maxLeft) left = maxLeft;
        if (left < offset) left = offset;

        // Clamp vertically
        const maxTop = window.innerHeight - tooltipRect.height - offset;
        if (top > maxTop) top = maxTop;
        if (top < viewportTop + offset) top = viewportTop + offset;

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    /**
     * Hide line chart tooltip
     */
    function hideLineChartTooltip() {
        const existing = document.querySelector('.line-chart-tooltip');
        if (existing) {
            existing.remove();
        }
    }

    /**
     * Render distribution view (box plots) for prefill-mixed TTFT data
     * Shows TTFT distribution per mixed_case_id with min, p50 (median), max
     */
    function renderDistribution() {
        console.log('[dashboard] renderDistribution()');

        // Get filtered data and filter to prefill_mixed scenario only
        const allFiltered = applyFilters();
        const filtered = allFiltered.filter(m => m.scenario === 'prefill_mixed');

        if (filtered.length === 0) {
            elements.viewContainer.innerHTML = `
                <div class="empty-state">
                    <p>No prefill-mixed data available</p>
                    <p class="empty-hint">Load benchmark data with prefill_mixed scenario or adjust filters</p>
                </div>
            `;
            return;
        }

        // Group data by mixed_case_id and collect TTFT arrays
        const distributionData = aggregateDistributionData(filtered);

        if (distributionData.length === 0) {
            elements.viewContainer.innerHTML = `
                <div class="empty-state">
                    <p>No TTFT distribution data available</p>
                    <p class="empty-hint">Data must include ttft_ms_local arrays or TTFT summary stats</p>
                </div>
            `;
            return;
        }

        // Render the box plot chart
        renderDistributionSVG(distributionData);
    }

    /**
     * Aggregate distribution data from prefill_mixed measures
     * Groups by mixed_case_id and collects all TTFT values
     * @param {Array} measures - Filtered prefill_mixed measures
     * @returns {Array} Array of distribution data objects per mixed_case_id
     */
    function aggregateDistributionData(measures) {
        // Group by mixed_case_id
        const byCase = new Map();

        measures.forEach(m => {
            const caseId = m.mixed_case_id;
            if (!caseId) return;

            if (!byCase.has(caseId)) {
                byCase.set(caseId, {
                    mixed_case_id: caseId,
                    ttft_values: [],
                    model_name: m.model_name,
                    backend_id: m.backend_id,
                    batch_size: m.batch_size,
                    token_chunk_size: m.token_chunk_size_effective || m.token_chunk_size_requested,
                    count: 0
                });
            }

            const entry = byCase.get(caseId);
            entry.count++;

            // Collect TTFT values from the array if present
            if (Array.isArray(m.ttft_ms_local)) {
                entry.ttft_values.push(...m.ttft_ms_local);
            } else if (m.ttft_min_ms != null && m.ttft_p50_ms != null && m.ttft_max_ms != null) {
                // If we don't have the raw array, use the summary stats
                // This is a fallback - we'll reconstruct approximate values
                entry.ttft_values.push(m.ttft_min_ms, m.ttft_p50_ms, m.ttft_max_ms);
            }
        });

        // Compute statistics for each case
        const distributionData = [];
        byCase.forEach((entry, caseId) => {
            if (entry.ttft_values.length === 0) return;

            // Sort values for percentile calculations
            const sorted = [...entry.ttft_values].sort((a, b) => a - b);

            // Compute statistics
            const stats = {
                mixed_case_id: caseId,
                model_name: entry.model_name,
                backend_id: entry.backend_id,
                batch_size: entry.batch_size,
                token_chunk_size: entry.token_chunk_size,
                count: entry.count,
                n_values: sorted.length,
                min: sorted[0],
                q1: calcPercentile(sorted, 25),
                median: calcPercentile(sorted, 50),
                q3: calcPercentile(sorted, 75),
                max: sorted[sorted.length - 1],
                values: sorted  // Keep for potential violin/jitter
            };

            distributionData.push(stats);
        });

        // Sort by mixed_case_id for consistent ordering
        distributionData.sort((a, b) => a.mixed_case_id.localeCompare(b.mixed_case_id));

        console.log(`[dashboard] Distribution data: ${distributionData.length} cases with TTFT data`);
        return distributionData;
    }

    /**
     * Calculate percentile from sorted array
     * @param {Array} sorted - Sorted array of numbers
     * @param {number} p - Percentile (0-100)
     * @returns {number} Percentile value
     */
    function calcPercentile(sorted, p) {
        if (sorted.length === 0) return 0;
        if (sorted.length === 1) return sorted[0];

        const index = (p / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const fraction = index - lower;

        if (lower === upper) {
            return sorted[lower];
        }

        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
    }

    /**
     * Render distribution box plots using D3
     * @param {Array} data - Distribution data objects
     */
    function renderDistributionSVG(data) {
        // Clear container
        elements.viewContainer.innerHTML = '';

        // Create wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'distribution-wrapper';
        elements.viewContainer.appendChild(wrapper);

        // Dimensions
        const margin = { top: 50, right: 40, bottom: 100, left: 80 };
        const containerWidth = elements.viewContainer.clientWidth || 800;
        const width = Math.min(containerWidth - margin.left - margin.right, 900);
        const boxWidth = Math.min(60, Math.max(20, (width - 40) / data.length - 10));
        const height = 400;

        // Create SVG
        const svg = d3.select(wrapper)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .attr('class', 'distribution-svg');

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // X scale (categorical: mixed_case_id)
        const xScale = d3.scaleBand()
            .domain(data.map(d => d.mixed_case_id))
            .range([0, width])
            .padding(0.3);

        // Y scale (TTFT ms)
        const allValues = data.flatMap(d => [d.min, d.max]);
        const yMin = Math.min(...allValues) * 0.9;
        const yMax = Math.max(...allValues) * 1.1;

        const yScale = d3.scaleLinear()
            .domain([yMin, yMax])
            .range([height, 0])
            .nice();

        // Draw box plots
        data.forEach(d => {
            const x = xScale(d.mixed_case_id) + xScale.bandwidth() / 2;
            const boxHalfWidth = boxWidth / 2;

            // Vertical line (whisker) from min to max
            g.append('line')
                .attr('class', 'boxplot-whisker')
                .attr('x1', x)
                .attr('x2', x)
                .attr('y1', yScale(d.min))
                .attr('y2', yScale(d.max))
                .attr('stroke', 'var(--text-secondary)')
                .attr('stroke-width', 1);

            // Min whisker cap
            g.append('line')
                .attr('class', 'boxplot-cap')
                .attr('x1', x - boxHalfWidth * 0.5)
                .attr('x2', x + boxHalfWidth * 0.5)
                .attr('y1', yScale(d.min))
                .attr('y2', yScale(d.min))
                .attr('stroke', 'var(--text-secondary)')
                .attr('stroke-width', 1);

            // Max whisker cap
            g.append('line')
                .attr('class', 'boxplot-cap')
                .attr('x1', x - boxHalfWidth * 0.5)
                .attr('x2', x + boxHalfWidth * 0.5)
                .attr('y1', yScale(d.max))
                .attr('y2', yScale(d.max))
                .attr('stroke', 'var(--text-secondary)')
                .attr('stroke-width', 1);

            // Box (Q1 to Q3)
            const boxTop = yScale(d.q3);
            const boxBottom = yScale(d.q1);
            const boxHeight = boxBottom - boxTop;

            g.append('rect')
                .attr('class', 'boxplot-box')
                .attr('x', x - boxHalfWidth)
                .attr('y', boxTop)
                .attr('width', boxWidth)
                .attr('height', Math.max(1, boxHeight))
                .attr('fill', 'var(--accent-subtle)')
                .attr('stroke', 'var(--accent)')
                .attr('stroke-width', 1.5)
                .on('mouseover', function(event) {
                    showDistributionTooltip(event, d);
                })
                .on('mouseout', hideDistributionTooltip);

            // Median line
            g.append('line')
                .attr('class', 'boxplot-median')
                .attr('x1', x - boxHalfWidth)
                .attr('x2', x + boxHalfWidth)
                .attr('y1', yScale(d.median))
                .attr('y2', yScale(d.median))
                .attr('stroke', 'var(--accent)')
                .attr('stroke-width', 2);
        });

        // X axis
        const xAxis = g.append('g')
            .attr('class', 'axis x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale));

        // Rotate x-axis labels for readability
        xAxis.selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end')
            .attr('dx', '-0.8em')
            .attr('dy', '0.15em');

        // X axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('x', width / 2)
            .attr('y', height + 80)
            .attr('text-anchor', 'middle')
            .text('Mixed Case ID');

        // Y axis
        g.append('g')
            .attr('class', 'axis y-axis')
            .call(d3.axisLeft(yScale)
                .tickFormat(d => formatMs(d)));

        // Y axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -60)
            .attr('text-anchor', 'middle')
            .text('TTFT (ms)');

        // Title
        svg.append('text')
            .attr('class', 'chart-title')
            .attr('x', margin.left + width / 2)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .text('TTFT Distribution by Mixed Case (Prefill-Mixed)');

        // Legend
        renderDistributionLegend(svg, margin.left + width - 150, margin.top + 10);

        console.log(`[dashboard] Distribution chart rendered with ${data.length} box plots`);
    }

    /**
     * Format milliseconds for display
     * @param {number} ms - Milliseconds value
     * @returns {string} Formatted string
     */
    function formatMs(ms) {
        if (ms >= 1000) {
            return (ms / 1000).toFixed(1) + 's';
        } else if (ms >= 1) {
            return ms.toFixed(1) + 'ms';
        } else {
            return ms.toFixed(2) + 'ms';
        }
    }

    /**
     * Render legend for distribution chart
     */
    function renderDistributionLegend(svg, x, y) {
        const legendGroup = svg.append('g')
            .attr('class', 'distribution-legend')
            .attr('transform', `translate(${x},${y})`);

        // Box legend item
        legendGroup.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', 16)
            .attr('height', 16)
            .attr('fill', 'var(--accent-subtle)')
            .attr('stroke', 'var(--accent)')
            .attr('stroke-width', 1);

        legendGroup.append('text')
            .attr('x', 22)
            .attr('y', 12)
            .attr('class', 'legend-text')
            .text('Q1-Q3 (IQR)');

        // Median legend item
        legendGroup.append('line')
            .attr('x1', 0)
            .attr('x2', 16)
            .attr('y1', 30)
            .attr('y2', 30)
            .attr('stroke', 'var(--accent)')
            .attr('stroke-width', 2);

        legendGroup.append('text')
            .attr('x', 22)
            .attr('y', 34)
            .attr('class', 'legend-text')
            .text('Median (p50)');

        // Whisker legend item
        legendGroup.append('line')
            .attr('x1', 8)
            .attr('x2', 8)
            .attr('y1', 45)
            .attr('y2', 60)
            .attr('stroke', 'var(--text-secondary)')
            .attr('stroke-width', 1);

        legendGroup.append('text')
            .attr('x', 22)
            .attr('y', 56)
            .attr('class', 'legend-text')
            .text('Min-Max');
    }

    /**
     * Show tooltip for distribution box plot
     * @param {Event} event - Mouse event
     * @param {Object} d - Distribution data object
     */
    function showDistributionTooltip(event, d) {
        // Remove existing tooltip
        hideDistributionTooltip();

        const tooltip = document.createElement('div');
        tooltip.className = 'distribution-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header"><strong>${escapeHtml(d.mixed_case_id)}</strong></div>
            <div class="tooltip-row"><strong>Min:</strong> ${formatMs(d.min)}</div>
            <div class="tooltip-row"><strong>Q1 (25%):</strong> ${formatMs(d.q1)}</div>
            <div class="tooltip-row"><strong>Median (50%):</strong> ${formatMs(d.median)}</div>
            <div class="tooltip-row"><strong>Q3 (75%):</strong> ${formatMs(d.q3)}</div>
            <div class="tooltip-row"><strong>Max:</strong> ${formatMs(d.max)}</div>
            <div class="tooltip-divider"></div>
            <div class="tooltip-row"><strong>Samples:</strong> ${d.n_values}</div>
            <div class="tooltip-row"><strong>Batch:</strong> ${d.batch_size}</div>
            <div class="tooltip-row"><strong>Chunk:</strong> ${d.token_chunk_size}</div>
        `;

        document.body.appendChild(tooltip);

        // Position tooltip
        const tooltipRect = tooltip.getBoundingClientRect();
        let left = event.pageX + 10;
        let top = event.pageY + 10;

        // Keep tooltip in viewport
        if (left + tooltipRect.width > window.innerWidth) {
            left = event.pageX - tooltipRect.width - 10;
        }
        if (top + tooltipRect.height > window.innerHeight) {
            top = event.pageY - tooltipRect.height - 10;
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    /**
     * Hide distribution tooltip
     */
    function hideDistributionTooltip() {
        const existing = document.querySelector('.distribution-tooltip');
        if (existing) {
            existing.remove();
        }
    }

    /**
     * Render run comparison view
     * Allows selecting two runs and displays % deltas for matched case_ids
     */
    function renderCompare() {
        console.log('[dashboard] renderCompare()');

        const runs = state.data.runs;
        const measures = state.data.measures;

        // Check if we have enough data
        if (runs.length < 2) {
            elements.viewContainer.innerHTML = `
                <div class="empty-state">
                    <p>Need at least two runs to compare</p>
                    <p class="empty-hint">Load benchmark data from multiple runs</p>
                </div>
            `;
            return;
        }

        // Build the run selectors and comparison content
        let html = buildCompareUI(runs);

        // If both runs are selected, compute and display the comparison
        if (compareState.baseRun && compareState.compareRun) {
            const comparisonData = computeRunComparison(compareState.baseRun, compareState.compareRun, measures);
            html += buildComparisonTable(comparisonData);
        } else {
            html += `
                <div class="compare-placeholder">
                    <p>Select two runs above to compare their performance</p>
                </div>
            `;
        }

        elements.viewContainer.innerHTML = html;

        // Attach event listeners
        attachCompareListeners();

        console.log(`[dashboard] Compare view rendered with ${runs.length} runs available`);
    }

    /**
     * Build the run selector UI for comparison
     */
    function buildCompareUI(runs) {
        // Sort runs by timestamp (most recent first) if available
        const sortedRuns = [...runs].sort((a, b) => {
            const aTime = a.started_at_utc || a.timestamp || '';
            const bTime = b.started_at_utc || b.timestamp || '';
            return bTime.localeCompare(aTime);
        });

        // Build run options
        const runOptions = sortedRuns.map(run => {
            const timestamp = run.started_at_utc || run.timestamp || 'unknown time';
            const shortId = run.run_id.length > 20 ? run.run_id.substring(0, 17) + '...' : run.run_id;
            const gitInfo = run.git_sha ? ` (${run.git_sha.substring(0, 7)})` : '';
            const label = `${shortId}${gitInfo} - ${formatTimestamp(timestamp)}`;
            return { id: run.run_id, label: label };
        }).map(opt =>
            `<option value="${escapeHtml(opt.id)}">${escapeHtml(opt.label)}</option>`
        ).join('');

        const baseSelected = compareState.baseRun || '';
        const compareSelected = compareState.compareRun || '';

        return `
            <div class="compare-controls">
                <div class="compare-selectors">
                    <div class="compare-selector">
                        <label for="base-run-select">Baseline Run (older)</label>
                        <select id="base-run-select" class="compare-select">
                            <option value="">Select baseline run...</option>
                            ${runOptions}
                        </select>
                    </div>
                    <div class="compare-arrow">&#8594;</div>
                    <div class="compare-selector">
                        <label for="compare-run-select">Compare Run (newer)</label>
                        <select id="compare-run-select" class="compare-select">
                            <option value="">Select comparison run...</option>
                            ${runOptions}
                        </select>
                    </div>
                </div>
                <div class="compare-options">
                    <div class="threshold-control">
                        <label for="threshold-input">Regression Threshold (%)</label>
                        <input type="number" id="threshold-input" class="threshold-input"
                               value="${compareState.threshold}" min="0" max="100" step="1">
                    </div>
                    <div class="compare-legend">
                        <span class="legend-item regression-legend">&#9660; Regression</span>
                        <span class="legend-item improvement-legend">&#9650; Improvement</span>
                        <span class="legend-item unchanged-legend">&#8212; Unchanged</span>
                    </div>
                    <div class="compare-export">
                        <select id="export-format-select" class="export-format-select">
                            <option value="json">JSON</option>
                            <option value="markdown">Markdown</option>
                        </select>
                        <button id="export-report-btn" class="export-report-btn" title="Export comparison report">
                            Export Report
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Format timestamp for display
     */
    function formatTimestamp(timestamp) {
        if (!timestamp || timestamp === 'unknown time') return 'unknown time';
        try {
            const date = new Date(timestamp);
            return date.toLocaleString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return timestamp;
        }
    }

    /**
     * Compute comparison data between two runs
     * @param {string} baseRunId - The baseline run ID
     * @param {string} compareRunId - The comparison run ID
     * @param {Array} measures - All measure records
     * @returns {Object} Comparison results
     */
    function computeRunComparison(baseRunId, compareRunId, measures) {
        // Filter measures by run
        const baseMeasures = measures.filter(m => m.run_id === baseRunId && m.status === 'ok');
        const compareMeasures = measures.filter(m => m.run_id === compareRunId && m.status === 'ok');

        // Aggregate measures by case_id
        const baseByCase = aggregateMeasuresByCase(baseMeasures);
        const compareByCase = aggregateMeasuresByCase(compareMeasures);

        // Get all unique case_ids
        const allCaseIds = new Set([...baseByCase.keys(), ...compareByCase.keys()]);

        // Compute deltas
        const matched = [];
        const baseOnly = [];
        const compareOnly = [];

        allCaseIds.forEach(caseId => {
            const baseData = baseByCase.get(caseId);
            const compareData = compareByCase.get(caseId);

            if (baseData && compareData) {
                // Matched case - compute deltas
                const comparison = computeCaseDeltas(caseId, baseData, compareData);
                matched.push(comparison);
            } else if (baseData && !compareData) {
                // Only in base run
                baseOnly.push({ case_id: caseId, ...baseData });
            } else if (!baseData && compareData) {
                // Only in compare run
                compareOnly.push({ case_id: caseId, ...compareData });
            }
        });

        // Sort matched results
        const sorted = sortComparisonData(matched, compareState.sortColumn, compareState.sortDirection);

        // Compute summary statistics
        const regressions = matched.filter(m => m.delta_pct < -compareState.threshold);
        const improvements = matched.filter(m => m.delta_pct > compareState.threshold);
        const unchanged = matched.filter(m => Math.abs(m.delta_pct) <= compareState.threshold);

        return {
            matched: sorted,
            baseOnly,
            compareOnly,
            summary: {
                totalMatched: matched.length,
                regressions: regressions.length,
                improvements: improvements.length,
                unchanged: unchanged.length,
                avgDelta: matched.length > 0
                    ? matched.reduce((sum, m) => sum + m.delta_pct, 0) / matched.length
                    : 0
            }
        };
    }

    /**
     * Aggregate measures by case_id, computing averages
     */
    function aggregateMeasuresByCase(measures) {
        const byCase = new Map();

        measures.forEach(m => {
            const caseId = m.case_id || generateCaseId(m);
            if (!byCase.has(caseId)) {
                byCase.set(caseId, {
                    records: [],
                    scenario: m.scenario,
                    model_name: m.model_name,
                    model_size: m.model_size,
                    backend_id: m.backend_id,
                    wgpu_backend: m.wgpu_backend,
                    batch_size: m.batch_size,
                    token_chunk_size: m.token_chunk_size_effective || m.token_chunk_size_requested,
                    seq_len: m.seq_len,
                    decode_steps: m.decode_steps,
                    mixed_case_id: m.mixed_case_id
                });
            }
            byCase.get(caseId).records.push(m);
        });

        // Compute averages for each case
        byCase.forEach((caseData, caseId) => {
            const records = caseData.records;

            // Get primary throughput metric based on scenario
            if (caseData.scenario === 'decode_only') {
                caseData.throughput = average(records, 'decode_tok_per_s');
                caseData.metric_name = 'decode_tok_per_s';
            } else if (caseData.scenario === 'prefill_uniform' || caseData.scenario === 'prefill_mixed') {
                caseData.throughput = average(records, 'prefill_tok_per_s');
                caseData.metric_name = 'prefill_tok_per_s';
            }

            caseData.repeat_count = records.length;
        });

        return byCase;
    }

    /**
     * Compute deltas between base and compare case data
     */
    function computeCaseDeltas(caseId, baseData, compareData) {
        const baseThroughput = baseData.throughput || 0;
        const compareThroughput = compareData.throughput || 0;

        // Compute percentage change: ((new - old) / old) * 100
        // Positive = improvement (faster), Negative = regression (slower)
        let delta_pct = 0;
        if (baseThroughput !== 0) {
            delta_pct = ((compareThroughput - baseThroughput) / baseThroughput) * 100;
        }

        return {
            case_id: caseId,
            scenario: baseData.scenario,
            model_name: baseData.model_name,
            model_size: baseData.model_size,
            backend_id: baseData.backend_id,
            wgpu_backend: baseData.wgpu_backend,
            batch_size: baseData.batch_size,
            token_chunk_size: baseData.token_chunk_size,
            seq_len: baseData.seq_len,
            decode_steps: baseData.decode_steps,
            mixed_case_id: baseData.mixed_case_id,
            metric_name: baseData.metric_name,
            base_value: baseThroughput,
            compare_value: compareThroughput,
            delta_pct: delta_pct,
            base_repeats: baseData.repeat_count,
            compare_repeats: compareData.repeat_count
        };
    }

    /**
     * Sort comparison data
     */
    function sortComparisonData(data, column, direction) {
        const sorted = [...data];

        sorted.sort((a, b) => {
            let aVal = a[column];
            let bVal = b[column];

            // Handle nulls
            if (aVal === null || aVal === undefined) aVal = -Infinity;
            if (bVal === null || bVal === undefined) bVal = -Infinity;

            let cmp;
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                cmp = aVal - bVal;
            } else {
                cmp = String(aVal).localeCompare(String(bVal));
            }

            return direction === 'asc' ? cmp : -cmp;
        });

        return sorted;
    }

    /**
     * Build the comparison results table
     */
    function buildComparisonTable(comparisonData) {
        const { matched, baseOnly, compareOnly, summary } = comparisonData;

        // Build summary stats
        const summaryHtml = `
            <div class="compare-summary">
                <div class="summary-stat">
                    <span class="summary-label">Matched Cases</span>
                    <span class="summary-value">${summary.totalMatched}</span>
                </div>
                <div class="summary-stat regression">
                    <span class="summary-label">Regressions</span>
                    <span class="summary-value">${summary.regressions}</span>
                </div>
                <div class="summary-stat improvement">
                    <span class="summary-label">Improvements</span>
                    <span class="summary-value">${summary.improvements}</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-label">Unchanged</span>
                    <span class="summary-value">${summary.unchanged}</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-label">Avg Change</span>
                    <span class="summary-value ${summary.avgDelta < 0 ? 'regression' : (summary.avgDelta > 0 ? 'improvement' : '')}">${formatDeltaPercent(summary.avgDelta)}</span>
                </div>
            </div>
        `;

        // Build matched results table
        let matchedTableHtml = '';
        if (matched.length > 0) {
            const headerHtml = buildCompareTableHeader();
            const bodyHtml = matched.map(row => buildCompareTableRow(row)).join('');

            matchedTableHtml = `
                <div class="compare-section">
                    <h3>Comparison Results</h3>
                    <div class="table-wrapper">
                        <table class="data-table compare-table">
                            <thead>${headerHtml}</thead>
                            <tbody>${bodyHtml}</tbody>
                        </table>
                    </div>
                </div>
            `;
        }

        // Build unmatched sections
        let unmatchedHtml = '';

        if (baseOnly.length > 0 || compareOnly.length > 0) {
            unmatchedHtml = '<div class="compare-unmatched">';

            if (baseOnly.length > 0) {
                unmatchedHtml += `
                    <div class="unmatched-section">
                        <h4>Only in Baseline (${baseOnly.length} cases)</h4>
                        <ul class="unmatched-list">
                            ${baseOnly.slice(0, 10).map(c => `<li class="unmatched-item">${escapeHtml(c.case_id)}</li>`).join('')}
                            ${baseOnly.length > 10 ? `<li class="unmatched-more">... and ${baseOnly.length - 10} more</li>` : ''}
                        </ul>
                    </div>
                `;
            }

            if (compareOnly.length > 0) {
                unmatchedHtml += `
                    <div class="unmatched-section">
                        <h4>Only in Compare Run (${compareOnly.length} cases)</h4>
                        <ul class="unmatched-list">
                            ${compareOnly.slice(0, 10).map(c => `<li class="unmatched-item">${escapeHtml(c.case_id)}</li>`).join('')}
                            ${compareOnly.length > 10 ? `<li class="unmatched-more">... and ${compareOnly.length - 10} more</li>` : ''}
                        </ul>
                    </div>
                `;
            }

            unmatchedHtml += '</div>';
        }

        return summaryHtml + matchedTableHtml + unmatchedHtml;
    }

    /**
     * Build compare table header
     */
    function buildCompareTableHeader() {
        const columns = [
            { key: 'scenario', label: 'Scenario', sortable: true },
            { key: 'model_name', label: 'Model', sortable: true },
            { key: 'backend_id', label: 'Backend', sortable: true },
            { key: 'batch_size', label: 'Batch', sortable: true, numeric: true },
            { key: 'token_chunk_size', label: 'Chunk', sortable: true, numeric: true },
            { key: 'base_value', label: 'Base (tok/s)', sortable: true, numeric: true },
            { key: 'compare_value', label: 'Compare (tok/s)', sortable: true, numeric: true },
            { key: 'delta_pct', label: 'Change %', sortable: true, numeric: true }
        ];

        let headerHtml = '<tr>';
        columns.forEach(col => {
            const sortClass = col.sortable ? 'sortable' : '';
            const activeClass = col.key === compareState.sortColumn ? 'sort-active' : '';
            const dirClass = col.key === compareState.sortColumn ? `sort-${compareState.sortDirection}` : '';
            const numericClass = col.numeric ? 'numeric' : '';
            const sortIndicator = col.key === compareState.sortColumn
                ? (compareState.sortDirection === 'asc' ? ' &#9650;' : ' &#9660;')
                : '';

            headerHtml += `<th class="${sortClass} ${activeClass} ${dirClass} ${numericClass}" data-column="${col.key}">${col.label}${sortIndicator}</th>`;
        });
        headerHtml += '</tr>';

        return headerHtml;
    }

    /**
     * Build compare table row
     */
    function buildCompareTableRow(row) {
        const threshold = compareState.threshold;
        const isRegression = row.delta_pct < -threshold;
        const isImprovement = row.delta_pct > threshold;

        const rowClass = isRegression ? 'row-regression' : (isImprovement ? 'row-improvement' : '');

        // Format the change indicator
        const changeIndicator = isRegression ? '&#9660;' : (isImprovement ? '&#9650;' : '&#8212;');
        const changeClass = isRegression ? 'delta-regression' : (isImprovement ? 'delta-improvement' : 'delta-unchanged');

        // Truncate long model names
        const modelDisplay = row.model_name && row.model_name.length > 20
            ? `<span title="${escapeHtml(row.model_name)}">${escapeHtml(row.model_name.substring(0, 17))}...</span>`
            : escapeHtml(row.model_name || '--');

        return `
            <tr class="${rowClass}">
                <td>${escapeHtml(row.scenario || '--')}</td>
                <td>${modelDisplay}</td>
                <td>${escapeHtml(row.backend_id || '--')}${row.wgpu_backend ? '/' + escapeHtml(row.wgpu_backend) : ''}</td>
                <td class="numeric">${row.batch_size != null ? row.batch_size : '--'}</td>
                <td class="numeric">${row.token_chunk_size != null ? row.token_chunk_size : '--'}</td>
                <td class="numeric">${formatThroughput(row.base_value)}</td>
                <td class="numeric">${formatThroughput(row.compare_value)}</td>
                <td class="numeric ${changeClass}">
                    <span class="change-indicator">${changeIndicator}</span>
                    ${formatDeltaPercent(row.delta_pct)}
                </td>
            </tr>
        `;
    }

    /**
     * Format delta percentage for display
     */
    function formatDeltaPercent(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return '--';
        }
        const sign = value > 0 ? '+' : '';
        return `${sign}${value.toFixed(2)}%`;
    }

    /**
     * Export regression report in the specified format
     * @param {string} format - 'json' or 'markdown'
     */
    function exportRegressionReport(format) {
        console.log(`[dashboard] Exporting regression report as ${format}`);

        // Validate that we have a comparison selected
        if (!compareState.baseRun || !compareState.compareRun) {
            alert('Please select two runs to compare before exporting.');
            return;
        }

        const measures = state.data.measures;
        const comparisonData = computeRunComparison(compareState.baseRun, compareState.compareRun, measures);

        // Get run metadata
        const runsMap = new Map();
        state.data.runs.forEach(run => runsMap.set(run.run_id, run));
        const baseRunInfo = runsMap.get(compareState.baseRun) || { run_id: compareState.baseRun };
        const compareRunInfo = runsMap.get(compareState.compareRun) || { run_id: compareState.compareRun };

        // Build report data structure
        const reportData = buildReportData(comparisonData, baseRunInfo, compareRunInfo);

        // Generate content based on format
        let content, filename, mimeType;
        if (format === 'markdown') {
            content = generateMarkdownReport(reportData);
            filename = `regression-report-${formatFilenameTimestamp()}.md`;
            mimeType = 'text/markdown';
        } else {
            content = JSON.stringify(reportData, null, 2);
            filename = `regression-report-${formatFilenameTimestamp()}.json`;
            mimeType = 'application/json';
        }

        // Trigger download
        downloadFile(content, filename, mimeType);

        console.log(`[dashboard] Exported ${format} report: ${filename}`);
    }

    /**
     * Build the report data structure for export
     */
    function buildReportData(comparisonData, baseRunInfo, compareRunInfo) {
        const { matched, baseOnly, compareOnly, summary } = comparisonData;
        const threshold = compareState.threshold;

        // Categorize results
        const regressions = matched.filter(m => m.delta_pct < -threshold);
        const improvements = matched.filter(m => m.delta_pct > threshold);
        const unchanged = matched.filter(m => Math.abs(m.delta_pct) <= threshold);

        return {
            metadata: {
                generated_at: new Date().toISOString(),
                dashboard_version: '0.1',
                threshold_percent: threshold
            },
            baseline_run: {
                run_id: baseRunInfo.run_id,
                timestamp: baseRunInfo.started_at_utc || baseRunInfo.timestamp || null,
                git_sha: baseRunInfo.git_sha || null,
                git_branch: baseRunInfo.git_branch || null,
                hostname: baseRunInfo.hostname || null
            },
            compare_run: {
                run_id: compareRunInfo.run_id,
                timestamp: compareRunInfo.started_at_utc || compareRunInfo.timestamp || null,
                git_sha: compareRunInfo.git_sha || null,
                git_branch: compareRunInfo.git_branch || null,
                hostname: compareRunInfo.hostname || null
            },
            summary: {
                total_matched_cases: summary.totalMatched,
                regressions_count: summary.regressions,
                improvements_count: summary.improvements,
                unchanged_count: summary.unchanged,
                average_change_percent: parseFloat(summary.avgDelta.toFixed(4)),
                baseline_only_count: baseOnly.length,
                compare_only_count: compareOnly.length
            },
            regressions: regressions.map(formatResultForExport),
            improvements: improvements.map(formatResultForExport),
            unchanged: unchanged.map(formatResultForExport),
            unmatched: {
                baseline_only: baseOnly.map(c => c.case_id),
                compare_only: compareOnly.map(c => c.case_id)
            }
        };
    }

    /**
     * Format a comparison result for export
     */
    function formatResultForExport(row) {
        return {
            case_id: row.case_id,
            scenario: row.scenario,
            model_name: row.model_name,
            backend_id: row.backend_id,
            wgpu_backend: row.wgpu_backend || null,
            batch_size: row.batch_size,
            token_chunk_size: row.token_chunk_size,
            seq_len: row.seq_len || null,
            decode_steps: row.decode_steps || null,
            mixed_case_id: row.mixed_case_id || null,
            metric_name: row.metric_name,
            baseline_value: parseFloat(row.base_value.toFixed(4)),
            compare_value: parseFloat(row.compare_value.toFixed(4)),
            delta_percent: parseFloat(row.delta_pct.toFixed(4)),
            baseline_repeats: row.base_repeats,
            compare_repeats: row.compare_repeats
        };
    }

    /**
     * Generate a Markdown formatted regression report
     */
    function generateMarkdownReport(reportData) {
        const lines = [];

        // Header
        lines.push('# Benchmark Regression Report');
        lines.push('');
        lines.push(`Generated: ${reportData.metadata.generated_at}`);
        lines.push(`Dashboard Version: ${reportData.metadata.dashboard_version}`);
        lines.push('');

        // Run info
        lines.push('## Run Information');
        lines.push('');
        lines.push('### Baseline Run');
        lines.push(`- **Run ID**: \`${reportData.baseline_run.run_id}\``);
        if (reportData.baseline_run.timestamp) {
            lines.push(`- **Timestamp**: ${reportData.baseline_run.timestamp}`);
        }
        if (reportData.baseline_run.git_sha) {
            lines.push(`- **Git SHA**: \`${reportData.baseline_run.git_sha}\``);
        }
        if (reportData.baseline_run.git_branch) {
            lines.push(`- **Branch**: ${reportData.baseline_run.git_branch}`);
        }
        lines.push('');

        lines.push('### Compare Run');
        lines.push(`- **Run ID**: \`${reportData.compare_run.run_id}\``);
        if (reportData.compare_run.timestamp) {
            lines.push(`- **Timestamp**: ${reportData.compare_run.timestamp}`);
        }
        if (reportData.compare_run.git_sha) {
            lines.push(`- **Git SHA**: \`${reportData.compare_run.git_sha}\``);
        }
        if (reportData.compare_run.git_branch) {
            lines.push(`- **Branch**: ${reportData.compare_run.git_branch}`);
        }
        lines.push('');

        // Summary
        lines.push('## Summary');
        lines.push('');
        lines.push(`- **Regression Threshold**: ${reportData.metadata.threshold_percent}%`);
        lines.push(`- **Total Matched Cases**: ${reportData.summary.total_matched_cases}`);
        lines.push(`- **Regressions**: ${reportData.summary.regressions_count}`);
        lines.push(`- **Improvements**: ${reportData.summary.improvements_count}`);
        lines.push(`- **Unchanged**: ${reportData.summary.unchanged_count}`);
        lines.push(`- **Average Change**: ${reportData.summary.average_change_percent >= 0 ? '+' : ''}${reportData.summary.average_change_percent.toFixed(2)}%`);
        lines.push('');

        // Regressions table
        if (reportData.regressions.length > 0) {
            lines.push('## Regressions');
            lines.push('');
            lines.push('Cases that got **slower** (throughput decreased by more than threshold):');
            lines.push('');
            lines.push(buildMarkdownTable(reportData.regressions));
            lines.push('');
        }

        // Improvements table
        if (reportData.improvements.length > 0) {
            lines.push('## Improvements');
            lines.push('');
            lines.push('Cases that got **faster** (throughput increased by more than threshold):');
            lines.push('');
            lines.push(buildMarkdownTable(reportData.improvements));
            lines.push('');
        }

        // Unmatched cases
        if (reportData.unmatched.baseline_only.length > 0 || reportData.unmatched.compare_only.length > 0) {
            lines.push('## Unmatched Cases');
            lines.push('');

            if (reportData.unmatched.baseline_only.length > 0) {
                lines.push('### Only in Baseline');
                lines.push('');
                reportData.unmatched.baseline_only.forEach(caseId => {
                    lines.push(`- \`${caseId}\``);
                });
                lines.push('');
            }

            if (reportData.unmatched.compare_only.length > 0) {
                lines.push('### Only in Compare Run');
                lines.push('');
                reportData.unmatched.compare_only.forEach(caseId => {
                    lines.push(`- \`${caseId}\``);
                });
                lines.push('');
            }
        }

        return lines.join('\n');
    }

    /**
     * Build a Markdown table from result data
     */
    function buildMarkdownTable(results) {
        if (results.length === 0) return '';

        const lines = [];

        // Header
        lines.push('| Scenario | Model | Backend | Batch | Chunk | Base (tok/s) | Compare (tok/s) | Change |');
        lines.push('|----------|-------|---------|-------|-------|--------------|-----------------|--------|');

        // Rows
        results.forEach(row => {
            const backend = row.wgpu_backend ? `${row.backend_id}/${row.wgpu_backend}` : row.backend_id;
            const changeSign = row.delta_percent >= 0 ? '+' : '';
            lines.push(`| ${row.scenario} | ${row.model_name || '--'} | ${backend || '--'} | ${row.batch_size ?? '--'} | ${row.token_chunk_size ?? '--'} | ${row.baseline_value.toFixed(1)} | ${row.compare_value.toFixed(1)} | ${changeSign}${row.delta_percent.toFixed(2)}% |`);
        });

        return lines.join('\n');
    }

    /**
     * Format a timestamp for filenames (YYYYMMDD-HHMMSS)
     */
    function formatFilenameTimestamp() {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        return `${year}${month}${day}-${hours}${minutes}${seconds}`;
    }

    /**
     * Trigger a file download in the browser
     */
    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();

        // Cleanup
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }

    /**
     * Attach event listeners for compare view
     */
    function attachCompareListeners() {
        // Base run selector
        const baseSelect = document.getElementById('base-run-select');
        if (baseSelect) {
            // Set initial value
            if (compareState.baseRun) {
                baseSelect.value = compareState.baseRun;
            }

            baseSelect.addEventListener('change', (e) => {
                compareState.baseRun = e.target.value || null;
                renderCompare();
            });
        }

        // Compare run selector
        const compareSelect = document.getElementById('compare-run-select');
        if (compareSelect) {
            // Set initial value
            if (compareState.compareRun) {
                compareSelect.value = compareState.compareRun;
            }

            compareSelect.addEventListener('change', (e) => {
                compareState.compareRun = e.target.value || null;
                renderCompare();
            });
        }

        // Threshold input
        const thresholdInput = document.getElementById('threshold-input');
        if (thresholdInput) {
            thresholdInput.addEventListener('change', (e) => {
                const value = parseFloat(e.target.value);
                if (!isNaN(value) && value >= 0 && value <= 100) {
                    compareState.threshold = value;
                    renderCompare();
                }
            });
        }

        // Sortable column headers
        document.querySelectorAll('.compare-table th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const column = th.dataset.column;

                if (compareState.sortColumn === column) {
                    compareState.sortDirection = compareState.sortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    compareState.sortColumn = column;
                    // Default to desc for numeric columns
                    const isNumeric = ['batch_size', 'token_chunk_size', 'base_value', 'compare_value', 'delta_pct'].includes(column);
                    compareState.sortDirection = isNumeric ? 'desc' : 'asc';
                }

                renderCompare();
            });
        });

        // Export report button
        const exportReportBtn = document.getElementById('export-report-btn');
        if (exportReportBtn) {
            exportReportBtn.addEventListener('click', () => {
                const formatSelect = document.getElementById('export-format-select');
                const format = formatSelect ? formatSelect.value : 'json';
                exportRegressionReport(format);
            });
        }
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Expose API for debugging and integration
    window.benchDashboard = {
        getState: () => state,
        getConfig: () => config,
        clearData: clearData,
        // Helper to get filtered data (integration point for filters/views)
        getFilteredMeasures: applyFilters,
        // Get runs map for lookup
        getRunsMap: () => {
            const map = new Map();
            state.data.runs.forEach(run => map.set(run.run_id, run));
            return map;
        },
        // Filter API for programmatic access
        getFilters: () => state.filters,
        getFilterOptions: () => state.filterOptions,
        setFilter: (key, values) => {
            if (!(key in state.filters)) {
                console.warn(`[dashboard] Unknown filter key: ${key}`);
                return;
            }
            if (values === null) {
                state.filters[key] = null;
            } else if (Array.isArray(values)) {
                state.filters[key] = new Set(values);
            } else {
                state.filters[key] = new Set([values]);
            }
            renderFilterUI();
            onFiltersChanged();
        },
        clearAllFilters: clearAllFilters,
        // Export API
        exportToJsonl: exportToJsonl,
        exportRegressionReport: exportRegressionReport,
        // Compare state for programmatic access
        getCompareState: () => compareState,
        // Server files API
        getServerFilesState: () => serverFilesState,
        fetchServerFiles: fetchServerFiles,
        loadServerFile: loadServerFile,
        version: '0.1'
    };

})();
