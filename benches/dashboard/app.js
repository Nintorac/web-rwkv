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
        filters: {
            scenario: null,
            model_name: null,
            backend_id: null,
            batch_size: null,
            seq_len: null
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
            }
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
     * Update filter UI based on loaded data
     */
    function updateFilters() {
        // Placeholder: will be implemented in BD-BENCH-17
        console.log('[dashboard] updateFilters() - Not yet implemented');
    }

    /**
     * Apply current filters to data
     * @returns {Array} Filtered measure records
     */
    function applyFilters() {
        // Placeholder: will be implemented in BD-BENCH-17
        console.log('[dashboard] applyFilters() - Not yet implemented');
        return state.data.measures;
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

        console.log(`[dashboard] Stats updated: ${runs.length} runs, ${measures.length} cases, ${uniqueBackends.size} backends, ${uniqueModels.size} models`);
    }

    /**
     * Render summary table view
     */
    function renderTable() {
        // Placeholder: will be implemented in later tickets
        console.log('[dashboard] renderTable() - Not yet implemented');
        elements.viewContainer.innerHTML = `
            <div class="empty-state">
                <p>Summary table view</p>
                <p class="empty-hint">Coming soon</p>
            </div>
        `;
    }

    /**
     * Render heatmap view using D3
     */
    function renderHeatmap() {
        // Placeholder: will be implemented in later tickets
        console.log('[dashboard] renderHeatmap() - Not yet implemented');
        elements.viewContainer.innerHTML = `
            <div class="empty-state">
                <p>Heatmap view</p>
                <p class="empty-hint">Coming soon</p>
            </div>
        `;
    }

    /**
     * Render line chart view using D3
     */
    function renderLineChart() {
        // Placeholder: will be implemented in later tickets
        console.log('[dashboard] renderLineChart() - Not yet implemented');
        elements.viewContainer.innerHTML = `
            <div class="empty-state">
                <p>Line chart view</p>
                <p class="empty-hint">Coming soon</p>
            </div>
        `;
    }

    /**
     * Render run comparison view
     */
    function renderCompare() {
        // Placeholder: will be implemented in later tickets
        console.log('[dashboard] renderCompare() - Not yet implemented');
        elements.viewContainer.innerHTML = `
            <div class="empty-state">
                <p>Run comparison view</p>
                <p class="empty-hint">Coming soon</p>
            </div>
        `;
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
        version: '0.1'
    };

})();
