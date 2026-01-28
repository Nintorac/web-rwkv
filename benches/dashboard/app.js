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
        const measureFields = ['scenario', 'model_name', 'model_size', 'backend_id', 'wgpu_backend',
                               'batch_size', 'token_chunk_size', 'seq_len', 'mixed_case_id'];
        measureFields.forEach(field => {
            const values = new Set();
            measures.forEach(m => {
                if (m[field] !== undefined && m[field] !== null) {
                    values.add(m[field]);
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
        const runFields = ['run_id', 'git_sha', 'timestamp'];
        runFields.forEach(field => {
            const values = new Set();
            runs.forEach(r => {
                if (r[field] !== undefined && r[field] !== null) {
                    values.add(r[field]);
                }
            });
            state.filterOptions[field] = Array.from(values).sort((a, b) => {
                // Reverse sort for timestamps (most recent first)
                if (field === 'timestamp') {
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

        // Apply each filter
        const measureFilters = ['scenario', 'model_name', 'model_size', 'backend_id', 'wgpu_backend',
                                'batch_size', 'token_chunk_size', 'seq_len', 'mixed_case_id'];
        const runFilters = ['run_id', 'git_sha', 'timestamp'];

        // Apply measure-level filters
        measureFilters.forEach(key => {
            const filterValue = state.filters[key];
            if (filterValue !== null && filterValue instanceof Set) {
                filtered = filtered.filter(m => {
                    const value = m[key];
                    if (value === undefined || value === null) {
                        return false;  // Exclude records without the field if filter is active
                    }
                    return filterValue.has(value);
                });
            }
        });

        // Apply run-level filters (filter measures by their associated run)
        runFilters.forEach(key => {
            const filterValue = state.filters[key];
            if (filterValue !== null && filterValue instanceof Set) {
                filtered = filtered.filter(m => {
                    const run = runsMap.get(m.run_id);
                    if (!run) {
                        // If we can't find the run, check if the measure has run_id directly
                        if (key === 'run_id') {
                            return filterValue.has(m.run_id);
                        }
                        return false;
                    }
                    const value = run[key];
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
        version: '0.1'
    };

})();
