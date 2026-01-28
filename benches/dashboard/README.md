# web-rwkv Benchmark Dashboard

A static HTML+JS dashboard for visualizing benchmark results from the web-rwkv benchmarking infrastructure.

## Overview

This dashboard provides interactive visualization of benchmark data stored in JSONL format. It is designed with an engineering aesthetic: neutral colors, high information density, and minimal decoration.

## Serving Locally

The dashboard is a static bundle with no build step required. Serve it with any static file server:

```bash
# Using Python (simplest)
cd benches/dashboard
python3 -m http.server 8080

# Using Node.js http-server
npx http-server benches/dashboard -p 8080

# Using PHP built-in server
php -S localhost:8080 -t benches/dashboard
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

## Usage

1. Open the dashboard in your browser
2. Drag and drop a JSONL benchmark file onto the drop zone (or click to browse)
3. Use the filter controls to narrow down the data
4. Switch between views using the tabs:
   - **Summary Table**: Sortable table of all benchmark cases
   - **Heatmap**: batch_size x seq_len colored by throughput
   - **Line Chart**: Throughput vs sequence length
   - **Compare Runs**: Side-by-side comparison of two runs

## Technology

- **Charting**: [D3.js v7](https://d3js.org/) for flexible, lightweight visualizations
- **No build step**: Pure HTML/CSS/JS, loaded via CDN

## File Structure

```
benches/dashboard/
├── index.html    # Main HTML document
├── app.js        # Application logic
├── style.css     # Styles (engineering aesthetic)
└── README.md     # This file
```

## Planned Features

The following features are planned for future tickets:

- [ ] Drag-and-drop JSONL loading with multi-file support (BD-BENCH-16)
- [ ] Filter UI with multi-select controls (BD-BENCH-17)
- [ ] Summary table with sorting and drill-down (BD-BENCH-18)
- [ ] Heatmap visualization (BD-BENCH-19)
- [ ] Line chart visualization (BD-BENCH-20)
- [ ] Run comparison with regression highlighting (BD-BENCH-21)

## Design Principles

Following the project's visual design guidelines:

**Avoid:**
- Gradient backgrounds, neon accents, glow effects
- Over-rounded cards, bento grids, glassmorphism
- Generic AI branding, chat-bubble motifs, sparkles

**Prefer:**
- Simple neutral background
- High-information density tables with strong typography
- Subtle borders/dividers, minimal shadows
- One restrained accent color (blue) for highlights
- Engineering-style charts with legible axes and clear legends
