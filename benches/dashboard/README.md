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
2. Load benchmark data using one of two methods:
   - **Drag and drop**: Drop a JSONL benchmark file onto the drop zone (or click to browse)
   - **Server loading**: Click files in the "Load from Server" panel (requires server configuration)
3. Use the filter controls to narrow down the data
4. Switch between views using the tabs:
   - **Summary Table**: Sortable table of all benchmark cases
   - **Heatmap**: batch_size x seq_len colored by throughput
   - **Line Chart**: Throughput vs sequence length
   - **Compare Runs**: Side-by-side comparison of two runs

## Server-Side File Loading

The dashboard can load JSONL files directly from the server, which is useful for deployed dashboards with pre-existing benchmark results.

### Setup

1. **Create the data directory structure:**
   ```
   dashboard/
   ├── index.html
   ├── app.js
   ├── style.css
   └── data/
       ├── index.json       # Manifest file (required)
       ├── bench_001.jsonl  # Benchmark results
       ├── bench_002.jsonl
       └── ...
   ```

2. **Generate the manifest:**
   ```bash
   python scripts/gen_manifest.py /path/to/results/directory data/
   ```

   Or create `data/index.json` manually:
   ```json
   {
     "files": [
       {"name": "bench_smoke_20260128.jsonl", "size": 3100, "modified": "2026-01-28T10:40:00Z"},
       {"name": "bench_full_20260127.jsonl", "size": 45000, "modified": "2026-01-27T15:30:00Z"}
     ]
   }
   ```

3. **Serve the dashboard** with a static file server.

### nginx Configuration

For production deployments, configure nginx to serve the dashboard and results:

```nginx
server {
    listen 80;
    server_name bench.example.com;

    # Dashboard static files
    location / {
        alias /var/www/dashboard/;
        index index.html;
        try_files $uri $uri/ =404;
    }

    # Benchmark data directory
    location /data/ {
        alias /var/www/dashboard/data/;
        autoindex off;

        # CORS headers (if accessing from different origin)
        add_header Access-Control-Allow-Origin *;

        # Cache manifest briefly, cache JSONL files longer
        location ~* index\.json$ {
            expires 1m;
        }
        location ~* \.jsonl$ {
            expires 1h;
        }
    }
}
```

### Automatic Manifest Generation

Use the included script to regenerate the manifest when new results are added:

```bash
# One-time generation
python scripts/gen_manifest.py /var/www/dashboard/data

# As a cron job (every hour)
0 * * * * python /path/to/scripts/gen_manifest.py /var/www/dashboard/data
```

## Technology

- **Charting**: [D3.js v7](https://d3js.org/) for flexible, lightweight visualizations
- **No build step**: Pure HTML/CSS/JS, loaded via CDN

## File Structure

```
benches/dashboard/
├── index.html              # Main HTML document
├── app.js                  # Application logic
├── style.css               # Styles (engineering aesthetic)
├── README.md               # This file
├── data/                   # Server-side data directory
│   └── index.json          # Sample manifest file
└── scripts/
    └── gen_manifest.py     # Manifest generation script
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
