#!/usr/bin/env python3
"""
Generate index.json manifest for the dashboard server-side file loading feature.

Usage:
    python gen_manifest.py /path/to/results/directory [output_path]

Arguments:
    results_dir  Directory containing .jsonl benchmark result files
    output_path  Output path for index.json (default: results_dir/index.json)

Example:
    python gen_manifest.py ./results
    python gen_manifest.py /data/benchmarks /var/www/dashboard/data/index.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def get_file_info(filepath: Path) -> dict:
    """Get file information for manifest entry."""
    stat = filepath.stat()
    return {
        "name": filepath.name,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
    }


def generate_manifest(results_dir: Path, output_path: Path = None) -> dict:
    """
    Generate manifest from a directory of JSONL files.

    Args:
        results_dir: Directory containing .jsonl files
        output_path: Where to write the manifest (default: results_dir/index.json)

    Returns:
        The manifest dictionary
    """
    if not results_dir.is_dir():
        raise ValueError(f"Not a directory: {results_dir}")

    # Find all .jsonl files
    jsonl_files = sorted(results_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Build file list
    files = [get_file_info(f) for f in jsonl_files]

    manifest = {
        "files": files,
        "generated": datetime.utcnow().isoformat() + "Z",
        "count": len(files)
    }

    # Determine output path
    if output_path is None:
        output_path = results_dir / "index.json"

    # Write manifest
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest with {len(files)} file(s): {output_path}")

    return manifest


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        generate_manifest(results_dir, output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
