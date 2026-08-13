"""
Stage 2.4 driver: score a batch of pipeline outputs and print a generalizability report.

Feed it a manifest describing (original, output) joint pairs tagged by genre / verb /
performer. Each joint file is a .npy of shape (T, 22, 3). No torch, no GPU.

Manifest (JSON list):
    [
      {"original": "runs/p1_walk_orig.npy", "output": "runs/p1_walk_ballet.npy",
       "genre": "ballet", "verb": "walk", "performer": "p1"},
      ...
    ]

Usage:
    python scripts/evaluate_generalizability.py --manifest runs/manifest.json
    python scripts/evaluate_generalizability.py --manifest runs/manifest.json --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_spectrum.evaluate import (
    evaluate_matrix, generalizability_report, format_report)


def _load(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[1] == 66:
        arr = arr.reshape(arr.shape[0], 22, 3)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        raise ValueError(f"{path}: expected (T, 22, 3), got {arr.shape}")
    return arr.astype(np.float32)


def load_manifest(path: str | Path) -> list[dict]:
    entries = json.loads(Path(path).read_text())
    records = []
    for e in entries:
        records.append({
            "original": _load(e["original"]),
            "output": _load(e["output"]),
            "genre": e.get("genre", ""),
            "verb": e.get("verb", ""),
            "performer": e.get("performer", ""),
        })
    return records


def main():
    ap = argparse.ArgumentParser(description="Generalizability report over pipeline outputs.")
    ap.add_argument("--manifest", required=True, help="JSON manifest of original/output pairs.")
    ap.add_argument("--out", default=None, help="Optional JSON path for the full report.")
    args = ap.parse_args()

    records = load_manifest(args.manifest)
    metrics = evaluate_matrix(records)
    report = generalizability_report(metrics)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
