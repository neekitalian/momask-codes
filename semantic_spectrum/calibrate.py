"""
Stage 2.3 - bin spectrum scores and calibrate from participant feedback.

Two jobs:

1. Binning
   Continuous z scores are snapped onto the discrete mode grid the fine-tuned model
   was swept over (video_to_spectrum sweeps the style term across a fixed set of
   steps). Binning keeps estimated z coordinates on the grid the model actually saw.

2. Feedback calibration
   The Round 2 perceptual study rated rendered clips on identity preservation and
   expressivity, at three style-allocation levels (20 / 40 / 60) for three genres.
   Higher allocation transfers more style but tends to erode identity. calibrate_alpha
   reads the summary CSV and, per genre, picks the allocation that best trades the two,
   returning the implied blending alpha for the Stage 2.1 pipeline.

The recalibrated per-genre alpha is written to JSON with write_calibration(); the actual
model retrain that consumes rebinned labels runs on GPU (train_t2m_transformer.py /
train_res_transformer.py) and is out of scope for this pure-kinematics module.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allocation levels used in the study, as fractions of full style strength.
STUDY_ALLOCATIONS = [20, 40, 60]


# ─────────────────────────────────────────────────────────────────────
# 1. Binning
# ─────────────────────────────────────────────────────────────────────

def mode_grid(n_steps: int = 10, lo: float = 0.09, hi: float = 1.0) -> np.ndarray:
    """The discrete style-term grid the model is swept over (default: 0.09 .. 1.00)."""
    return np.linspace(lo, hi, n_steps)


def snap_to_grid(z: float, grid: Optional[np.ndarray] = None) -> float:
    """Snap a continuous style value to the nearest grid step."""
    grid = mode_grid() if grid is None else grid
    return float(grid[int(np.argmin(np.abs(grid - z)))])


def bin_z(z: float, n_bins: int = 10) -> int:
    """Index of the equal-width bin (0 .. n_bins-1) that z falls in, z in [0, 1]."""
    z = float(np.clip(z, 0.0, 1.0 - 1e-9))
    return int(z * n_bins)


# ─────────────────────────────────────────────────────────────────────
# 2. Feedback calibration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GenreCalibration:
    genre: str
    allocation: int          # chosen study allocation level (20 / 40 / 60)
    alpha: float             # implied blending alpha in [0, 1]
    identity: float          # mean identity preservation at that allocation
    expressivity: float      # mean expressivity at that allocation
    per_allocation: Dict[int, Dict[str, float]]   # full trade-off curve


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_feedback(csv_path: str | Path) -> List[dict]:
    """Load the rendered rows of the Round 2 summary CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("stimulus_type") != "rendered":
                continue
            alloc = _to_float(r.get("allocation_level"))
            ident = _to_float(r.get("mean_identity_preservation"))
            expr = _to_float(r.get("mean_expressivity"))
            n = _to_float(r.get("n")) or 1.0
            if alloc is None or ident is None or expr is None:
                continue
            rows.append({"genre": r["genre"], "allocation": int(alloc),
                         "identity": ident, "expressivity": expr, "n": n})
    if not rows:
        raise ValueError(f"No usable rendered rows in {csv_path}")
    return rows


def _aggregate(rows: List[dict]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """genre -> allocation -> n-weighted mean identity / expressivity across source actions."""
    acc: Dict[str, Dict[int, List[dict]]] = {}
    for r in rows:
        acc.setdefault(r["genre"], {}).setdefault(r["allocation"], []).append(r)
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for genre, by_alloc in acc.items():
        out[genre] = {}
        for alloc, rs in by_alloc.items():
            w = np.array([x["n"] for x in rs], dtype=float)
            wi = float(np.average([x["identity"] for x in rs], weights=w))
            we = float(np.average([x["expressivity"] for x in rs], weights=w))
            out[genre][alloc] = {"identity": round(wi, 3), "expressivity": round(we, 3)}
    return out


def calibrate_alpha(
    csv_path: str | Path,
    objective: str = "balanced",
    identity_floor: float = 5.0,
    weight_identity: float = 0.5,
) -> Dict[str, GenreCalibration]:
    """
    Recommend a per-genre blending alpha from the perceptual study.

    objective
      "expressivity" : maximise expressivity subject to identity >= identity_floor.
      "identity"     : maximise identity preservation.
      "balanced"     : maximise weight_identity * norm(identity)
                       + (1 - weight_identity) * norm(expressivity),
                       where each metric is min-max normalised across all observations.

    identity_floor is on the study's rating scale (identity preservation ratings run
    to ~7). alpha is the chosen allocation as a fraction (allocation / 100).
    """
    agg = _aggregate(load_feedback(csv_path))

    # global min-max ranges for the balanced objective
    all_i = [v["identity"] for g in agg.values() for v in g.values()]
    all_e = [v["expressivity"] for g in agg.values() for v in g.values()]
    i_lo, i_hi = min(all_i), max(all_i)
    e_lo, e_hi = min(all_e), max(all_e)
    norm = lambda x, lo, hi: 0.0 if hi <= lo else (x - lo) / (hi - lo)

    result: Dict[str, GenreCalibration] = {}
    for genre, by_alloc in agg.items():
        allocs = sorted(by_alloc)

        def score(alloc: int) -> float:
            v = by_alloc[alloc]
            if objective == "identity":
                return v["identity"]
            if objective == "expressivity":
                # penalise anything under the identity floor so it never wins
                pen = 0.0 if v["identity"] >= identity_floor else -1e6 + v["identity"]
                return v["expressivity"] + pen
            return (weight_identity * norm(v["identity"], i_lo, i_hi)
                    + (1.0 - weight_identity) * norm(v["expressivity"], e_lo, e_hi))

        best = max(allocs, key=score)
        v = by_alloc[best]
        result[genre] = GenreCalibration(
            genre=genre, allocation=best, alpha=round(best / 100.0, 3),
            identity=v["identity"], expressivity=v["expressivity"],
            per_allocation={a: by_alloc[a] for a in allocs},
        )
    return result


def write_calibration(path: str | Path, calibration: Dict[str, GenreCalibration]) -> None:
    """Write the per-genre alpha table to JSON for the pipeline / retrain to consume."""
    payload = {g: {"alpha": c.alpha, "allocation": c.allocation,
                   "identity": c.identity, "expressivity": c.expressivity}
               for g, c in calibration.items()}
    Path(path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":   # pragma: no cover - thin CLI
    import argparse
    ap = argparse.ArgumentParser(description="Calibrate per-genre alpha from the perceptual study.")
    ap.add_argument("csv", help="Path to round2_summary.csv")
    ap.add_argument("--objective", default="balanced", choices=["balanced", "identity", "expressivity"])
    ap.add_argument("--identity-floor", type=float, default=5.0)
    ap.add_argument("--weight-identity", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    cal = calibrate_alpha(args.csv, objective=args.objective,
                          identity_floor=args.identity_floor, weight_identity=args.weight_identity)
    for genre, c in sorted(cal.items()):
        print(f"{genre:8s} alpha={c.alpha:.2f} (alloc {c.allocation})  "
              f"identity={c.identity:.2f}  expressivity={c.expressivity:.2f}")
    if args.out:
        write_calibration(args.out, cal)
        print(f"wrote {args.out}")
