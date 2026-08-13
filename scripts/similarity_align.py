"""
similarity_align.py — DTW-based identity similarity metric between two motion sequences.

Ported from identity_preservation/scripts/similarity_align.py (Neekita Lian).
Adapted to use motion_io.py instead of webcam_spectrum.py.

Given:
  --original   .npy / .npz / .mp4 — identity anchor (source / performer)
  --generated  .npy / .npz / .mp4 — output to measure (MoMask, blended, etc.)

Outputs in --output-dir:
  alignment.json       matched frame pairs + per-pair L2 distance
  distance_matrix.png  heatmap of pairwise pose distances + DTW path overlay
  side_by_side.mp4     two-panel skeleton video playing aligned frames in sync
  report.txt           summary stats (use avg_per_pair to compare runs)

Usage
-----
    python scripts/similarity_align.py \\
        --original  recordings/walk.mp4 \\
        --generated blend_outputs/walk_001/walk_to_dance/joints/blended.npy \\
        --output-dir blend_outputs/walk_001/walk_to_dance/dtw

Lower avg_per_pair distance = higher identity retention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.motion_io import load_joints                    # noqa: E402
from utils.paramUtil import t2m_kinematic_chain              # noqa: E402


# ---------------------------------------------------------------------------
# Pose features + DTW
# ---------------------------------------------------------------------------

def pose_features(js: np.ndarray) -> np.ndarray:
    """(T, 22, 3) → (T, 66) descriptor centred at pelvis, normalised by torso length."""
    centred = js - js[:, 0:1, :]
    torso   = np.linalg.norm(js[:, 12] - js[:, 0], axis=-1)
    scale   = np.clip(torso, 1e-6, None)[:, None, None]
    normed  = centred / scale
    return normed.reshape(normed.shape[0], -1).astype(np.float32)


def distance_matrix(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """Pairwise L2 between every pose in a and every pose in b."""
    return np.linalg.norm(feat_a[:, None, :] - feat_b[None, :, :], axis=-1).astype(np.float32)


def dtw(cost: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Classic DTW. Returns (path, total_cost)."""
    Ta, Tb = cost.shape
    INF = np.float32(np.inf)
    dp     = np.full((Ta + 1, Tb + 1), INF, dtype=np.float32)
    parent = np.zeros((Ta + 1, Tb + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            diag, up, left = dp[i-1, j-1], dp[i-1, j], dp[i, j-1]
            best = min(diag, up, left)
            dp[i, j] = cost[i-1, j-1] + best
            parent[i, j] = 0 if best == diag else (1 if best == up else 2)
    path = []
    i, j = Ta, Tb
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = parent[i, j]
        if k == 0:   i, j = i-1, j-1
        elif k == 1: i -= 1
        else:        j -= 1
    return path[::-1], float(dp[Ta, Tb])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_skeleton(ax, joints: np.ndarray, color: str, lw: float = 2.4) -> None:
    ax.cla()
    ax.set_facecolor("#0f0f18")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.06, 0.06, 0.10, 1.0))
        pane.set_edgecolor((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([]); axis.set_ticks([]); axis.line.set_visible(False)
    j = joints[:, [0, 2, 1]]   # Y-up → Z-up for matplotlib
    for chain in t2m_kinematic_chain:
        ax.plot(j[chain, 0], j[chain, 1], j[chain, 2], color=color, linewidth=lw)
    ax.scatter(j[:, 0], j[:, 1], j[:, 2], s=12, color=color)


def render_heatmap(cost: np.ndarray, path: list, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(cost, cmap="magma", aspect="auto", origin="lower")
    ax.plot([p[1] for p in path], [p[0] for p in path],
            color="#e8b880", linewidth=2, label="DTW path")
    ax.set_xlabel("generated frame")
    ax.set_ylabel("original frame")
    ax.set_title("pose-distance matrix + optimal alignment")
    ax.legend(loc="lower right", facecolor="black", labelcolor="white")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#0a0a12")
    plt.close(fig)
    print(f"[out] {out_png}")


def render_side_by_side(orig: np.ndarray, gen: np.ndarray, path: list,
                        out_mp4: Path, fps: int = 20,
                        width: int = 1600, height: int = 720, dpi: int = 120) -> None:
    fig_w, fig_h = width / dpi, height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0a0a12")
    ax_l = fig.add_subplot(121, projection="3d", facecolor="#0f0f18")
    ax_r = fig.add_subplot(122, projection="3d", facecolor="#0f0f18")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.94, wspace=0.05)

    mins   = np.minimum(orig.reshape(-1, 3).min(0), gen.reshape(-1, 3).min(0))
    maxs   = np.maximum(orig.reshape(-1, 3).max(0), gen.reshape(-1, 3).max(0))
    center = (mins + maxs) / 2
    radius = float(np.max(maxs - mins)) * 0.55

    def setup(ax, title, color):
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius * 0.05, center[1] + radius * 1.2)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-75)
        ax.set_title(title, color=color, fontsize=13, weight="light", family="serif")

    def draw(t):
        oi, gi = path[t]
        draw_skeleton(ax_l, orig[oi], "#5cd6ff")
        draw_skeleton(ax_r, gen[gi],  "#e8b880")
        setup(ax_l, f"ORIGINAL   frame {oi:3d}", "#e8e6ff")
        setup(ax_r, f"GENERATED  frame {gi:3d}", "#e8b880")
        fig.suptitle(f"aligned pair {t+1}/{len(path)}",
                     color="#8b8aa8", fontsize=12, style="italic", family="serif", y=0.98)

    print(f"[out] rendering side_by_side.mp4 ({len(path)} aligned frames) ...")
    ani    = FuncAnimation(fig, draw, frames=len(path), interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=5000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    ani.save(str(out_mp4), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "#0a0a12"})
    plt.close(fig)
    print(f"[out] {out_mp4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_alignment(original: str | Path, generated: str | Path,
                  output_dir: str | Path,
                  fps: int = 20, width: int = 1600, height: int = 720) -> dict:
    """
    Compute DTW alignment between two motion sequences and write outputs.
    Returns the alignment summary dict (same as alignment.json).
    Can be called programmatically or via CLI.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = load_joints(Path(original),  "original")
    gen  = load_joints(Path(generated), "generated")
    print(f"[shapes] original={orig.shape}  generated={gen.shape}")

    feat_o = pose_features(orig)
    feat_g = pose_features(gen)

    print("[dtw] building distance matrix ...")
    cost = distance_matrix(feat_o, feat_g)
    print(f"[dtw] dist min={cost.min():.3f}  max={cost.max():.3f}  mean={cost.mean():.3f}")

    print("[dtw] running dynamic time warping ...")
    path, total = dtw(cost)
    per_pair    = total / max(len(path), 1)
    print(f"[dtw] pairs={len(path)}  total={total:.2f}  avg_per_pair={per_pair:.3f}")

    summary = {
        "num_pairs":       len(path),
        "total_cost":      total,
        "avg_per_pair":    per_pair,
        "shape_original":  list(orig.shape),
        "shape_generated": list(gen.shape),
        "pairs": [
            {"t": t, "original_frame": int(i), "generated_frame": int(j),
             "distance": float(cost[i, j])}
            for t, (i, j) in enumerate(path)
        ],
    }
    (out_dir / "alignment.json").write_text(json.dumps(summary, indent=2))
    print(f"[out] {out_dir / 'alignment.json'}")

    render_heatmap(cost, path, out_dir / "distance_matrix.png")
    render_side_by_side(orig, gen, path, out_dir / "side_by_side.mp4",
                        fps=fps, width=width, height=height)

    report = (
        f"similarity_align.py — report\n"
        f"{'='*40}\n"
        f"original  : {original}\n"
        f"generated : {generated}\n\n"
        f"sequence lengths      : original={orig.shape[0]}  generated={gen.shape[0]}\n"
        f"pose-distance         : min={cost.min():.3f}  max={cost.max():.3f}  mean={cost.mean():.3f}\n"
        f"DTW alignment         : {len(path)} pairs\n"
        f"total alignment cost  : {total:.2f}\n"
        f"average per-pair cost : {per_pair:.3f}\n\n"
        f"lower avg_per_pair => higher identity retention.\n"
        f"identity retention ~= 1 - avg_per_pair / mean_dist_cost\n"
    )
    (out_dir / "report.txt").write_text(report)
    print(f"[out] {out_dir / 'report.txt'}")
    print(f"\n[done] outputs at {out_dir}")
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description="DTW identity similarity metric for motion sequences.")
    ap.add_argument("--original",   required=True, help=".npy/.npz/.mp4 — identity anchor")
    ap.add_argument("--generated",  required=True, help=".npy/.npz/.mp4 — output to measure")
    ap.add_argument("--output-dir", default="dtw_output")
    ap.add_argument("--fps",    type=int, default=20)
    ap.add_argument("--width",  type=int, default=1600)
    ap.add_argument("--height", type=int, default=720)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_alignment(args.original, args.generated, args.output_dir,
                  fps=args.fps, width=args.width, height=args.height)
