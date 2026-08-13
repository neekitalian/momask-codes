"""
render_spectrum_tour.py — SPECTRUM TOUR

Ported from identity_preservation/scripts/render_spectrum_tour.py (Neekita Lian).
Adapted to load blended .npy files from run_zone_blend.py output structure.

A single 3D skeleton moves through N blend outputs sequentially:
  - clips play back-to-back with a short crossfade between them
  - skeleton colour morphs cool blue → warm orange as alpha/intensity rises
  - slow camera orbit over the whole piece
  - spectrum bar overlay at the bottom with a moving marker

Usage
-----
Automatic (after running run_zone_blend.py with --ablate):
    python scripts/render_spectrum_tour.py \\
        --joints_dir blend_outputs/walk_001/walk_to_dance/joints \\
        --out blend_outputs/walk_001/walk_to_dance/spectrum_tour.mp4

Manual (explicit list of .npy files):
    python scripts/render_spectrum_tour.py \\
        --files source.npy,momask.npy,blended.npy \\
        --labels "source,momask,blended" \\
        --out tour.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

CLIP_LEN         = 196
CROSSFADE_FRAMES = 5


def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--joints_dir", type=Path,
                   help="Folder of .npy files — all *.npy loaded in sorted order.")
    g.add_argument("--files", type=str,
                   help="Comma-separated list of .npy file paths.")
    ap.add_argument("--labels",  type=str, default=None,
                    help="Comma-separated clip labels (optional, defaults to filenames).")
    ap.add_argument("--out",    type=Path, default=Path("spectrum_tour.mp4"))
    ap.add_argument("--width",  type=int,   default=1920)
    ap.add_argument("--height", type=int,   default=1080)
    ap.add_argument("--fps",    type=int,   default=20)
    ap.add_argument("--dpi",    type=int,   default=120)
    ap.add_argument("--zoom",   type=float, default=2.0)
    return ap.parse_args()


def load_motions(paths: list[Path]) -> list[np.ndarray]:
    motions = []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[1:] != (22, 3):
            raise ValueError(f"Expected (T, 22, 3), got {arr.shape} in {p}")
        T = CLIP_LEN
        if arr.shape[0] > T:
            arr = arr[:T]
        elif arr.shape[0] < T:
            arr = np.concatenate([arr, np.repeat(arr[-1:], T - arr.shape[0], axis=0)])
        # Y-up → Z-up for matplotlib
        arr = arr[:, :, [0, 2, 1]]
        motions.append(arr)
        print(f"  loaded {p.name}: {arr.shape}")
    return motions


def build_timeline(motions: list[np.ndarray]):
    parts, meta = [], []
    n = len(motions)
    for i, motion in enumerate(motions):
        parts.append(motion.copy())
        for t in range(CLIP_LEN):
            meta.append((i, t / (CLIP_LEN - 1)))
        if i < n - 1:
            nxt  = motions[i + 1]
            fade = np.stack([
                (1 - (k+1)/(CROSSFADE_FRAMES+1)) * motion[-1] + (k+1)/(CROSSFADE_FRAMES+1) * nxt[0]
                for k in range(CROSSFADE_FRAMES)
            ])
            parts.append(fade)
            for k in range(CROSSFADE_FRAMES):
                meta.append((i, 1.0 + (k+1)/CROSSFADE_FRAMES))
    return np.concatenate(parts, axis=0), meta


def color_for_clip(idx: int, n: int):
    norm = idx / max(n - 1, 1)
    return plt.get_cmap("turbo")(0.15 + 0.75 * norm)


def draw_spectrum_bar(ax, current_idx: float, n: int, labels: list[str]) -> None:
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    bar_y, bar_h    = 0.5, 0.18
    bar_l, bar_r    = 0.12, 0.88
    bar_w           = bar_r - bar_l
    segs            = max(n * 20, 200)
    for i in range(segs):
        x0 = bar_l + bar_w * i / segs
        w  = bar_w / segs
        c  = color_for_clip(i / segs * (n - 1), n)
        ax.add_patch(mpatches.Rectangle((x0, bar_y - bar_h/2), w, bar_h,
                                        facecolor=c, edgecolor="none", alpha=0.85))
    ax.add_patch(mpatches.Rectangle((bar_l, bar_y - bar_h/2), bar_w, bar_h,
                                    facecolor="none", edgecolor="#e8e6ff",
                                    linewidth=0.8, alpha=0.6))
    frac     = current_idx / max(n - 1, 1)
    marker_x = bar_l + frac * bar_w
    ax.plot([marker_x, marker_x],
            [bar_y - bar_h*0.9, bar_y + bar_h*0.9],
            color="#ffffff", linewidth=2.4)
    ax.scatter([marker_x], [bar_y + bar_h*1.05], s=60,
               facecolor="#ffffff", edgecolor="none", zorder=5)
    if labels:
        ax.text(bar_l, bar_y - bar_h*1.4, labels[0],
                color="#8b8aa8", fontsize=11, ha="left", va="top",
                family="serif", style="italic")
        ax.text(bar_r, bar_y - bar_h*1.4, labels[-1],
                color="#e8b880", fontsize=11, ha="right", va="top",
                family="serif", style="italic")


def main() -> int:
    args = parse_args()

    if args.joints_dir:
        paths = sorted(args.joints_dir.glob("*.npy"))
        if not paths:
            print(f"No .npy files in {args.joints_dir}")
            return 1
    else:
        paths = [Path(p.strip()) for p in args.files.split(",")]

    labels = ([l.strip() for l in args.labels.split(",")]
              if args.labels else [p.stem for p in paths])
    n = len(paths)
    print(f"[tour] {n} clips: {[p.name for p in paths]}")

    motions = load_motions(paths)
    full_motion, meta = build_timeline(motions)
    N = full_motion.shape[0]
    print(f"[tour] total frames: {N} (~{N/args.fps:.1f}s)")

    mins   = full_motion.reshape(-1, 3).min(0)
    maxs   = full_motion.reshape(-1, 3).max(0)
    center = (mins + maxs) / 2
    char_h = float(maxs[2] - mins[2])
    radius = char_h * 0.7 / max(0.1, args.zoom)
    z_half = char_h * 0.65 / max(0.1, args.zoom)

    fig_w, fig_h = args.width / args.dpi, args.height / args.dpi
    fig      = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi, facecolor="#0a0a12")
    ax_3d    = fig.add_axes([0, 0.14, 1, 0.86], projection="3d")
    ax_bar   = fig.add_axes([0, 0.00, 1, 0.14])

    def setup_scene():
        ax_3d.set_facecolor("#0f0f18")
        ax_3d.grid(False)
        for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
            pane.set_facecolor((0.06, 0.06, 0.10, 1.0))
            pane.set_edgecolor((0, 0, 0, 0))
        for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
            axis.set_ticklabels([]); axis.set_ticks([]); axis.line.set_visible(False)

    def draw(frame: int):
        ax_3d.cla()
        setup_scene()

        t_all = frame / max(N - 1, 1)
        ax_3d.view_init(elev=12 + 4*np.sin(t_all*np.pi), azim=-75 + 45*t_all)
        ax_3d.set_xlim(center[0]-radius, center[0]+radius)
        ax_3d.set_ylim(center[1]-radius, center[1]+radius)
        ax_3d.set_zlim(center[2]-z_half, center[2]+z_half)
        ax_3d.set_box_aspect((1, 1, 1))

        clip_idx, in_clip = meta[frame]
        if in_clip > 1.0:
            w         = in_clip - 1.0
            eff_idx   = clip_idx + w
        else:
            eff_idx   = float(clip_idx)
        color = color_for_clip(eff_idx, n)

        j = full_motion[frame]
        for chain in KINEMATIC_CHAIN:
            ax_3d.plot(j[chain, 0], j[chain, 1], j[chain, 2],
                       color=color, linewidth=2.6, solid_capstyle="round", alpha=0.95)
        ax_3d.scatter(j[:, 0], j[:, 1], j[:, 2], s=14, color=color, edgecolors="none")

        clip_label = labels[min(int(clip_idx), n-1)]
        ax_3d.text2D(0.02, 0.95, f"C L I P   {int(clip_idx)+1}/{n}",
                     transform=ax_3d.transAxes,
                     color="#e8e6ff", alpha=0.85, fontsize=13,
                     weight="light", family="serif")
        ax_3d.text2D(0.02, 0.92, clip_label,
                     transform=ax_3d.transAxes,
                     color="#8b8aa8", alpha=0.75, fontsize=11,
                     style="italic", family="serif")

        draw_spectrum_bar(ax_bar, eff_idx, n, labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tour] rendering → {args.out}")
    ani    = FuncAnimation(fig, draw, frames=N, interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=6000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    ani.save(str(args.out), writer=writer, dpi=args.dpi,
             savefig_kwargs={"facecolor": "#0a0a12"})
    plt.close(fig)
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
