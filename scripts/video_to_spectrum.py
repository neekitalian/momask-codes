"""
video_to_spectrum.py — Full pipeline: video → MediaPipe → MoMask → quad + reel

Usage:
    # Drop a video into input_videos/ then run:
    python scripts/video_to_spectrum.py

    # Or name the file explicitly (just the filename, not the full path):
    python scripts/video_to_spectrum.py --video my_dance.mp4

What it does:
  1. Runs MediaPipe on the input video to produce momask_input.npz
  2. Runs edit_t2m.py 10 times, sweeping dance intensity across mode steps
     while keeping walk fixed at 0.91
  3. Saves all 10 generated mp4s into outputs/<video_stem>/
  4. Concatenates them into a single progression reel  (11th file)
  5. Builds a 2×5 quad grid video of all 10 clips side by side (12th file)

Output folder layout:
    outputs/<video_stem>/
        dance_mode_0.09_repeat0.mp4   (first repeat of each step)
        dance_mode_0.11_repeat0.mp4
        ...
        dance_mode_1.00_repeat0.mp4
        progression.mp4               (all 10 in sequence)
        quad.mp4                      (2x5 grid, all 10 at once)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dance intensity steps (mode-based from spectrum score distribution analysis)
# ---------------------------------------------------------------------------
STEPS = [
    {'dance': 0.09, 'walk': 0.91, 'prompt': '[dance:0.09][walk:0.91] A person walks forward.'},
    {'dance': 0.11, 'walk': 0.91, 'prompt': '[dance:0.11][walk:0.91] A person walks forward.'},
    {'dance': 0.28, 'walk': 0.91, 'prompt': '[dance:0.28][walk:0.91] A person dances with a walking base.'},
    {'dance': 0.39, 'walk': 0.91, 'prompt': '[dance:0.39][walk:0.91] A person dances with a walking base.'},
    {'dance': 0.40, 'walk': 0.91, 'prompt': '[dance:0.40][walk:0.91] A person dances with a walking base.'},
    {'dance': 0.51, 'walk': 0.91, 'prompt': '[dance:0.51][walk:0.91] A person dances with a walking base.'},
    {'dance': 0.63, 'walk': 0.91, 'prompt': '[dance:0.63][walk:0.91] A person dances with a walking base.'},
    {'dance': 0.74, 'walk': 0.91, 'prompt': '[dance:0.74][walk:0.91] A person dances with a walking base.'},
    {'dance': 0.80, 'walk': 0.91, 'prompt': '[dance:0.80][walk:0.91] A person dances expressively.'},
    {'dance': 1.00, 'walk': 0.91, 'prompt': '[dance:1.00][walk:0.91] A person dances expressively.'},
]

MASK_NAME    = 'MaskTransformer'
RES_NAME     = 'ResTransformer'
REPEAT_TIMES = 3


def build_steps(custom_text: str | None) -> list[dict]:
    """Return the 10 intensity steps, optionally overriding the prompt text tail."""
    if custom_text is None:
        return STEPS
    return [
        {**s, 'prompt': f"[dance:{s['dance']:.2f}][walk:{s['walk']:.2f}] {custom_text}"}
        for s in STEPS
    ]


# Stage 2.2 z integration. Guarded so the base pipeline still runs if the
# semantic_spectrum deps (or torch, pulled in by its package __init__) are absent.
try:
    from semantic_spectrum.estimate_z import estimate_z, _load_joints
except Exception:          # pragma: no cover - optional dependency
    estimate_z = None
    _load_joints = None


def apply_estimated_base(steps: list[dict], npz_path) -> list[dict]:
    """
    Replace the fixed [walk:0.91] base term with one estimated from the recording
    (Stage 2.2). The style (dance) axis still sweeps; only the base verb and its
    weight are read off the performer's motion.
    """
    if estimate_z is None:
        print('  --auto-z requested but semantic_spectrum.estimate_z is unavailable; keeping fixed base.')
        return steps
    try:
        est = estimate_z(_load_joints(npz_path))
    except Exception as e:                                    # keep the pipeline running
        print(f'  --auto-z: estimation failed ({e}); keeping fixed base.')
        return steps
    base, bw = est.base, round(1.0 - est.z, 2)
    print(f'  Estimated base: {base}:{bw:.2f}  (style axis {est.style}, z={est.z:.2f})')
    out = []
    for s in steps:
        tail = s['prompt'].split('] ', 1)[-1]
        out.append({**s, 'base_verb': base, 'base_weight': bw,
                    'prompt': f"[dance:{s['dance']:.2f}][{base}:{bw:.2f}] {tail}"})
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    """Run a command, printing it first, raising on failure."""
    print('  $', ' '.join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ffmpeg(*args, check=True):
    result = subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'warning'] + list(args),
        capture_output=True, text=True
    )
    if check and result.returncode != 0:
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return result


def find_repeat_mp4(editing_dir: Path, ext: str, repeat: int = 0) -> Path | None:
    """Find the mp4 for a given ext and repeat index inside editing/<ext>/animations/0/."""
    anim = editing_dir / ext / 'animations' / '0'
    if not anim.exists():
        return None
    cands = sorted(p for p in anim.glob('*.mp4')
                   if '_source_' not in p.name and '_ik.' not in p.name)
    return cands[repeat] if repeat < len(cands) else (cands[0] if cands else None)


def scale_video(src: Path, w: int, h: int, out: Path):
    ffmpeg(
        '-i', str(src),
        '-vf', f'scale={w}:{h}:force_original_aspect_ratio=decrease,'
               f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-an', str(out)
    )


def add_label(src: Path, label: str, out: Path):
    """Burn a text label — silently falls back to plain copy if drawtext fails."""
    safe = label.replace(':', '\\:').replace("'", "\\'")
    result = ffmpeg(
        '-i', str(src),
        '-vf', f"drawtext=text='{safe}':fontsize=18:fontcolor=white:"
               f"x=10:y=10:box=1:boxcolor=black@0.6:boxborderw=4",
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-an', str(out),
        check=False
    )
    if result.returncode != 0:
        ffmpeg('-i', str(src), '-c:v', 'libx264', '-preset', 'fast',
               '-crf', '22', '-an', str(out))


def concat_videos(clips: list[Path], out: Path):
    if not clips:
        raise RuntimeError('No clips to concatenate.')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(f"file '{str(p).replace(chr(92), '/')}'" for p in clips))
        list_path = f.name
    ffmpeg('-f', 'concat', '-safe', '0', '-i', list_path,
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', str(out))
    Path(list_path).unlink(missing_ok=True)


def make_grid(clips: list[Path], cols: int, pw: int, ph: int, out: Path):
    """Stack clips into a grid. clips is ordered left-to-right, top-to-bottom."""
    rows = (len(clips) + cols - 1) // cols
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Scale all panels
        scaled = []
        for i, c in enumerate(clips):
            s = tmp / f'p{i:02d}.mp4'
            scale_video(c, pw, ph, s)
            scaled.append(s)

        # Pad to full grid with blank if needed
        while len(scaled) < rows * cols:
            blank = tmp / f'blank{len(scaled)}.mp4'
            ffmpeg('-f', 'lavfi', '-i', f'color=c=black:s={pw}x{ph}:r=20:d=1',
                   '-c:v', 'libx264', '-preset', 'fast', str(blank))
            scaled.append(blank)

        # Build row videos with hstack
        row_vids = []
        for r in range(rows):
            row_clips = scaled[r * cols:(r + 1) * cols]
            inputs = []
            for c in row_clips:
                inputs += ['-i', str(c)]
            filter_complex = ''.join(f'[{i}:v]' for i in range(len(row_clips)))
            filter_complex += f'hstack=inputs={len(row_clips)}[v]'
            row_out = tmp / f'row{r}.mp4'
            ffmpeg(*inputs, '-filter_complex', filter_complex,
                   '-map', '[v]', '-c:v', 'libx264', '-preset', 'fast', str(row_out))
            row_vids.append(row_out)

        if len(row_vids) == 1:
            shutil.copy(str(row_vids[0]), str(out))
        else:
            inputs = []
            for rv in row_vids:
                inputs += ['-i', str(rv)]
            filter_complex = ''.join(f'[{i}:v]' for i in range(len(row_vids)))
            filter_complex += f'vstack=inputs={len(row_vids)}[v]'
            ffmpeg(*inputs, '-filter_complex', filter_complex,
                   '-map', '[v]', '-c:v', 'libx264', '-preset', 'fast', str(out))


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_mediapipe(video_path: Path, npz_out: Path, visualize: bool):
    """Run video_bridge/video_to_humanml3d.py to produce momask_input.npz."""
    bridge = Path(__file__).parent.parent / 'video_bridge' / 'video_to_humanml3d.py'
    cmd = [
        sys.executable, str(bridge),
        '--video', str(video_path),
        '--output', str(npz_out.with_suffix('')),  # bridge adds .npz itself
    ]
    if visualize:
        cmd.append('--visualize')
    run(cmd)


def stage_inference(npz_path: Path, editing_dir: Path, steps: list[dict] | None = None):
    """Run edit_t2m.py for all 10 dance intensity steps."""
    for s in (steps or STEPS):
        d   = f"{s['dance']:.2f}"
        ext = f'dance_mode_{d}'
        out_dir = editing_dir / ext
        if out_dir.exists():
            print(f'  Skipping {ext} (already exists)')
            continue
        cmd = [
            sys.executable, 'edit_t2m.py',
            '--gpu_id',            '0',
            '--ext',               ext,
            '--name',              MASK_NAME,
            '--dataset_name',      't2m',
            '--res_name',          RES_NAME,
            '--text_prompt',       s['prompt'],
            '--source_motion',     str(npz_path),
            '--mask_edit_section', '0.0,1.0',
            '--use_res_model',
            '--repeat_times',      str(REPEAT_TIMES),
        ]
        run(cmd)


def stage_collect(editing_dir: Path, out_dir: Path, panel_w: int, panel_h: int,
                  steps: list[dict] | None = None):
    """Copy first-repeat mp4s into out_dir, label them, build progression + quad."""
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_clips = []
    for s in (steps or STEPS):
        d   = f"{s['dance']:.2f}"
        ext = f'dance_mode_{d}'
        src = find_repeat_mp4(editing_dir, ext, repeat=0)
        if src is None:
            print(f'  WARNING: no output mp4 found for {ext} — skipping')
            continue

        # Copy raw
        raw_dst = out_dir / f'{ext}_repeat0.mp4'
        shutil.copy(str(src), str(raw_dst))

        # Labeled + scaled for quad/reel
        labeled = out_dir / f'{ext}_labeled.mp4'
        add_label(raw_dst, f'd={d}', labeled)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            scaled_path = Path(f.name)
        scale_video(labeled, panel_w, panel_h, scaled_path)
        labeled_clips.append(scaled_path)

        print(f'  Collected {ext}')

    if not labeled_clips:
        print('ERROR: no output clips found. Did inference complete?')
        return

    # Progression reel — all 10 in sequence
    reel = out_dir / 'progression.mp4'
    print(f'\nBuilding progression reel -> {reel}')
    concat_videos(labeled_clips, reel)

    # Quad grid — 2 columns × 5 rows (or 5×2, your choice)
    quad = out_dir / 'quad.mp4'
    print(f'Building quad grid -> {quad}')
    make_grid(labeled_clips, cols=2, pw=panel_w, ph=panel_h, out=quad)

    # Clean up temp scaled files
    for p in labeled_clips:
        p.unlink(missing_ok=True)

    print(f'\nOutputs saved to: {out_dir}')
    print(f'  progression.mp4  — all 10 steps in sequence')
    print(f'  quad.mp4         — 2x5 grid of all 10 steps')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INPUT_VIDEOS_DIR = Path('input_videos')


def pick_video(name: str | None) -> Path:
    """Resolve the input video from input_videos/.
    If --video is just a filename (no directory), look it up in input_videos/.
    If it is an absolute/relative path that exists, use it directly.
    If --video is omitted and exactly one video is in input_videos/, use that.
    """
    INPUT_VIDEOS_DIR.mkdir(exist_ok=True)

    if name is None:
        videos = sorted(INPUT_VIDEOS_DIR.glob('*.mp4')) + \
                 sorted(INPUT_VIDEOS_DIR.glob('*.mov')) + \
                 sorted(INPUT_VIDEOS_DIR.glob('*.avi'))
        if not videos:
            print(f'ERROR: no video files found in {INPUT_VIDEOS_DIR}/.')
            print('Drop an mp4/mov/avi into that folder and re-run, or use --video <name>.')
            sys.exit(1)
        if len(videos) > 1:
            print(f'Multiple videos found in {INPUT_VIDEOS_DIR}/. Specify one with --video:')
            for v in videos:
                print(f'  {v.name}')
            sys.exit(1)
        return videos[0].resolve()

    p = Path(name)
    if p.exists():
        return p.resolve()
    candidate = INPUT_VIDEOS_DIR / p
    if candidate.exists():
        return candidate.resolve()
    print(f'ERROR: video not found: {name}')
    print(f'  Looked in: . and {INPUT_VIDEOS_DIR}/')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Full pipeline: video → MediaPipe → MoMask dance spectrum → videos'
    )
    parser.add_argument('--video', default=None,
                        help='Video filename inside input_videos/, or a full path. '
                             'Omit if exactly one video is in input_videos/.')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Output folder. Default: outputs/<video_stem>/.')
    parser.add_argument('--editing_dir', type=Path, default=Path('editing'),
                        help='MoMask editing output directory. Default: editing/.')
    parser.add_argument('--panel_w', type=int, default=480,
                        help='Width of each panel in quad/reel. Default: 480.')
    parser.add_argument('--panel_h', type=int, default=360,
                        help='Height of each panel in quad/reel. Default: 360.')
    parser.add_argument('--skip_mediapipe', action='store_true',
                        help='Skip MediaPipe step (use existing momask_input.npz).')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip edit_t2m.py runs (use existing editing/ outputs).')
    parser.add_argument('--visualize', action='store_true',
                        help='Save MediaPipe skeleton overlay video for sanity checks.')
    parser.add_argument('--auto-z', dest='auto_z', action='store_true',
                        help='Stage 2.2: estimate the base verb and weight from the recording '
                             'instead of the fixed [walk:0.91]. The dance axis still sweeps.')
    parser.add_argument('--text', default=None,
                        help='Custom text tail appended after the spectrum tags in every prompt. '
                             'E.g. --text "A person performs a hip-hop routine." '
                             'If omitted, the default per-step descriptions are used.')
    args = parser.parse_args()

    steps = build_steps(args.text)

    video_path = pick_video(args.video)
    out_dir = args.out_dir or Path('outputs') / video_path.stem
    npz_path = Path('momask_input.npz')

    print(f'\n=== Stage 1: MediaPipe extraction ===')
    if args.skip_mediapipe:
        print(f'  Skipping (using existing {npz_path})')
    else:
        stage_mediapipe(video_path, npz_path, args.visualize)

    if args.auto_z:
        print('  Stage 2.2: estimating base verb from the recording (--auto-z)')
        steps = apply_estimated_base(steps, npz_path)

    print(f'\n=== Stage 2: MoMask inference (10 intensity steps) ===')
    if args.text:
        print(f'  Using custom text: "{args.text}"')
    if args.skip_inference:
        print('  Skipping (using existing editing/ outputs)')
    else:
        stage_inference(npz_path, args.editing_dir, steps)

    print(f'\n=== Stage 3: Collecting outputs and building videos ===')
    stage_collect(args.editing_dir, out_dir, args.panel_w, args.panel_h, steps)


if __name__ == '__main__':
    main()
