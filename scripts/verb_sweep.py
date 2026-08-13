"""
verb_sweep.py — Multi-verb spectrum sweep for MoMask perceptual study.

For each source video (named <verb>_<anything>.mp4), sweeps one or all of the
7 other spectrum verbs across N quantile steps, building prompts from one of
4 strategies and calling edit_t2m.py once per (source_video, secondary_verb,
quantile) combination.

Usage:
    python scripts/verb_sweep.py \\
        --videos walk_001.mp4 dance_002.mp4 \\
        --secondary_verb dance \\
        --num_quantiles 10 \\
        --quantile_prompt_strategy 1 \\
        --source_verb_mode fixed \\
        --videos_per_section 1 \\
        --gpu_id 0

    # 5-quantile run selecting odd scales from the 10-scale prompt table:
    python scripts/verb_sweep.py \\
        --videos walk_001.mp4 \\
        --secondary_verb all \\
        --num_quantiles 5 \\
        --text_prompts 1,3,5,7,9 \\
        --quantile_prompt_strategy 1 \\
        --source_verb_mode sync \\
        --videos_per_section 2

    # Skip MediaPipe for multiple videos (npz files named <video_stem>.npz):
    python scripts/verb_sweep.py \\
        --videos walk_001.mp4 dance_002.mp4 \\
        --skip_mediapipe walk_001.npz dance_002.npz \\
        --secondary_verb run \\
        --num_quantiles 10 \\
        --quantile_prompt_strategy 3

Source verb:  extracted from the video filename (first word before the first '_').
              e.g.  walk_trial1.mp4  →  walk

Prompt strategies:
    1  [source:mode][secondary:mode] <text from prompts JSON>
    2  <text from prompts JSON>  (tags omitted)
    3  [source:mode][secondary:mode]  (no text, mode values)
    4  [source:q/N][secondary:q/N]    (no text, rounded quantile fraction)

Source-verb tag modes:
    fixed   source verb tag is always its Q10 (highest) mode value
    sync    source verb tag advances through quantiles in sync with secondary verb

Prompts JSON (verb_prompts.json):
    Auto-generated on first run at outputs/verb_prompts.json and never
    overwritten unless --regenerate_prompts is passed. Edit the file freely
    to customise the text sentences used in strategies 1 and 2.
    Structure:  { "walk_dance": ["sentence scale 1", ..., "sentence scale 10"] }
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Spectrum constants (computed from spectrum_scores.json, bin=0.05, Q=10)
# ---------------------------------------------------------------------------

VERBS = ['walk', 'dance', 'run', 'jump', 'spin', 'kick', 'wave', 'stand']

# Mode of the most frequent 0.05-bin within each 10th-percentile quantile.
# Index 0 = lowest decile, index 9 = highest decile.
# Modes computed from synthetic training captions in HumanML3D/texts/*_spec_*.txt
# Fixed-width 10% score bins (0-0.1, 0.1-0.2, ...) with bin=0.01 to find the mode.
# Matches the per-verb histogram charts in the report.
VERB_QUANTILE_MODES: dict[str, list[float]] = {
    'walk':  [0.08, 0.19, 0.25, 0.31, 0.49, 0.54, 0.60, 0.72, 0.85, 0.91],
    'dance': [0.09, 0.11, 0.28, 0.39, 0.40, 0.51, 0.63, 0.74, 0.80, 1.00],
    'run':   [0.07, 0.13, 0.29, 0.38, 0.41, 0.50, 0.61, 0.75, 0.85, 0.90],
    'jump':  [0.07, 0.15, 0.26, 0.39, 0.47, 0.55, 0.63, 0.70, 0.80, 0.97],
    'spin':  [0.07, 0.11, 0.26, 0.32, 0.44, 0.51, 0.63, 0.75, 0.87, 0.93],
    'kick':  [0.07, 0.14, 0.29, 0.38, 0.49, 0.57, 0.69, 0.71, 0.80, 0.90],
    'wave':  [0.05, 0.15, 0.29, 0.39, 0.41, 0.58, 0.60, 0.70, 0.80, 0.95],
    'stand': [0.05, 0.19, 0.29, 0.34, 0.40, 0.58, 0.69, 0.78, 0.89, 0.92],
}

MASK_NAME = 'MaskTransformer'
RES_NAME  = 'ResTransformer'
PROMPTS_PATH = Path('outputs') / 'verb_prompts.json'


# ---------------------------------------------------------------------------
# Prompt JSON generation
# ---------------------------------------------------------------------------

def _auto_sentence(sv: str, dv: str, scale: int) -> str:
    """Graduated sentence for (source_verb, secondary_verb) at scale 1-10."""
    if scale <= 2:
        return f"A person {sv}s."
    elif scale <= 4:
        return f"A person {sv}s with subtle {dv} elements."
    elif scale <= 6:
        return f"A person {sv}s and {dv}s."
    elif scale <= 8:
        return f"A person {dv}s with a {sv}ing base."
    else:
        return f"A person {dv}s expressively."


def load_or_generate_prompts(regenerate: bool = False) -> dict[str, list[str]]:
    """Load verb_prompts.json, generating it first if missing or if regenerate=True."""
    if PROMPTS_PATH.exists() and not regenerate:
        with open(PROMPTS_PATH) as f:
            data = json.load(f)
        print(f"Loaded prompts from {PROMPTS_PATH}")
        return data

    print(f"Generating prompts JSON ({len(VERBS) * (len(VERBS) - 1)} pairs x 10 scales) ...")
    prompts: dict[str, list[str]] = {}
    for sv in VERBS:
        for dv in VERBS:
            if sv == dv:
                continue
            key = f"{sv}_{dv}"
            prompts[key] = [_auto_sentence(sv, dv, s) for s in range(1, 11)]

    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_PATH, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved to {PROMPTS_PATH}  (edit freely to customise text sentences)")
    return prompts


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _mode(verb: str, slot: int) -> float:
    """Return the pre-computed mode for verb at 0-based 10-quantile slot."""
    return VERB_QUANTILE_MODES[verb][slot]


def build_prompt(
    strategy: int,
    sv: str,
    dv: str,
    q_idx: int,
    slot: int,          # 0-based index into the 10-scale table
    source_verb_mode: str,
    prompts: dict[str, list[str]],
) -> str:
    """
    Build the full text_prompt string for one edit_t2m.py call.

    Args:
        strategy:         1-4
        sv:               source verb  (from filename)
        dv:               secondary (target) verb
        q_idx:            0-based quantile index within the current run
        slot:             0-based index into the 10-scale mode/prompt table
        source_verb_mode: 'fixed' or 'sync'
        prompts:          loaded prompts JSON dict
    """
    sv_slot = 9 if source_verb_mode == 'fixed' else slot  # Q10 = index 9
    sv_val  = _mode(sv, sv_slot)
    dv_val  = _mode(dv, slot)

    if strategy == 4:
        # Rounded quantile fraction: (slot+1)/10 for both verbs
        sv_frac = round((sv_slot + 1) / 10, 2)
        dv_frac = round((slot + 1) / 10, 2)
        return f"[{sv}:{sv_frac:.2f}][{dv}:{dv_frac:.2f}]"

    key = f"{sv}_{dv}"
    text = prompts.get(key, [""] * 10)[slot]

    tags = f"[{sv}:{sv_val:.2f}][{dv}:{dv_val:.2f}]"

    if strategy == 1:
        return f"{tags} {text}"
    elif strategy == 2:
        return text
    elif strategy == 3:
        return tags

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list, **kwargs):
    """Print and run a command, raising on failure."""
    print('  $', ' '.join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def run_batch(jobs: list[dict], args) -> None:
    """Write jobs to a temp JSON and call edit_t2m_batch.py once (loads models once)."""
    import tempfile
    jobs_file = Path(tempfile.mktemp(suffix='_sweep_jobs.json'))
    jobs_file.write_text(json.dumps(jobs, indent=2))
    print(f"\n  Submitting {len(jobs)} job(s) to edit_t2m_batch.py (models load once) ...")
    batch_script = Path(__file__).parent.parent / 'edit_t2m_batch.py'
    cmd = [
        sys.executable, str(batch_script),
        '--jobs_file',      str(jobs_file),
        '--gpu_id',         str(args.gpu_id),
        '--name',           MASK_NAME,
        '--res_name',       RES_NAME,
        '--dataset_name',   't2m',
        '--mask_edit_section', '0.0,1.0',
        '--render_workers', '1' if args.no_parallel else str(args.render_workers),
    ] + (['--skip_ik'] if args.skip_ik else [])
    try:
        run_cmd(cmd)
    finally:
        jobs_file.unlink(missing_ok=True)


def extract_verb(video_path: Path) -> str:
    """Extract the source verb from the video filename (first word before '_')."""
    stem = video_path.stem
    if '_' not in stem:
        raise ValueError(
            f"Video filename '{video_path.name}' must start with a verb followed by "
            f"'_' (e.g. walk_001.mp4). Got: '{stem}'"
        )
    verb = stem.split('_')[0].lower()
    if verb not in VERBS:
        raise ValueError(
            f"Verb '{verb}' extracted from '{video_path.name}' is not one of the "
            f"known verbs: {VERBS}"
        )
    return verb


def resolve_npz(video_path: Path, skip_map: dict[Path, Path]) -> Path | None:
    """Return the pre-computed .npz for this video if provided, else None."""
    return skip_map.get(video_path.resolve())


def ensure_npz(video_path: Path, npz_path: Path | None, mp_python: str | None = None) -> Path:
    """Return an existing npz or run MediaPipe to produce one."""
    if npz_path is not None:
        if not npz_path.exists():
            raise FileNotFoundError(f"--skip_mediapipe file not found: {npz_path}")
        print(f"  Using existing npz: {npz_path}")
        return npz_path

    # Run MediaPipe bridge using the designated Python (may differ from sys.executable)
    bridge = Path(__file__).parent.parent / 'video_bridge' / 'video_to_humanml3d.py'
    python = mp_python or sys.executable
    if mp_python:
        print(f"  Using MediaPipe Python: {mp_python}")
    out_stem = Path('outputs') / video_path.stem / 'mediapipe_out'
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([python, str(bridge), '--video', str(video_path), '--output', str(out_stem)])
    npz = out_stem.with_suffix('.npz')
    if not npz.exists():
        raise RuntimeError(f"MediaPipe bridge did not produce expected output: {npz}")
    return npz


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def collect_jobs(
    video_path: Path,
    sv: str,
    secondary_verbs: list[str],
    npz_path: Path,
    slots: list[int],
    strategies: list[int],
    source_verb_mode: str,
    videos_per_section: int,
    editing_dir: Path,
    prompts: dict[str, list[str]],
) -> list[dict]:
    """Build the list of pending jobs, skipping already-completed ones."""
    jobs = []
    for dv in secondary_verbs:
        for strategy in strategies:
            for q_idx, slot in enumerate(slots):
                prompt  = build_prompt(strategy, sv, dv, q_idx, slot, source_verb_mode, prompts)
                folder  = f"{sv}_to_{dv}_s{strategy}_{source_verb_mode}_q{q_idx + 1:02d}of{len(slots)}"
                ext     = f"{video_path.stem}/{folder}"
                out_dir = editing_dir / folder

                if out_dir.exists():
                    mp4s = list(out_dir.rglob('*.mp4'))
                    if mp4s:
                        print(f"  Skipping {folder} (already has {len(mp4s)} mp4s)")
                        continue
                    else:
                        print(f"  Queuing  {folder} (folder exists but no mp4s)")
                else:
                    print(f"  Queuing  {folder}")

                jobs.append({
                    'ext':           ext,
                    'text_prompt':   prompt,
                    'source_motion': str(npz_path),
                    'repeat_times':  videos_per_section,
                })
    return jobs


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Multi-verb spectrum sweep for MoMask perceptual study.'
    )

    # Input videos
    parser.add_argument(
        '--videos', nargs='+', default=None, type=Path,
        help='1-8 source videos. Each must be named <verb>_<anything>.mp4. '
             'The leading verb is extracted automatically.'
    )
    parser.add_argument(
        '--videos_dir', type=Path, default=None,
        help='Folder of videos to process. All .mp4/.mov/.avi files whose names '
             'start with a known verb are included. Use instead of --videos.'
    )
    parser.add_argument(
        '--skip_mediapipe', nargs='*', type=Path, default=None,
        help='Pre-computed .npz files, one per --videos entry in the same order. '
             'If omitted, MediaPipe is run on each video. '
             'If fewer .npz files are given than videos, the remainder run MediaPipe.'
    )

    # Verb selection
    parser.add_argument(
        '--secondary_verb', default='all',
        help="Target verb to sweep, or 'all' to sweep all 7 verbs other than the "
             f"source verb. One of: {VERBS + ['all']}."
    )

    # Quantile settings
    parser.add_argument(
        '--num_quantiles', type=int, default=10,
        help='Number of quantile steps to generate (1-10). Default: 10.'
    )
    parser.add_argument(
        '--text_prompts', type=str, default=None,
        help='Comma-separated 1-based scale indices from the 10-scale prompts table '
             'to use when --num_quantiles < 10. '
             'E.g. --text_prompts=1,3,5,7,9 for 5 quantiles. '
             'Must have exactly --num_quantiles entries. '
             'Defaults to evenly spaced indices across 1-10.'
    )

    # Prompt strategy
    parser.add_argument(
        '--quantile_prompt_strategy', default='all',
        help="Which prompt strategy to use: 1, 2, 3, 4, or 'all'. "
             "1=[tags+text]  2=[text only]  3=[tags only, modes]  "
             "4=[tags only, rounded fractions]. Default: all."
    )

    # Source verb tag mode
    parser.add_argument(
        '--source_verb_mode', default='fixed', choices=['fixed', 'sync'],
        help="How the source-verb tag value changes across quantiles. "
             "'fixed' always uses the Q10 (highest) mode. "
             "'sync' advances in step with the secondary verb. Default: fixed."
    )

    # Generation settings
    parser.add_argument(
        '--videos_per_section', type=int, default=1,
        help='Number of repeat generations per quantile step (--repeat_times in '
             'edit_t2m.py). Default: 1.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0,
        help='GPU index passed to edit_t2m.py. Default: 0.'
    )
    parser.add_argument(
        '--editing_dir', type=Path, default=None,
        help='Base directory for outputs. Default: editing/<video_stem>/.'
    )
    parser.add_argument(
        '--regenerate_prompts', action='store_true',
        help='Re-generate outputs/verb_prompts.json even if it already exists.'
    )
    parser.add_argument(
        '--skip_ik', action='store_true',
        help='Skip IK BVH conversion and IK mp4 (saves ~30s per job).'
    )
    parser.add_argument(
        '--render_workers', type=int, default=4,
        help='CPU processes for parallel mp4 rendering. Default: 4.'
    )
    parser.add_argument(
        '--no_parallel', action='store_true',
        help='Disable parallel rendering (sets render_workers=1). Use if you get CUDA/memory errors.'
    )
    parser.add_argument(
        '--mediapipe_python', default=None,
        help='Python executable for the MediaPipe stage (needs NumPy>=2 + mediapipe). '
             'The main env stays on NumPy<2 for torch. '
             'E.g. --mediapipe_python C:\\Users\\maura\\anaconda3\\envs\\momask-mp\\python.exe '
             'Can also be set via the MOMASK_MP_PYTHON environment variable.'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def resolve_slots(num_quantiles: int, text_prompts_arg: str | None) -> list[int]:
    """Return 0-based slot indices into the 10-scale table."""
    if num_quantiles < 1 or num_quantiles > 10:
        raise ValueError('--num_quantiles must be between 1 and 10.')

    if text_prompts_arg is not None:
        slots_1based = [int(x.strip()) for x in text_prompts_arg.split(',')]
        if len(slots_1based) != num_quantiles:
            raise ValueError(
                f'--text_prompts has {len(slots_1based)} entries but '
                f'--num_quantiles={num_quantiles}. They must match.'
            )
        for s in slots_1based:
            if not 1 <= s <= 10:
                raise ValueError(f'--text_prompts entries must be 1-10, got {s}.')
        return [s - 1 for s in slots_1based]

    # Default: evenly spaced across 0-9
    if num_quantiles == 10:
        return list(range(10))
    step = 9 / (num_quantiles - 1) if num_quantiles > 1 else 0
    return [round(i * step) for i in range(num_quantiles)]


def resolve_strategies(arg: str) -> list[int]:
    if arg == 'all':
        return [1, 2, 3, 4]
    parts = [x.strip() for x in arg.split(',')]
    strategies = []
    for p in parts:
        s = int(p)
        if s not in (1, 2, 3, 4):
            raise ValueError(f'--quantile_prompt_strategy must be 1-4 or all, got {s}.')
        strategies.append(s)
    return strategies


def resolve_secondary_verbs(arg: str, source_verb: str) -> list[str]:
    if arg == 'all':
        return [v for v in VERBS if v != source_verb]
    if arg not in VERBS:
        raise ValueError(f'--secondary_verb must be one of {VERBS} or "all", got "{arg}".')
    if arg == source_verb:
        return []   # caller skips this video
    return [arg]


def build_skip_map(videos: list[Path], npz_args: list[Path] | None) -> dict[Path, Path]:
    """Map resolved video path → resolved npz path for skip_mediapipe entries."""
    if not npz_args:
        return {}
    skip_map: dict[Path, Path] = {}
    for i, video in enumerate(videos):
        if i < len(npz_args):
            skip_map[video.resolve()] = npz_args[i].resolve()
    return skip_map


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import os
    import time
    args = parse_args()
    mp_python = args.mediapipe_python or os.environ.get('MOMASK_MP_PYTHON')
    t_start = time.time()
    _fmt_time = lambda s: f"{int(s//3600):02d}h {int((s%3600)//60):02d}m {int(s%60):02d}s"

    # Resolve video list from --videos or --videos_dir
    if args.videos_dir is not None:
        if not args.videos_dir.is_dir():
            print(f'ERROR: --videos_dir "{args.videos_dir}" is not a directory.')
            sys.exit(1)
        candidates = sorted(
            p for p in args.videos_dir.iterdir()
            if p.suffix.lower() in ('.mp4', '.mov', '.avi')
            and p.stem.split('_')[0].lower() in VERBS
        )
        if not candidates:
            print(f'ERROR: no verb-named videos found in {args.videos_dir}/')
            sys.exit(1)
        args.videos = candidates
        print(f'  Found {len(args.videos)} video(s) in {args.videos_dir}/')
    elif args.videos is None:
        print('ERROR: provide --videos or --videos_dir.')
        sys.exit(1)

    # Validate and load
    slots      = resolve_slots(args.num_quantiles, args.text_prompts)
    strategies = resolve_strategies(args.quantile_prompt_strategy)
    skip_map   = build_skip_map(args.videos, args.skip_mediapipe)
    prompts    = load_or_generate_prompts(args.regenerate_prompts)

    from datetime import datetime
    print(f"\n=== Verb Sweep ===  (started {datetime.now().strftime('%H:%M:%S')})")
    print(f"  Videos:          {[v.name for v in args.videos]}")
    print(f"  Secondary verb:  {args.secondary_verb}")
    print(f"  Quantiles:       {args.num_quantiles}  (slots: {[s+1 for s in slots]})")
    print(f"  Strategies:      {strategies}")
    print(f"  Source verb tag: {args.source_verb_mode}")
    print(f"  Per section:     {args.videos_per_section} generation(s)")

    all_jobs: list[dict] = []

    for video_path in args.videos:
        if not video_path.exists():
            # Also look in input_videos/
            candidate = Path('input_videos') / video_path
            if candidate.exists():
                video_path = candidate
            else:
                print(f"\nERROR: video not found: {video_path}")
                sys.exit(1)

        sv = extract_verb(video_path)
        secondary_verbs = resolve_secondary_verbs(args.secondary_verb, sv)
        if not secondary_verbs:
            print(f"\n--- {video_path.name}  (source verb: {sv}) ---")
            print(f"    Skipping — source verb and secondary verb are both '{sv}'.")
            continue
        editing_dir = (args.editing_dir or Path('editing')) / video_path.stem

        print(f"\n--- {video_path.name}  (source verb: {sv}) ---")
        print(f"    Secondary verbs: {secondary_verbs}")
        print(f"    Output dir:      {editing_dir}/")

        npz_path = ensure_npz(video_path, resolve_npz(video_path, skip_map), mp_python)

        all_jobs += collect_jobs(
            video_path       = video_path,
            sv               = sv,
            secondary_verbs  = secondary_verbs,
            npz_path         = npz_path,
            slots            = slots,
            strategies       = strategies,
            source_verb_mode = args.source_verb_mode,
            videos_per_section = args.videos_per_section,
            editing_dir      = editing_dir,
            prompts          = prompts,
        )

    if all_jobs:
        print(f"\n  {len(all_jobs)} job(s) to run (models load once)")
        run_batch(all_jobs, args)
    else:
        print("\n  All jobs already complete — nothing to run.")

    elapsed = time.time() - t_start
    print(f"\n=== Done. Outputs in editing/<video_stem>/ ===")
    print(f"    Total time: {_fmt_time(elapsed)}")


if __name__ == '__main__':
    main()
