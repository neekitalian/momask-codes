"""
run_zone_blend.py — CLI entry point for the Phase 2 zone-aware blending pipeline.

Loads MoMask models once, then runs the ZoneBlendPipeline for each job in a
JSON jobs file.

Usage
-----
    python scripts/run_zone_blend.py \\
        --jobs_file sweep_jobs.json \\
        --alpha 0.5 \\
        --zone_mode standard \\
        --gpu_id 0

Jobs file format — two supported variants:

  Video input (bridge runs automatically if .npz not yet cached):
    [
      {
        "ext":          "walk_001/walk_to_dance_q01",
        "text_prompt":  "[walk:0.91] A person walks.",
        "source_video": "video_bridge/inputs/walk_001.mp4",
        "repeat_times": 1
      },
      ...
    ]

  Pre-extracted .npz input (bridge skipped):
    [
      {
        "ext":           "walk_001/walk_to_dance_q01",
        "text_prompt":   "[walk:0.91] A person walks.",
        "source_motion": "outputs/walk_001/mediapipe_out.npz",
        "repeat_times":  1
      },
      ...
    ]

When "source_video" is provided, the bridge writes a .npz alongside the video
(same stem, .npz extension) on first run and reuses it on subsequent runs.
If both "source_video" and "source_motion" are given, "source_motion" wins.

The source_motion .npz must contain either:
  - 'motion'  : (T, 263) HumanML3D motion vector  (preferred)
  - 'joints'  : (T, 22, 3) raw joint positions    (fallback)
"""

import argparse
import json
import os
import sys
import time
from os.path import join as pjoin
from pathlib import Path

# Ensure repo root is on sys.path when script is run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from gen_t2m import load_res_model, load_trans_model, load_vq_model
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor

from semantic_spectrum.pipeline import ZoneBlendPipeline


def run_video_bridge(video_path: str) -> str:
    """
    Run video_bridge/video_to_humanml3d.py on *video_path* if the cached .npz
    does not already exist.  Returns the path to the .npz file.
    """
    video_path = Path(video_path).expanduser().resolve()
    npz_path   = video_path.with_suffix(".npz")

    if npz_path.exists():
        print(f"  [bridge] reusing cached extraction: {npz_path}")
        return str(npz_path)

    bridge_script = Path(__file__).resolve().parent.parent / "video_bridge" / "video_to_humanml3d.py"
    if not bridge_script.exists():
        raise FileNotFoundError(f"video_bridge script not found at {bridge_script}")

    print(f"  [bridge] extracting pose from {video_path.name} ...")
    cmd = [
        sys.executable, str(bridge_script),
        "--video",  str(video_path),
        "--output", str(npz_path),
    ]
    import subprocess
    result = subprocess.run(cmd, check=True)
    if not npz_path.exists():
        raise RuntimeError(f"Bridge completed but output not found: {npz_path}")
    print(f"  [bridge] saved to {npz_path}")
    return str(npz_path)


def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 2 zone-blend pipeline: feature-space blending of '
                    'original and MoMask-generated motion.'
    )
    p.add_argument('--jobs_file',   required=True,
                   help='Path to JSON jobs file.')
    p.add_argument('--alpha',       type=float, default=0.5,
                   help='Blend strength in [0, 1]. Default: 0.5')
    p.add_argument('--zone_mode',   default='standard',
                   choices=['standard', 'side_specific'],
                   help='Zone configuration mode. Default: standard')
    p.add_argument('--gpu_id',      type=int, default=0)
    p.add_argument('--name',        default='MaskTransformer')
    p.add_argument('--res_name',    default='ResTransformer')
    p.add_argument('--dataset_name', default='t2m')
    p.add_argument('--checkpoints_dir', default='./checkpoints')
    p.add_argument('--time_steps', type=int,   default=18)
    p.add_argument('--cond_scale', type=float, default=4.0)
    p.add_argument('--temperature',type=float, default=1.0)
    p.add_argument('--topkr',      type=float, default=0.9)
    p.add_argument('--tour',       action='store_true', default=False,
                   help='After all jobs complete, render a spectrum tour video '
                        'cycling through the blended outputs of every job.')
    p.add_argument('--ablate',     action='store_true', default=False,
                   help='Run once per feature dimension and save a named output '
                        'for each. Produces 7 extra output variants alongside '
                        'the normal blended output.')
    p.add_argument('--ik',         action='store_true', default=False,
                   help='Run foot IK pass on outputs. Off by default — '
                        'foot IK requires a calibrated ground plane and will '
                        'collapse the skeleton on video-bridge input.')
    p.add_argument('--out_dir',    default='./blend_outputs',
                   help='Root output directory. Default: ./blend_outputs')
    return p.parse_args()


def load_source(path: str) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Load source motion from .npz.

    Returns
    -------
    motion_vec : (T, 263) or None if not present
    joints     : (T, 22, 3)
    """
    data = np.load(path, allow_pickle=True)
    motion_vec = None
    if 'motion' in data:
        motion_vec = data['motion'].astype(np.float32)
    if 'joints' in data:
        joints = data['joints'].astype(np.float32)
    elif motion_vec is not None:
        # derive joints from motion vec
        joints = recover_from_ric(
            torch.from_numpy(motion_vec).float(), 22
        ).numpy()
    else:
        raise ValueError(f"Source file {path} must contain 'motion' or 'joints'.")
    return motion_vec, joints


def main():
    args = parse_args()
    fixseed(10107)

    with open(args.jobs_file, encoding='utf-8-sig') as f:
        jobs = json.load(f)
    if not jobs:
        print("No jobs to run.")
        return

    device    = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    dim_pose  = 251 if args.dataset_name == 'kit' else 263

    # ---- Load models ----
    root_dir       = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt      = get_opt(model_opt_path, device=device)

    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name,
                        model_opt.vq_name, 'opt.txt')
    vq_opt           = get_opt(vq_opt_path, device=device)
    vq_opt.dim_pose  = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens     = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim       = vq_opt.code_dim

    res_opt_path = pjoin(args.checkpoints_dir, args.dataset_name,
                         args.res_name, 'opt.txt')

    class _Opt:
        pass
    opt = _Opt()
    opt.device          = device
    opt.gpu_id          = args.gpu_id
    opt.name            = args.name
    opt.res_name        = args.res_name
    opt.dataset_name    = args.dataset_name
    opt.checkpoints_dir = args.checkpoints_dir
    opt.time_steps      = args.time_steps
    opt.cond_scale      = args.cond_scale
    opt.temperature     = args.temperature
    opt.topkr           = args.topkr
    opt.gumbel_sample   = False
    opt.force_mask      = False

    res_opt   = get_opt(res_opt_path, device=device)
    res_model = load_res_model(res_opt, vq_opt, opt)
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    for m in (vq_model, t2m_transformer, res_model):
        m.eval()
        m.to(device)

    mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name,
                         model_opt.vq_name, 'meta', 'mean.npy'))
    std  = np.load(pjoin(args.checkpoints_dir, args.dataset_name,
                         model_opt.vq_name, 'meta', 'std.npy'))

    converter = Joint2BVHConvertor()

    # ---- Build pipeline ----
    pipeline = ZoneBlendPipeline(
        vq_model=vq_model,
        mask_transformer=t2m_transformer,
        res_model=res_model,
        vq_opt=vq_opt,
        mean=mean,
        std=std,
        alpha=args.alpha,
        zone_mode=args.zone_mode,
        device=device,
        time_steps=args.time_steps,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topkr=args.topkr,
    )

    print(f"\nModels loaded. Running {len(jobs)} job(s) "
          f"[alpha={args.alpha}, zone_mode={args.zone_mode}] ...\n")
    t0 = time.time()

    for i, job in enumerate(jobs):
        torch.cuda.empty_cache()
        ext         = job['ext']
        text_prompt = job['text_prompt']

        print(f"\n[{i+1}/{len(jobs)}] {ext}")
        print(f"  prompt : {text_prompt}")

        result_dir    = pjoin(args.out_dir, ext)
        joints_dir    = pjoin(result_dir, 'joints')
        animation_dir = pjoin(result_dir, 'animations')
        os.makedirs(joints_dir,    exist_ok=True)
        os.makedirs(animation_dir, exist_ok=True)

        t_job = time.time()

        # Resolve source: video → bridge → .npz, or use pre-extracted .npz directly
        src_path = job.get('source_motion')
        if not src_path:
            src_video = job.get('source_video')
            if not src_video:
                raise ValueError(
                    f"Job '{ext}' must have 'source_motion' or 'source_video'."
                )
            src_path = run_video_bridge(src_video)

        motion_vec, orig_joints = load_source(src_path)

        if motion_vec is not None:
            source_joints, momask_joints, output_joints, M_output = pipeline.run_from_motion_vec(
                motion_vec, orig_joints, text_prompt, return_intermediates=True)
        else:
            source_joints, momask_joints, output_joints, M_output = pipeline.run(
                orig_joints, text_prompt, return_intermediates=True)

        T = len(output_joints)

        def save_variant(tag: str, joints: np.ndarray) -> None:
            np.save(pjoin(joints_dir, f'{tag}.npy'), joints)
            bvh = pjoin(animation_dir, f'{tag}_len{T}.bvh')
            _, sj = converter.convert(joints, filename=bvh, iterations=100, foot_ik=False)
            plot_3d_motion(pjoin(animation_dir, f'{tag}_len{T}.mp4'),
                           t2m_kinematic_chain, sj, title=tag, fps=20)
            if args.ik:
                bvh_ik = pjoin(animation_dir, f'{tag}_len{T}_ik.bvh')
                _, ij = converter.convert(joints, filename=bvh_ik, iterations=100, foot_ik=True)
                plot_3d_motion(pjoin(animation_dir, f'{tag}_len{T}_ik.mp4'),
                               t2m_kinematic_chain, ij, title=f'{tag} (IK)', fps=20)

        save_variant('source',  source_joints)
        save_variant('momask',  momask_joints)
        save_variant('blended', output_joints)

        # DTW identity metric: blended vs source
        dtw_dir = pjoin(result_dir, 'dtw')
        try:
            from scripts.similarity_align import run_alignment
            print(f"  [dtw] measuring identity retention ...")
            summary = run_alignment(
                original=pjoin(joints_dir, 'source.npy'),
                generated=pjoin(joints_dir, 'blended.npy'),
                output_dir=dtw_dir,
            )
            print(f"  [dtw] avg_per_pair={summary['avg_per_pair']:.3f}  "
                  f"(lower = more identity preserved)")
        except Exception as e:
            print(f"  [dtw] skipped: {e}")

        # Ablation: one output per feature dimension
        if args.ablate:
            from semantic_spectrum.blend import (
                FEATURE_NAMES, _IDX_FREQ, _IDX_FREQ_MAG,
            )
            # dom_freq and freq_mag are coupled in the blender (both must be active
            # to produce oscillation), so they are always ablated as a pair.
            _ABLATE_SETS = [
                ({i}, FEATURE_NAMES[i])
                for i in range(len(FEATURE_NAMES))
                if i not in (_IDX_FREQ, _IDX_FREQ_MAG)
            ] + [({_IDX_FREQ, _IDX_FREQ_MAG}, "dom_freq+freq_mag")]

            print(f"  [ablate] running {len(_ABLATE_SETS)} passes ...")
            for active_set, feat_name in _ABLATE_SETS:
                ablation_joints = pipeline.blender.reconstruct(
                    M_output, source_joints, active_features=active_set
                )
                idx_str = "+".join(f"{i:02d}" for i in sorted(active_set))
                tag = f'ablate_{idx_str}_{feat_name}'
                save_variant(tag, ablation_joints)
                print(f"    {feat_name} done")

        print(f"  done in {time.time() - t_job:.1f}s → {result_dir}")

    total = time.time() - t0
    print(f"\nAll {len(jobs)} jobs done in {total:.1f}s ({total/len(jobs):.1f}s/job)")

    if args.tour:
        import subprocess
        blended_npys = []
        for job in jobs:
            ext = job['ext']
            p   = pjoin(args.out_dir, ext, 'joints', 'blended.npy')
            if os.path.exists(p):
                blended_npys.append(p)
        if blended_npys:
            tour_out = pjoin(args.out_dir, 'spectrum_tour.mp4')
            labels   = ','.join(job['ext'].split('/')[-1] for job in jobs
                                if os.path.exists(pjoin(args.out_dir, job['ext'], 'joints', 'blended.npy')))
            print(f"\n[tour] rendering spectrum tour → {tour_out}")
            subprocess.run([
                sys.executable, 'scripts/render_spectrum_tour.py',
                '--files',  ','.join(blended_npys),
                '--labels', labels,
                '--out',    tour_out,
            ], check=True)
        else:
            print("[tour] no blended outputs found, skipping tour.")


if __name__ == '__main__':
    main()
