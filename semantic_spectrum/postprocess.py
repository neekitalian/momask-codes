"""
postprocess.py — modular biomechanical post-processing for blended motion sequences.

Each step is a pure function (T, 22, 3) → (T, 22, 3).  Steps are independent
and can be enabled/disabled individually.  apply_pipeline() runs them in order.

Recommended order (each step feeds into the next):
  1. smooth_temporal     — remove jitter before geometric corrections
  2. clamp_velocity      — cap implausibly fast joint motion
  3. clamp_acceleration  — cap implausibly snappy direction changes
  4. preserve_bone_lengths — re-project joints to fix limb stretching from blending
  5. enforce_floor       — prevent feet sinking below the ground plane

HumanML3D / SMPL-H joint layout (22 joints, Y-up):
  0 Pelvis  1 L_Hip   2 R_Hip   3 Spine1  4 L_Knee   5 R_Knee
  6 Spine2  7 L_Ankle 8 R_Ankle 9 Spine3 10 L_Toe   11 R_Toe
 12 Neck   13 L_Collar 14 R_Collar 15 Head
 16 L_Shoulder 17 R_Shoulder 18 L_Elbow 19 R_Elbow
 20 L_Wrist   21 R_Wrist
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Parent index for each joint (-1 = root).  Topological order: parent always < child.
SMPL_PARENTS = [
    -1,   # 0  Pelvis
     0,   # 1  L_Hip
     0,   # 2  R_Hip
     0,   # 3  Spine1
     1,   # 4  L_Knee
     2,   # 5  R_Knee
     3,   # 6  Spine2
     4,   # 7  L_Ankle
     5,   # 8  R_Ankle
     6,   # 9  Spine3
     7,   # 10 L_Toe
     8,   # 11 R_Toe
     9,   # 12 Neck
     9,   # 13 L_Collar
     9,   # 14 R_Collar
    12,   # 15 Head
    13,   # 16 L_Shoulder
    14,   # 17 R_Shoulder
    16,   # 18 L_Elbow
    17,   # 19 R_Elbow
    18,   # 20 L_Wrist
    19,   # 21 R_Wrist
]

FOOT_JOINTS = [7, 8, 10, 11]   # ankles + toes


# ── Step 1: temporal smoothing ───────────────────────────────────────────────

def smooth_temporal(
    joints: np.ndarray,
    sigma: float = 1.5,
    **_,
) -> np.ndarray:
    """
    Gaussian smoothing along the time axis per joint.
    Removes frame-to-frame jitter introduced by additive delta blending.
    sigma in frames (default 1.5 ≈ 75 ms at 20 fps).
    """
    return gaussian_filter1d(joints.astype(np.float32), sigma=sigma, axis=0)


# ── Step 2: velocity clamping ────────────────────────────────────────────────

def clamp_velocity(
    joints: np.ndarray,
    max_speed: float = 5.0,
    fps: float = 20.0,
    **_,
) -> np.ndarray:
    """
    Cap per-joint displacement per frame to max_speed (m/s).
    Joints moving faster than a sprinting human are rescaled along their
    direction of motion; slower joints are unchanged.
    """
    out      = joints.astype(np.float32).copy()
    max_disp = max_speed / fps
    for t in range(1, len(out)):
        disp  = out[t] - out[t - 1]                      # (22, 3)
        mag   = np.linalg.norm(disp, axis=-1, keepdims=True)   # (22, 1)
        mask  = (mag > max_disp).squeeze(-1)              # (22,)
        if mask.any():
            scale         = np.where(mag > 1e-8, max_disp / mag, 1.0)
            out[t, mask]  = out[t - 1, mask] + (disp * scale)[mask]
    return out


# ── Step 3: acceleration clamping ────────────────────────────────────────────

def clamp_acceleration(
    joints: np.ndarray,
    max_acc: float = 30.0,
    fps: float = 20.0,
    **_,
) -> np.ndarray:
    """
    Cap the change in per-joint velocity between consecutive frames.
    Prevents physically impossible snapping motions.
    max_acc in m/s^2 (default 30 ≈ 3g, well above human sprint acceleration).
    """
    out       = joints.astype(np.float32).copy()
    max_delta = max_acc / (fps * fps)
    prev_disp = np.zeros((22, 3), dtype=np.float32)
    for t in range(1, len(out)):
        disp      = out[t] - out[t - 1]
        delta     = disp - prev_disp
        delta_mag = np.linalg.norm(delta, axis=-1, keepdims=True)
        mask      = (delta_mag > max_delta).squeeze(-1)
        if mask.any():
            scale              = np.where(delta_mag > 1e-8, max_delta / delta_mag, 1.0)
            clamped            = prev_disp + delta * scale
            out[t, mask]       = out[t - 1, mask] + clamped[mask]
            prev_disp          = clamped
        else:
            prev_disp = disp
    return out


# ── Step 4: bone length preservation ─────────────────────────────────────────

def preserve_bone_lengths(
    joints: np.ndarray,
    reference: np.ndarray,
    **_,
) -> np.ndarray:
    """
    Re-project each joint so parent-child distances match the reference skeleton.
    Walks the kinematic chain outward from the root each frame, preserving the
    direction from parent to child but fixing the length to reference[0].

    This corrects limb stretching introduced by additive position blending.
    """
    # Compute rest bone lengths from first frame of reference
    rest = np.linalg.norm(
        reference[0, 1:] - reference[0, [SMPL_PARENTS[j] for j in range(1, 22)]],
        axis=-1,
    )   # (21,)

    out = joints.astype(np.float32).copy()
    for t in range(len(out)):
        for j in range(1, 22):
            p    = SMPL_PARENTS[j]
            diff = out[t, j] - out[t, p]
            n    = np.linalg.norm(diff)
            if n > 1e-6:
                out[t, j] = out[t, p] + diff / n * rest[j - 1]
            else:
                # Degenerate — push child along reference direction
                ref_diff = reference[0, j] - reference[0, p]
                ref_n    = np.linalg.norm(ref_diff)
                if ref_n > 1e-6:
                    out[t, j] = out[t, p] + ref_diff / ref_n * rest[j - 1]
    return out


# ── Step 5: floor constraint ──────────────────────────────────────────────────

def enforce_floor(
    joints: np.ndarray,
    reference: np.ndarray,
    margin: float = 0.0,
    **_,
) -> np.ndarray:
    """
    Prevent foot joints sinking below the ground plane.
    Floor level is estimated from the reference sequence (minimum Y of foot joints).
    If any foot joint in the blended output falls below that level, the entire
    skeleton is lifted uniformly for that frame.
    """
    floor_y  = float(reference[:, FOOT_JOINTS, 1].min()) - margin
    out      = joints.astype(np.float32).copy()
    foot_y   = out[:, FOOT_JOINTS, 1]             # (T, 4)
    min_foot = foot_y.min(axis=1)                 # (T,)
    sink     = np.minimum(min_foot - floor_y, 0)  # (T,) ≤ 0
    out[:, :, 1] -= sink[:, None]                 # lift frames that sank
    return out


# ── Pipeline registry ─────────────────────────────────────────────────────────

# Ordered list of (key, display_name, function).
# Functions that need `reference` receive it as a keyword arg.
STEPS: list[tuple[str, str, callable]] = [
    ("smooth",       "Smooth",            smooth_temporal),
    ("velocity",     "Velocity clamp",    clamp_velocity),
    ("acceleration", "Acceleration clamp", clamp_acceleration),
    ("bone",         "Bone lengths",       preserve_bone_lengths),
    ("floor",        "Floor constraint",   enforce_floor),
]

STEP_KEYS = [k for k, _, _ in STEPS]


def apply_pipeline(
    joints: np.ndarray,
    reference: np.ndarray,
    enabled: list[str] | None = None,
) -> np.ndarray:
    """
    Apply post-processing steps in order.

    Parameters
    ----------
    joints    : (T, 22, 3) blended motion to correct
    reference : (T, 22, 3) source motion used for bone lengths and floor level
    enabled   : list of step keys to apply; None means all steps

    Returns
    -------
    (T, 22, 3) post-processed motion
    """
    active = set(enabled) if enabled is not None else set(STEP_KEYS)
    out    = joints.copy()
    for key, _, fn in STEPS:
        if key in active:
            out = fn(out, reference=reference)
    return out
