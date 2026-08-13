"""
motion_io.py — shared pose-loading helpers used by similarity_align.py
and render_spectrum_tour.py.

Ported from identity_preservation/scripts/webcam_spectrum.py (Neekita Lian).
Adapted to use the existing video_bridge for MediaPipe extraction instead
of the legacy solutions API.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_FRAMES = 196


# ---------------------------------------------------------------------------
# MediaPipe 33 → SMPL 22 mapping
# (Neekita's version — dict-of-landmarks input from legacy solutions API)
# ---------------------------------------------------------------------------

def mediapipe_to_smpl22(mp_landmarks: list[dict]) -> np.ndarray:
    def g(i):
        lm = mp_landmarks[i]
        return np.array([lm["x"], lm["y"], lm["z"]], dtype=np.float32)

    Lhip, Rhip = g(23), g(24)
    Lsho, Rsho = g(11), g(12)
    pelvis = (Lhip + Rhip) * 0.5
    neck   = (Lsho + Rsho) * 0.5
    chest  = pelvis + (neck - pelvis) * 0.8

    j = np.zeros((22, 3), dtype=np.float32)
    j[0]  = pelvis
    j[1]  = Lhip;  j[2]  = Rhip
    j[3]  = pelvis + (neck - pelvis) * 0.33
    j[4]  = g(25); j[5]  = g(26)
    j[6]  = pelvis + (neck - pelvis) * 0.66
    j[7]  = g(27); j[8]  = g(28)
    j[9]  = chest
    j[10] = g(31); j[11] = g(32)
    j[12] = neck
    j[13] = Lsho;  j[14] = Rsho
    j[15] = g(0)
    j[16] = g(13); j[17] = g(14)
    j[18] = g(15); j[19] = g(16)
    j[20] = g(15) + (g(15) - g(13)) * 0.15
    j[21] = g(16) + (g(16) - g(14)) * 0.15
    return j


def normalise_worldish(js: np.ndarray, target_torso: float = 0.55) -> np.ndarray:
    j = js.copy()
    j[..., 1] *= -1
    torso = float(np.linalg.norm(j[:, 12] - j[:, 0], axis=-1).mean())
    j *= target_torso / max(torso, 1e-6)
    pelvis_mean_xz = j[:, 0, :].mean(axis=0)
    pelvis_mean_xz[1] = 0.0
    j -= pelvis_mean_xz
    j[..., 1] -= float(j[..., 1].min())
    return j.astype(np.float32)


def confidence_smooth(joints_seq: np.ndarray, mp_frames: list, threshold: float = 0.5) -> np.ndarray:
    SMPL_TO_MP = {
        0: [23, 24], 1: [23], 2: [24], 4: [25], 5: [26], 7: [27], 8: [28],
        10: [31], 11: [32], 12: [11, 12], 13: [11], 14: [12], 15: [0],
        16: [13], 17: [14], 18: [15], 19: [16], 20: [15], 21: [16],
        3: [11, 12, 23, 24], 6: [11, 12, 23, 24], 9: [11, 12, 23, 24],
    }
    T = joints_seq.shape[0]
    smoothed = joints_seq.copy()
    fixed = 0
    for smpl_j, mp_ids in SMPL_TO_MP.items():
        vis = np.array([
            min(mp_frames[t][mp_i]["visibility"] for mp_i in mp_ids)
            for t in range(T)
        ])
        good = vis >= threshold
        if good.all() or not good.any():
            continue
        good_idx = np.where(good)[0]
        for t in np.where(~good)[0]:
            left  = good_idx[good_idx < t]
            right = good_idx[good_idx > t]
            if len(left) and len(right):
                l, r  = left[-1], right[0]
                alpha = (t - l) / (r - l)
                smoothed[t, smpl_j] = (1 - alpha) * joints_seq[l, smpl_j] + alpha * joints_seq[r, smpl_j]
            elif len(left):
                smoothed[t, smpl_j] = joints_seq[left[-1], smpl_j]
            elif len(right):
                smoothed[t, smpl_j] = joints_seq[right[0], smpl_j]
            fixed += 1
    if fixed:
        print(f"[motion_io] confidence smoothing: {fixed} low-visibility samples interpolated")
    return smoothed


def extract_frames(video_path: Path) -> list[dict]:
    """Extract MediaPipe 33-landmark frames from a video.
    Returns list of per-frame landmark-dicts compatible with mediapipe_to_smpl22."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[motion_io] {video_path.name}  fps={fps:.1f}  frames={total}")

    pose = mp.solutions.pose.Pose(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    frames = []
    detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)
        if res.pose_landmarks:
            frames.append([
                {"x": float(lm.x), "y": float(lm.y),
                 "z": float(lm.z), "visibility": float(lm.visibility)}
                for lm in res.pose_landmarks.landmark
            ])
            detected += 1
        else:
            frames.append(frames[-1] if frames else
                          [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}] * 33)
    cap.release()
    pose.close()
    print(f"[motion_io] detected pose in {detected}/{len(frames)} frames")
    return frames


def standardise_length(js: np.ndarray, T: int = NUM_FRAMES) -> np.ndarray:
    if js.shape[0] > T:
        idx = np.linspace(0, js.shape[0] - 1, T).astype(int)
        return js[idx]
    if js.shape[0] < T:
        pad = np.repeat(js[-1:], T - js.shape[0], axis=0)
        return np.concatenate([js, pad], axis=0)
    return js


def load_joints(path: Path, label: str = "") -> np.ndarray:
    """
    Load (T, 22, 3) joints from either a .npy/.npz or an .mp4 video.
    Applies normalisation so both sources are in the same coordinate frame.
    """
    tag = f"[{label}] " if label else ""
    p   = Path(path)

    if p.suffix.lower() in (".npy", ".npz"):
        print(f"{tag}loading joints from {p.name}")
        if p.suffix.lower() == ".npz":
            data = np.load(p)
            j = data["joints"] if "joints" in data else data["motion"]
        else:
            j = np.load(p)
        j = j.astype(np.float32)
        if j.ndim == 4:
            j = j[0]
        if j.shape[1:] != (22, 3):
            raise ValueError(f"{p} has shape {j.shape} — expected (T, 22, 3)")
        j = standardise_length(j, NUM_FRAMES)
        return _common_normalise(j)

    elif p.suffix.lower() in (".mp4", ".avi", ".mov"):
        print(f"{tag}extracting MediaPipe pose from {p.name}")
        frames = extract_frames(p)
        j = np.stack([mediapipe_to_smpl22(f) for f in frames])
        j = confidence_smooth(j, frames, threshold=0.5)
        j = standardise_length(j, NUM_FRAMES)
        return normalise_worldish(j)

    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def _common_normalise(js: np.ndarray, target_torso: float = 0.55) -> np.ndarray:
    j = js.copy()
    torso = float(np.linalg.norm(j[:, 12] - j[:, 0], axis=-1).mean())
    j *= target_torso / max(torso, 1e-6)
    pelvis_xz = j[:, 0, :].mean(axis=0)
    pelvis_xz[1] = 0.0
    j -= pelvis_xz
    j[..., 1] -= float(j[..., 1].min())
    return j.astype(np.float32)
