"""
Feature-space blending for the Phase 2 zone-aware pipeline.

Two-step process
----------------
Step 1  — Feature update (per zone i):
    M'_orig,i  = M_orig,i + delta_i
    delta_i    = alpha * ||M_orig,i|| * (M_mom,i - M_orig,i)

Step 2  — Autoregressive reconstruction (per frame t = 1 … T):
    J_out(t) = J_out(t-1) ⊛ M'_orig,i
    J_out(0) = J_original(0)

The ⊛ operator applies all nine feature dimensions:

  [0] vel_mean   — scale displacement so mean zone speed matches target
  [1] vel_std    — spread per-joint speeds around the mean by target std
  [2] acc_mean   — smooth displacement changes; high acc = snappy, low = smooth
  [3] rom        — clamp zone centroid drift from its t=0 position
  [4] dom_freq   — add sinusoidal oscillation at target frequency
  [5] freq_mag   — amplitude of the sinusoidal modulation
  [6] pairwise   — scale inter-joint spread around zone centroid
  [7] zone_speed — scale the zone centroid displacement (rigid-body translation)
  [8] orientation — yaw the zone about vertical to match a target turn rate (the Space axis)
"""

from __future__ import annotations

import numpy as np

from semantic_spectrum.zones import ZoneConfig

FPS = 20.0

# Feature vector indices (must match ZoneFeatureExtractor output order)
_IDX_VEL_MEAN  = 0
_IDX_VEL_STD   = 1
_IDX_ACC_MEAN  = 2
_IDX_ROM       = 3
_IDX_FREQ      = 4
_IDX_FREQ_MAG  = 5
_IDX_PDIST     = 6
_IDX_ZONE_SPEED = 7   # centroid displacement magnitude (zone-level speed)
_IDX_ORIENT     = 8   # yaw/turn rate about vertical (the Space axis)

# Human-readable names for ablation outputs
FEATURE_NAMES = [
    "vel_mean",
    "vel_std",
    "acc_mean",
    "rom",
    "dom_freq",
    "freq_mag",
    "pairwise_dist",
    "zone_speed",
    "orientation",
]


class FeatureBlender:
    """
    Blend original and MoMask feature vectors, then reconstruct the output
    joint sequence autoregressively using all nine kinematic feature dimensions.
    """

    def __init__(self, alpha: float = 0.5, zone_mode: str = "standard"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha     = alpha
        self.zone_mode = zone_mode
        self._config   = ZoneConfig(zone_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_blend(
        self,
        M_original: dict[str, np.ndarray],
        M_momask:   dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Step 1: compute updated feature vectors per zone.

        M'_orig,i = M_orig,i + alpha * ||M_orig,i|| * (M_mom,i - M_orig,i)
        """
        M_output: dict[str, np.ndarray] = {}
        for zone in M_original:
            if zone not in M_momask:
                M_output[zone] = M_original[zone].copy()
                continue
            m_o   = M_original[zone].astype(np.float32)
            m_m   = M_momask[zone].astype(np.float32)
            mag   = float(np.linalg.norm(m_o))
            delta = self.alpha * mag * (m_m - m_o)
            M_output[zone] = m_o + delta
        return M_output

    def reconstruct(
        self,
        M_output:        dict[str, np.ndarray],
        original_joints: np.ndarray,
        active_features: set[int] | None = None,
    ) -> np.ndarray:
        """
        Step 2: autoregressively build the output joint sequence.

        J_out(0) = J_original(0)  — seed from performer's starting pose.
        J_out(t) = J_out(t-1) ⊛ M'_orig,i  for t = 1 … T-1

        Parameters
        ----------
        active_features : set of feature indices to apply. If None, all are
                          used. Pass a single-element set (e.g. {0}) to isolate
                          one feature for ablation.

        Per-zone state is carried across frames:
          - prev_disp   : previous output displacement (for acc smoothing)
          - zone_origin : centroid at t=0 (for ROM clamping)
        """
        if original_joints.ndim != 3 or original_joints.shape[1:] != (22, 3):
            raise ValueError(
                f"Expected (T, 22, 3) joints, got {original_joints.shape}"
            )

        T         = original_joints.shape[0]
        out       = np.zeros_like(original_joints)
        out[0]    = original_joints[0]
        orig_disp = np.diff(original_joints, axis=0)   # (T-1, 22, 3)

        # Per-zone running state
        prev_disp   = {
            zone: np.zeros((len(indices), 3), dtype=np.float32)
            for zone, indices in self._config.zones.items()
        }
        zone_origin = {
            zone: out[0, indices].copy()
            for zone, indices in self._config.zones.items()
        }

        for t in range(1, T):
            frame = out[t - 1].copy()
            for zone, indices in self._config.zones.items():
                if zone not in M_output:
                    frame[indices] = out[t - 1, indices]
                    continue

                new_pos, new_disp = self._apply_motion_vec(
                    J_prev=out[t - 1, indices],
                    orig_disp_t=orig_disp[t - 1, indices],
                    feature_vec=M_output[zone],
                    t=t,
                    prev_disp=prev_disp[zone],
                    zone_origin=zone_origin[zone],
                    active_features=active_features,
                )
                # Guard against NaN/inf from chained feature scaling
                if not np.isfinite(new_pos).all():
                    new_pos  = out[t - 1, indices].copy()
                    new_disp = np.zeros_like(new_disp)
                frame[indices]    = new_pos
                prev_disp[zone]   = new_disp
            out[t] = frame

        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_motion_vec(
        self,
        J_prev:          np.ndarray,          # (k, 3) joints at t-1
        orig_disp_t:     np.ndarray,          # (k, 3) original displacement at frame t
        feature_vec:     np.ndarray,          # (7,)   updated feature vector for this zone
        t:               int,                 # current frame index
        prev_disp:       np.ndarray,          # (k, 3) output displacement at t-1
        zone_origin:     np.ndarray,          # (k, 3) zone joint positions at t=0
        active_features: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply all nine feature dimensions to advance zone joints by one frame.

        Returns
        -------
        new_positions : (k, 3)
        new_disp      : (k, 3)  — stored as prev_disp for the next frame
        """
        # Helper: is this feature active in the current ablation run?
        def on(idx: int) -> bool:
            return active_features is None or idx in active_features

        target_vel_mean   = float(feature_vec[_IDX_VEL_MEAN])   / FPS   # m/frame
        target_vel_std    = float(feature_vec[_IDX_VEL_STD])    / FPS
        target_acc_mean   = float(feature_vec[_IDX_ACC_MEAN])   / (FPS * FPS)
        target_rom        = float(feature_vec[_IDX_ROM])
        target_freq       = float(feature_vec[_IDX_FREQ])
        target_freq_mag   = float(feature_vec[_IDX_FREQ_MAG])
        target_pdist      = float(feature_vec[_IDX_PDIST])
        target_zone_speed = (float(feature_vec[_IDX_ZONE_SPEED]) / FPS
                             if len(feature_vec) > _IDX_ZONE_SPEED else 0.0)
        target_orient     = (float(feature_vec[_IDX_ORIENT])
                             if len(feature_vec) > _IDX_ORIENT else 0.0)   # rad/s

        # ---- 1. Velocity mean: scale displacement to match target speed ----
        orig_speeds     = np.linalg.norm(orig_disp_t, axis=-1)   # (k,)
        mean_orig_speed = float(orig_speeds.mean())

        if on(_IDX_VEL_MEAN) and mean_orig_speed > 1e-6:
            disp = orig_disp_t * (target_vel_mean / mean_orig_speed)
        else:
            disp = orig_disp_t.copy()

        # ---- 2. Velocity std: spread per-joint speeds around the mean ----
        if on(_IDX_VEL_STD) and target_vel_std > 1e-8 and mean_orig_speed > 1e-6:
            joint_speeds = np.linalg.norm(disp, axis=-1)
            cur_std      = float(joint_speeds.std())
            cur_mean     = float(joint_speeds.mean())
            if cur_std > 1e-8 and cur_mean > 1e-8:
                std_ratio   = np.clip(target_vel_std / cur_std, 0.1, 10.0)
                scale_per_j = 1.0 + (joint_speeds - cur_mean) / cur_mean * (std_ratio - 1.0)
                scale_per_j = np.clip(scale_per_j, 0.1, 10.0)
                disp        = disp * scale_per_j[:, None]

        # ---- 3. Acceleration: smooth transitions between frames ----
        if on(_IDX_ACC_MEAN) and target_acc_mean > 1e-8 and np.linalg.norm(prev_disp) > 1e-6:
            delta      = disp - prev_disp
            delta_mag  = np.linalg.norm(delta, axis=-1, keepdims=True)
            too_large  = (delta_mag > target_acc_mean).squeeze(-1)
            if too_large.any():
                clamped         = prev_disp + delta / (delta_mag + 1e-8) * target_acc_mean
                disp[too_large] = clamped[too_large]

        # ---- 4 & 5. Frequency + magnitude: sinusoidal oscillation ----
        if on(_IDX_FREQ) and on(_IDX_FREQ_MAG) and target_freq > 1e-4 and target_freq_mag > 1e-4:
            osc_amplitude = target_freq_mag * target_vel_mean
            osc_value     = osc_amplitude * np.sin(2.0 * np.pi * target_freq * t / FPS)
            centroid_disp = disp.mean(axis=0)
            centroid_norm = float(np.linalg.norm(centroid_disp))
            if centroid_norm > 1e-8:
                disp = disp + osc_value * (centroid_disp / centroid_norm)

        # Advance positions
        new_positions = J_prev + disp

        # ---- 6. ROM: clamp zone centroid drift from its t=0 position ----
        if on(_IDX_ROM) and target_rom > 1e-6:
            centroid_new    = new_positions.mean(axis=0)
            centroid_origin = zone_origin.mean(axis=0)
            drift           = float(np.linalg.norm(centroid_new - centroid_origin))
            if drift > target_rom:
                pull          = (centroid_new - centroid_origin) * (1.0 - target_rom / drift)
                new_positions = new_positions - pull

        # ---- 7. Pairwise distance: scale inter-joint spread ----
        if on(_IDX_PDIST) and target_pdist > 1e-6 and J_prev.shape[0] >= 2:
            centroid  = new_positions.mean(axis=0)
            offsets   = new_positions - centroid
            diff_mat  = new_positions[:, None, :] - new_positions[None, :, :]
            dist_mat  = np.linalg.norm(diff_mat, axis=-1)
            k         = new_positions.shape[0]
            cur_pdist = float(dist_mat.sum()) / (k * (k - 1) + 1e-8)
            if cur_pdist > 1e-8:
                pdist_scale   = np.clip(target_pdist / cur_pdist, 0.5, 2.0)
                new_positions = centroid + offsets * pdist_scale

        # ---- 8. Zone speed: scale centroid displacement to match target ----
        # Complements vel_mean (per-joint) by acting on the zone as a rigid unit.
        if on(_IDX_ZONE_SPEED) and target_zone_speed > 1e-8:
            centroid_disp = orig_disp_t.mean(axis=0)           # (3,) zone bulk motion
            cur_zone_speed = float(np.linalg.norm(centroid_disp))
            if cur_zone_speed > 1e-6:
                zone_scale    = np.clip(target_zone_speed / cur_zone_speed, 0.1, 5.0)
                new_positions = J_prev + (new_positions - J_prev) * zone_scale

        # ---- 9. Orientation: yaw the zone about vertical to match target turn rate ----
        # The Space axis: accumulate rotation about the zone centroid at target_orient rad/s.
        # Turn direction follows the zone's original rotation this frame (default +).
        if on(_IDX_ORIENT) and target_orient > 1e-6 and J_prev.shape[0] >= 2:
            r0     = J_prev - J_prev.mean(axis=0)
            omega0 = float(np.mean((r0[:, 0] * orig_disp_t[:, 2] - r0[:, 2] * orig_disp_t[:, 0])
                                   / (r0[:, 0] ** 2 + r0[:, 2] ** 2 + 1e-8)))
            sign   = 1.0 if omega0 >= 0.0 else -1.0
            dtheta = sign * target_orient / FPS          # radians this frame
            ca, sa = np.cos(dtheta), np.sin(dtheta)
            c      = new_positions.mean(axis=0)
            rel    = new_positions - c
            xr     =  rel[:, 0] * ca + rel[:, 2] * sa
            zr     = -rel[:, 0] * sa + rel[:, 2] * ca
            new_positions = c + np.stack([xr, rel[:, 1], zr], axis=-1)

        new_disp = new_positions - J_prev
        return new_positions, new_disp
