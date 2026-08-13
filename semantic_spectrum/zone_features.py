"""
Per-zone kinematic feature extraction for the Phase 2 blending pipeline.

ZoneFeatureExtractor applies a fixed set of kinematic feature extractors
(velocity, acceleration, range of motion, periodicity) to the joint subset
belonging to each zone, returning a dict of feature vectors keyed by zone name.

Feature vector layout per zone (9 values):
  [0] mean joint speed  (m/s)
  [1] std  joint speed
  [2] mean abs acceleration  (m/s^2)
  [3] range-of-motion volume  (m^3, cube-root normalised)
  [4] dominant FFT frequency  (Hz)
  [5] FFT magnitude of dominant frequency (normalised)
  [6] mean pairwise joint distance within zone (m)
  [7] zone centroid speed  (m/s) -- rigid-body translation speed of the zone
  [8] orientation rate  (rad/s) -- mean turn/twist of the zone about vertical (the Space axis)
"""

from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq

from semantic_spectrum.zones import ZoneConfig

FPS = 20.0
FEATURE_DIM = 9


class ZoneFeatureExtractor:
    """Extract per-zone feature vectors from a (T, 22, 3) joint array."""

    def __init__(self, zone_config: ZoneConfig):
        self.zone_config = zone_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, joints: np.ndarray) -> dict[str, np.ndarray]:
        """
        Parameters
        ----------
        joints : (T, 22, 3)  float32/float64, Y-up, metres, 20 fps

        Returns
        -------
        dict mapping zone name -> feature vector of shape (FEATURE_DIM,)
        """
        if joints.ndim != 3 or joints.shape[1] != 22 or joints.shape[2] != 3:
            raise ValueError(f"Expected (T, 22, 3) joints, got {joints.shape}")

        return {
            zone: self._extract_zone(joints[:, indices, :])
            for zone, indices in self.zone_config.zones.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_zone(self, zone_joints: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        zone_joints : (T, k, 3) for k joints in the zone

        Returns
        -------
        feature vector of shape (FEATURE_DIM,)
        """
        vel_mean, vel_std = self._velocity(zone_joints)
        acc_mean          = self._acceleration(zone_joints)
        rom               = self._range_of_motion(zone_joints)
        freq, freq_mag    = self._periodicity(zone_joints)
        pairwise          = self._pairwise_distance(zone_joints)
        zone_speed        = self._zone_centroid_speed(zone_joints)
        orient            = self._orientation_rate(zone_joints)

        return np.array([vel_mean, vel_std, acc_mean, rom,
                         freq, freq_mag, pairwise, zone_speed, orient], dtype=np.float32)

    def _velocity(self, zone_joints: np.ndarray) -> tuple[float, float]:
        """Mean and std of per-joint speed across time."""
        vel   = np.diff(zone_joints, axis=0) * FPS   # (T-1, k, 3)
        speed = np.linalg.norm(vel, axis=-1)           # (T-1, k)
        return float(np.mean(speed)), float(np.std(speed))

    def _acceleration(self, zone_joints: np.ndarray) -> float:
        """Mean absolute acceleration across all joints in the zone."""
        vel = np.diff(zone_joints, axis=0) * FPS      # (T-1, k, 3)
        acc = np.diff(vel, axis=0) * FPS               # (T-2, k, 3)
        return float(np.mean(np.abs(acc)))

    def _range_of_motion(self, zone_joints: np.ndarray) -> float:
        """Cube-root of bounding-box volume across the sequence."""
        lo  = zone_joints.reshape(-1, 3).min(axis=0)
        hi  = zone_joints.reshape(-1, 3).max(axis=0)
        vol = float(np.prod(np.clip(hi - lo, 0, None)))
        return float(np.cbrt(vol))

    def _periodicity(self, zone_joints: np.ndarray) -> tuple[float, float]:
        """
        Dominant FFT frequency and its normalised magnitude,
        computed from the mean per-frame displacement of the zone centroid.
        """
        T = zone_joints.shape[0]
        centroid = zone_joints.mean(axis=1)           # (T, 3)
        signal   = np.linalg.norm(centroid, axis=-1)  # (T,)
        signal  -= signal.mean()

        freqs = rfftfreq(T, d=1.0 / FPS)
        mags  = np.abs(rfft(signal))
        # exclude DC
        if len(mags) > 1:
            mags[0] = 0.0
        if mags.max() < 1e-8:
            return 0.0, 0.0
        idx     = int(np.argmax(mags))
        dom_freq = float(freqs[idx])
        dom_mag  = float(mags[idx] / (mags.sum() + 1e-8))
        return dom_freq, dom_mag

    def _zone_centroid_speed(self, zone_joints: np.ndarray) -> float:
        """Mean speed of the zone centroid — how fast the zone moves as a rigid unit."""
        centroid = zone_joints.mean(axis=1)               # (T, 3)
        vel      = np.diff(centroid, axis=0) * FPS        # (T-1, 3)
        return float(np.mean(np.linalg.norm(vel, axis=-1)))

    def _orientation_rate(self, zone_joints: np.ndarray) -> float:
        """
        Mean absolute turn/twist rate of the zone about the vertical (Y) axis, in rad/s.

        This is the Space dimension of Laban Movement Analysis: how much the body path
        rotates (turning, spinning, spatial re-facing). For each frame and joint we take the
        horizontal angular velocity about the zone centroid, (r x v)_y / |r|^2, average over
        joints for a per-frame zone angular velocity, then mean its magnitude over time.
        """
        T, k = zone_joints.shape[0], zone_joints.shape[1]
        if T < 2 or k < 2:
            return 0.0
        centroid = zone_joints.mean(axis=1, keepdims=True)   # (T, 1, 3)
        rel      = (zone_joints - centroid)[:-1]              # (T-1, k, 3) offset at frame t
        vel      = np.diff(zone_joints, axis=0) * FPS         # (T-1, k, 3) joint velocity
        rx, rz   = rel[..., 0], rel[..., 2]
        vx, vz   = vel[..., 0], vel[..., 2]
        r2       = rx * rx + rz * rz
        omega    = (rx * vz - rz * vx) / (r2 + 1e-8)          # (T-1, k) per-joint yaw rate
        zone_omega = omega.mean(axis=1)                       # (T-1,) zone yaw rate per frame
        return float(np.mean(np.abs(zone_omega)))

    def _pairwise_distance(self, zone_joints: np.ndarray) -> float:
        """Mean pairwise Euclidean distance between joints within the zone (averaged over time)."""
        k = zone_joints.shape[1]
        if k < 2:
            return 0.0
        dists = []
        for i in range(k):
            for j in range(i + 1, k):
                d = np.linalg.norm(zone_joints[:, i, :] - zone_joints[:, j, :], axis=-1)
                dists.append(float(np.mean(d)))
        return float(np.mean(dists))
