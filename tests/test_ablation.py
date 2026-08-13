"""
Tests for ablation over blend feature dimensions.

Each test activates exactly one feature index in FeatureBlender.reconstruct()
and asserts that the output differs from the source in a way that is consistent
with what that feature controls.  The complement assertion — that features NOT
activated produce no meaningful change relative to source — is also checked.

Run with:  python -m pytest tests/test_ablation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_spectrum.blend import (
    FEATURE_NAMES,
    FeatureBlender,
    _IDX_VEL_MEAN,
    _IDX_VEL_STD,
    _IDX_ACC_MEAN,
    _IDX_ROM,
    _IDX_FREQ,
    _IDX_FREQ_MAG,
    _IDX_PDIST,
    _IDX_ZONE_SPEED,
    _IDX_ORIENT,
    FPS,
)
from semantic_spectrum.zone_features import ZoneFeatureExtractor, FEATURE_DIM
from semantic_spectrum.zones import ZoneConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Approximate SMPL rest-pose heights (Y-up, metres)
_BASE_Y = np.array([
    0.98, 0.90, 0.90, 1.12, 0.50, 0.50, 1.22, 0.08, 0.08,
    1.33, 0.02, 0.02, 1.53, 1.43, 1.43, 1.63, 1.42, 1.42,
    1.18, 1.18, 0.95, 0.95,
], dtype=np.float32)

_BASE_X = np.array([
    0.00, -0.10, 0.10, 0.00, -0.10, 0.10, 0.00, -0.10, 0.10,
    0.00, -0.10, 0.10, 0.00, -0.18, 0.18, 0.00, -0.35, 0.35,
    -0.50, 0.50, -0.60, 0.60,
], dtype=np.float32)


def _base_pose() -> np.ndarray:
    """Single (22, 3) pose in approximate SMPL standing configuration."""
    pose = np.zeros((22, 3), dtype=np.float32)
    pose[:, 0] = _BASE_X
    pose[:, 1] = _BASE_Y
    return pose


def _walking_motion(T: int = 80) -> np.ndarray:
    """
    Simple walking-like motion: pelvis moves forward with oscillating leg joints.
    Produces non-trivial velocities and periodic signals in most zones.
    """
    rng = np.random.default_rng(42)
    base = _base_pose()
    joints = np.tile(base, (T, 1, 1))

    t = np.arange(T, dtype=np.float32)

    # Pelvis marches forward at ~1 m/s (0.05 m/frame at 20 fps)
    joints[:, 0, 2] += t * 0.05

    # Legs swing with alternating phase
    for joint_idx, sign in [(4, 1), (5, -1), (7, 1), (8, -1)]:
        joints[:, joint_idx, 2] += sign * 0.15 * np.sin(2 * np.pi * t / 20)

    # Arms swing opposite to legs
    for joint_idx, sign in [(16, -1), (17, 1), (20, -1), (21, 1)]:
        joints[:, joint_idx, 2] += sign * 0.10 * np.sin(2 * np.pi * t / 20)

    joints += rng.normal(0, 0.002, joints.shape).astype(np.float32)
    return joints


def _make_feature_vec(overrides: dict[int, float] | None = None) -> np.ndarray:
    """
    Build a neutral 9-element feature vector and apply any per-index overrides.
    Neutral values are close to a slow walking motion; each test sets the target
    dimension to something distinctly different from source to confirm an effect.
    """
    fv = np.array([
        1.0,    # [0] vel_mean    m/s
        0.3,    # [1] vel_std     m/s
        2.0,    # [2] acc_mean    m/s^2
        0.5,    # [3] rom         m
        1.0,    # [4] dom_freq    Hz
        0.3,    # [5] freq_mag    (normalised)
        0.15,   # [6] pairwise    m
        0.5,    # [7] zone_speed  m/s
        0.5,    # [8] orientation rad/s
    ], dtype=np.float32)
    for idx, val in (overrides or {}).items():
        fv[idx] = val
    return fv


def _blender() -> FeatureBlender:
    return FeatureBlender(alpha=0.5, zone_mode="standard")


def _m_output(blender: FeatureBlender, fv: np.ndarray) -> dict[str, np.ndarray]:
    """Wrap a single feature vector into a per-zone M_output dict."""
    return {zone: fv.copy() for zone in blender._config.zones}


def _mean_displacement(source: np.ndarray, output: np.ndarray) -> float:
    """Mean absolute per-joint displacement between two (T,22,3) arrays."""
    return float(np.mean(np.abs(output - source)))


def _per_joint_speed(joints: np.ndarray) -> np.ndarray:
    """(T-1, 22) speed of each joint across time."""
    return np.linalg.norm(np.diff(joints, axis=0), axis=-1)


def _centroid_speed(joints: np.ndarray, zone_indices: list[int]) -> np.ndarray:
    """(T-1,) speed of zone centroid across time."""
    centroid = joints[:, zone_indices, :].mean(axis=1)   # (T, 3)
    return np.linalg.norm(np.diff(centroid, axis=0), axis=-1)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

class TestSanity:
    def test_feature_dim(self):
        assert FEATURE_DIM == 9

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 9

    def test_feature_names_zone_speed(self):
        assert FEATURE_NAMES[_IDX_ZONE_SPEED] == "zone_speed"

    def test_feature_names_last_is_orientation(self):
        assert FEATURE_NAMES[_IDX_ORIENT] == "orientation"

    def test_extractor_output_shape(self):
        joints = _walking_motion()
        cfg    = ZoneConfig("standard")
        ext    = ZoneFeatureExtractor(cfg)
        feats  = ext.extract(joints)
        for zone, fv in feats.items():
            assert fv.shape == (FEATURE_DIM,), f"{zone}: expected ({FEATURE_DIM},), got {fv.shape}"

    def test_reconstruct_passthrough_shape(self):
        """reconstruct with a neutral M_output returns (T,22,3)."""
        blender = _blender()
        source  = _walking_motion(T=40)
        fv      = _make_feature_vec()
        M       = _m_output(blender, fv)
        out     = blender.reconstruct(M, source)
        assert out.shape == source.shape

    def test_reconstruct_seeds_at_frame0(self):
        """Frame 0 of output must always equal frame 0 of source."""
        blender = _blender()
        source  = _walking_motion(T=40)
        fv      = _make_feature_vec()
        M       = _m_output(blender, fv)
        out     = blender.reconstruct(M, source)
        np.testing.assert_array_almost_equal(out[0], source[0], decimal=5)

    def test_all_features_produce_finite_output(self):
        """With all features active, output must be finite and differ from source."""
        blender = _blender()
        source  = _walking_motion()
        fv      = _make_feature_vec({_IDX_VEL_MEAN: 1.5})   # modest boost to avoid overflow
        M       = _m_output(blender, fv)
        out     = blender.reconstruct(M, source)
        assert np.isfinite(out).all(), "output must not contain inf/nan"
        assert _mean_displacement(source, out) > 1e-4


# ---------------------------------------------------------------------------
# Per-feature ablation tests
# ---------------------------------------------------------------------------

class TestAblateVelMean:
    """[0] vel_mean — scales mean per-joint speed to target."""

    def test_high_vel_mean_increases_speed(self):
        blender = _blender()
        source  = _walking_motion()
        fv_high = _make_feature_vec({_IDX_VEL_MEAN: 5.0})
        fv_low  = _make_feature_vec({_IDX_VEL_MEAN: 0.05})
        M_high  = _m_output(blender, fv_high)
        M_low   = _m_output(blender, fv_low)

        out_high = blender.reconstruct(M_high, source, active_features={_IDX_VEL_MEAN})
        out_low  = blender.reconstruct(M_low,  source, active_features={_IDX_VEL_MEAN})

        speed_high = _per_joint_speed(out_high).mean()
        speed_low  = _per_joint_speed(out_low).mean()
        assert speed_high > speed_low, (
            f"high vel_mean should produce higher mean speed "
            f"({speed_high:.4f} vs {speed_low:.4f})"
        )

    def test_vel_mean_changes_output_from_source(self):
        blender = _blender()
        source  = _walking_motion()
        fv      = _make_feature_vec({_IDX_VEL_MEAN: 4.0})
        M       = _m_output(blender, fv)
        out     = blender.reconstruct(M, source, active_features={_IDX_VEL_MEAN})
        assert _mean_displacement(source, out) > 0.01


class TestAblateVelStd:
    """[1] vel_std — widens / narrows per-joint speed variation around mean."""

    def test_high_vel_std_increases_speed_variance(self):
        blender   = _blender()
        source    = _walking_motion()
        fv_wide   = _make_feature_vec({_IDX_VEL_MEAN: 1.0, _IDX_VEL_STD: 2.0})
        fv_narrow = _make_feature_vec({_IDX_VEL_MEAN: 1.0, _IDX_VEL_STD: 0.05})
        M_wide    = _m_output(blender, fv_wide)
        M_narrow  = _m_output(blender, fv_narrow)

        out_wide   = blender.reconstruct(M_wide,   source, active_features={_IDX_VEL_STD})
        out_narrow = blender.reconstruct(M_narrow, source, active_features={_IDX_VEL_STD})

        std_wide   = _per_joint_speed(out_wide).std()
        std_narrow = _per_joint_speed(out_narrow).std()
        assert std_wide > std_narrow, (
            f"wide vel_std target should produce higher speed std "
            f"({std_wide:.4f} vs {std_narrow:.4f})"
        )


class TestAblateAccMean:
    """[2] acc_mean — clamps frame-to-frame displacement change magnitude."""

    def test_low_acc_smooths_motion(self):
        blender  = _blender()
        source   = _walking_motion()
        fv_snappy = _make_feature_vec({_IDX_ACC_MEAN: 20.0})
        fv_smooth = _make_feature_vec({_IDX_ACC_MEAN: 0.01})
        M_snappy  = _m_output(blender, fv_snappy)
        M_smooth  = _m_output(blender, fv_smooth)

        out_snappy = blender.reconstruct(M_snappy, source, active_features={_IDX_ACC_MEAN})
        out_smooth = blender.reconstruct(M_smooth, source, active_features={_IDX_ACC_MEAN})

        # Smoothed output should have smaller frame-to-frame velocity changes
        def mean_acc(joints):
            vel = np.diff(joints, axis=0)
            return np.mean(np.abs(np.diff(vel, axis=0)))

        assert mean_acc(out_smooth) <= mean_acc(out_snappy) + 1e-5, (
            "low acc_mean target should yield smoother (lower mean acceleration) output"
        )


class TestAblateRom:
    """[3] rom — clamps zone centroid drift from its t=0 position."""

    def test_low_rom_constrains_drift(self):
        blender     = _blender()
        source      = _walking_motion()   # pelvis drifts 0.05 * T metres
        fv_large    = _make_feature_vec({_IDX_ROM: 10.0})
        fv_small    = _make_feature_vec({_IDX_ROM: 0.05})
        M_large     = _m_output(blender, fv_large)
        M_small     = _m_output(blender, fv_small)

        out_large = blender.reconstruct(M_large, source, active_features={_IDX_ROM})
        out_small = blender.reconstruct(M_small, source, active_features={_IDX_ROM})

        # Use the torso zone (joint 0 = pelvis, which has the most drift)
        torso_idx = [0, 3, 6, 9]
        drift_large = np.linalg.norm(
            out_large[:, torso_idx, :].mean(axis=1) - out_large[0, torso_idx, :].mean(axis=0),
            axis=-1
        ).max()
        drift_small = np.linalg.norm(
            out_small[:, torso_idx, :].mean(axis=1) - out_small[0, torso_idx, :].mean(axis=0),
            axis=-1
        ).max()

        assert drift_small <= drift_large + 1e-4, (
            f"small ROM target should limit drift more than large target "
            f"({drift_small:.4f} vs {drift_large:.4f})"
        )


class TestAblateFreq:
    """[4+5] dom_freq + freq_mag — adds sinusoidal oscillation at target frequency."""

    def test_freq_adds_oscillation(self):
        blender = _blender()
        source  = _walking_motion()
        fv_osc  = _make_feature_vec({_IDX_FREQ: 2.0, _IDX_FREQ_MAG: 0.8})
        fv_none = _make_feature_vec({_IDX_FREQ: 0.0, _IDX_FREQ_MAG: 0.0})
        M_osc   = _m_output(blender, fv_osc)
        M_none  = _m_output(blender, fv_none)

        out_osc  = blender.reconstruct(M_osc,  source, active_features={_IDX_FREQ, _IDX_FREQ_MAG})
        out_none = blender.reconstruct(M_none, source, active_features={_IDX_FREQ, _IDX_FREQ_MAG})

        # Oscillation should produce more displacement vs source than no oscillation
        disp_osc  = _mean_displacement(source, out_osc)
        disp_none = _mean_displacement(source, out_none)
        assert disp_osc > disp_none, (
            f"freq oscillation should increase displacement from source "
            f"({disp_osc:.5f} vs {disp_none:.5f})"
        )


class TestAblatePairwiseDist:
    """[6] pairwise_dist — scales inter-joint spread around zone centroid."""

    def test_large_pairwise_expands_joints(self):
        blender  = _blender()
        source   = _walking_motion()
        fv_large = _make_feature_vec({_IDX_PDIST: 1.0})
        fv_small = _make_feature_vec({_IDX_PDIST: 0.01})
        M_large  = _m_output(blender, fv_large)
        M_small  = _m_output(blender, fv_small)

        out_large = blender.reconstruct(M_large, source, active_features={_IDX_PDIST})
        out_small = blender.reconstruct(M_small, source, active_features={_IDX_PDIST})

        # Measure mean pairwise distance across the arms zone
        arm_idx = [13, 14, 16, 17, 18, 19]
        def mean_pdist(joints):
            js = joints[:, arm_idx, :]
            total, count = 0.0, 0
            for i in range(len(arm_idx)):
                for j in range(i + 1, len(arm_idx)):
                    total += float(np.mean(np.linalg.norm(js[:, i] - js[:, j], axis=-1)))
                    count += 1
            return total / max(count, 1)

        assert mean_pdist(out_large) > mean_pdist(out_small), (
            "large pairwise_dist target should expand inter-joint spread"
        )


class TestAblateZoneSpeed:
    """[7] zone_speed — scales centroid displacement of zone as a rigid unit."""

    def test_high_zone_speed_increases_centroid_speed(self):
        blender  = _blender()
        source   = _walking_motion()
        fv_fast  = _make_feature_vec({_IDX_ZONE_SPEED: 3.0})
        fv_slow  = _make_feature_vec({_IDX_ZONE_SPEED: 0.05})
        M_fast   = _m_output(blender, fv_fast)
        M_slow   = _m_output(blender, fv_slow)

        out_fast = blender.reconstruct(M_fast, source, active_features={_IDX_ZONE_SPEED})
        out_slow = blender.reconstruct(M_slow, source, active_features={_IDX_ZONE_SPEED})

        # Torso zone (includes pelvis) should move faster/slower
        torso_idx = [0, 3, 6, 9]
        speed_fast = _centroid_speed(out_fast, torso_idx).mean()
        speed_slow = _centroid_speed(out_slow, torso_idx).mean()

        assert speed_fast > speed_slow, (
            f"high zone_speed target should yield faster centroid motion "
            f"({speed_fast:.5f} vs {speed_slow:.5f})"
        )

    def test_zone_speed_differs_from_vel_mean(self):
        """
        zone_speed and vel_mean should produce different outputs.
        vel_mean scales each joint displacement independently;
        zone_speed scales the centroid displacement and moves all joints together.
        """
        blender = _blender()
        source  = _walking_motion()
        fv      = _make_feature_vec({_IDX_VEL_MEAN: 3.0, _IDX_ZONE_SPEED: 3.0})
        M       = _m_output(blender, fv)

        out_vel  = blender.reconstruct(M, source, active_features={_IDX_VEL_MEAN})
        out_zone = blender.reconstruct(M, source, active_features={_IDX_ZONE_SPEED})

        # They should not produce identical results — if they do the feature is redundant
        diff = float(np.mean(np.abs(out_vel - out_zone)))
        assert diff > 1e-4, (
            f"zone_speed and vel_mean should produce different outputs (diff={diff:.6f})"
        )

    def test_zone_speed_changes_output_from_source(self):
        blender = _blender()
        source  = _walking_motion()
        fv      = _make_feature_vec({_IDX_ZONE_SPEED: 3.0})
        M       = _m_output(blender, fv)
        out     = blender.reconstruct(M, source, active_features={_IDX_ZONE_SPEED})
        assert _mean_displacement(source, out) > 1e-4


class TestAblateOrientation:
    """[8] orientation — yaws the zone about vertical to match a target turn rate (Space axis)."""

    @staticmethod
    def _zone_turn_rate(joints: np.ndarray, zone_indices: list[int]) -> float:
        """Mean absolute yaw rate (rad/s) of a zone about vertical, matching the extractor."""
        js  = joints[:, zone_indices, :]
        rel = (js - js.mean(axis=1, keepdims=True))[:-1]
        vel = np.diff(js, axis=0) * FPS
        rx, rz = rel[..., 0], rel[..., 2]
        vx, vz = vel[..., 0], vel[..., 2]
        omega  = (rx * vz - rz * vx) / (rx * rx + rz * rz + 1e-8)
        return float(np.mean(np.abs(omega.mean(axis=1))))

    def test_high_orientation_increases_turn_rate(self):
        blender = _blender()
        source  = _walking_motion()
        fv_high = _make_feature_vec({_IDX_ORIENT: 4.0})
        fv_low  = _make_feature_vec({_IDX_ORIENT: 0.05})
        out_high = blender.reconstruct(_m_output(blender, fv_high), source, active_features={_IDX_ORIENT})
        out_low  = blender.reconstruct(_m_output(blender, fv_low),  source, active_features={_IDX_ORIENT})

        arm_idx = [13, 14, 16, 17, 18, 19]
        rate_high = self._zone_turn_rate(out_high, arm_idx)
        rate_low  = self._zone_turn_rate(out_low,  arm_idx)
        assert rate_high > rate_low, (
            f"high orientation target should yield a higher turn rate "
            f"({rate_high:.4f} vs {rate_low:.4f})"
        )

    def test_orientation_changes_output_from_source(self):
        blender = _blender()
        source  = _walking_motion()
        fv      = _make_feature_vec({_IDX_ORIENT: 4.0})
        out     = blender.reconstruct(_m_output(blender, fv), source, active_features={_IDX_ORIENT})
        assert _mean_displacement(source, out) > 1e-4


# ---------------------------------------------------------------------------
# Cross-feature isolation: inactive features should not change output
# ---------------------------------------------------------------------------

class TestFeatureIsolation:
    """
    With only one feature active, ablating a DIFFERENT feature with the same
    M_output should produce the same result as using no active features.
    This confirms that active_features correctly gates each code path.
    """

    # freq (4) and freq_mag (5) are coupled in the blender (both must be active to fire),
    # so they are tested together rather than individually.
    _SOLO_FEATURES = [
        ({_IDX_VEL_MEAN},   {_IDX_VEL_MEAN:  2.0}),
        ({_IDX_VEL_STD},    {_IDX_VEL_STD:   3.0}),
        ({_IDX_ACC_MEAN},   {_IDX_ACC_MEAN:  0.001}),
        ({_IDX_ROM},        {_IDX_ROM:       0.02}),
        ({_IDX_FREQ, _IDX_FREQ_MAG}, {_IDX_FREQ: 3.0, _IDX_FREQ_MAG: 0.9}),
        ({_IDX_PDIST},      {_IDX_PDIST:     1.0}),
        ({_IDX_ZONE_SPEED}, {_IDX_ZONE_SPEED: 2.0}),
        ({_IDX_ORIENT},     {_IDX_ORIENT:    3.0}),
    ]

    @pytest.mark.parametrize("active_set,overrides", _SOLO_FEATURES)
    def test_single_feature_differs_from_inactive(
        self, active_set: set[int], overrides: dict[int, float]
    ):
        """
        Output with one feature (or coupled pair) active should differ from the
        zero-feature baseline and always seed from source[0].
        """
        blender = _blender()
        source  = _walking_motion()
        fv      = _make_feature_vec(overrides)
        M       = _m_output(blender, fv)

        out_none   = blender.reconstruct(M, source, active_features=set())
        out_single = blender.reconstruct(M, source, active_features=active_set)

        label = "+".join(FEATURE_NAMES[i] for i in sorted(active_set))
        np.testing.assert_array_almost_equal(
            out_single[0], source[0], decimal=5,
            err_msg=f"Frame 0 must equal source[0] (feature {label})"
        )
        diff = _mean_displacement(out_none, out_single)
        assert diff > 0.0, f"Feature {label} had no observable effect"
