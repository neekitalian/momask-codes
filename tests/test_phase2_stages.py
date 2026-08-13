"""
Unit tests for Phase 2 Stages 2.2 - 2.4:
  estimate_z  (z estimation and integration)
  calibrate   (binning and feedback calibration)
  evaluate    (generalizability metrics)

Pure kinematics: no torch, no GPU. Run with `pytest tests/test_phase2_stages.py`.
"""

import os

import numpy as np
import pytest

from semantic_spectrum.estimate_z import (
    estimate_z, estimate_prompt, format_prompt, ZEstimate)
from semantic_spectrum.calibrate import (
    mode_grid, snap_to_grid, bin_z, calibrate_alpha, load_feedback)
from semantic_spectrum import evaluate as ev

CSV = os.path.join(os.path.dirname(__file__), "..", "semantic_spectrum_data", "round2_summary.csv")


class _StubAnalyzer:
    """Analyzer returning fixed scores, so estimate_z logic is tested deterministically."""
    def __init__(self, scores):
        self._scores = scores
    def analyze(self, motion):
        return dict(self._scores)


def _synthetic(T=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((22, 3)).astype(np.float32)
    t = np.linspace(0, 4 * np.pi, T)[:, None, None]
    return (base[None] + 0.1 * np.sin(t)).astype(np.float32)


# ── 2.2 estimate_z ───────────────────────────────────────────────────

def test_format_prompt_sums_to_one():
    p = format_prompt("walk", "dance", 0.09)
    assert p == "[walk:0.91][dance:0.09]"

def test_estimate_z_picks_dominant_verbs_and_axis():
    an = _StubAnalyzer({"walk": 0.8, "run": 0.2, "stand": 0.1,
                        "dance": 0.4, "spin": 0.05, "jump": 0.1})
    est = estimate_z(_synthetic(), analyzer=an)
    assert est.base == "walk" and est.style == "dance"
    # z = 0.4 / (0.4 + 0.8) = 0.333...
    assert abs(est.z - (0.4 / 1.2)) < 1e-6
    assert abs(sum(est.label.values()) - 1.0) < 1e-9

def test_estimate_z_monotonic_in_style():
    lo = estimate_z(_synthetic(), analyzer=_StubAnalyzer(
        {"walk": 0.9, "run": 0.1, "stand": 0.1, "dance": 0.1, "spin": 0.0, "jump": 0.0}))
    hi = estimate_z(_synthetic(), analyzer=_StubAnalyzer(
        {"walk": 0.9, "run": 0.1, "stand": 0.1, "dance": 0.8, "spin": 0.0, "jump": 0.0}))
    assert hi.z > lo.z

def test_gamma_warps_toward_base():
    an = _StubAnalyzer({"walk": 0.5, "run": 0.1, "stand": 0.1,
                        "dance": 0.5, "spin": 0.0, "jump": 0.0})
    plain = estimate_z(_synthetic(), analyzer=an, gamma=1.0)
    warped = estimate_z(_synthetic(), analyzer=an, gamma=2.0)
    assert warped.z < plain.z          # gamma>1 pulls z down toward the base verb

def test_estimate_prompt_from_array_real_analyzer():
    # the real analyzer must run on raw joints and return a valid two-term prompt
    prompt = estimate_prompt(_synthetic(T=80))
    assert prompt.startswith("[") and prompt.count("[") == 2


# ── 2.3 calibrate ────────────────────────────────────────────────────

def test_mode_grid_and_snap():
    g = mode_grid(10)
    assert len(g) == 10 and abs(g[-1] - 1.0) < 1e-9
    assert snap_to_grid(0.5, g) in g

def test_bin_z_range():
    assert bin_z(0.0, 10) == 0
    assert bin_z(0.999, 10) == 9

def test_load_feedback_rendered_only():
    rows = load_feedback(CSV)
    assert rows and all(r["allocation"] in (20, 40, 60) for r in rows)

def test_calibrate_alpha_per_genre():
    cal = calibrate_alpha(CSV, objective="balanced")
    assert set(cal) == {"ballet", "hip_hop", "jazz"}
    for c in cal.values():
        assert 0.0 <= c.alpha <= 1.0
        assert c.allocation in (20, 40, 60)

def test_expressivity_objective_respects_identity_floor():
    cal = calibrate_alpha(CSV, objective="expressivity", identity_floor=6.0)
    for c in cal.values():
        # chosen allocation must clear the floor whenever any allocation does
        curve = c.per_allocation
        if any(v["identity"] >= 6.0 for v in curve.values()):
            assert c.identity >= 6.0


# ── 2.4 evaluate ─────────────────────────────────────────────────────

def test_identity_one_for_identical():
    m = _synthetic(seed=1)
    assert ev.identity_preservation(m, m) > 0.999
    assert ev.style_strength(m, m) < 1e-3

def test_identity_drops_and_style_rises_with_change():
    a = _synthetic(seed=2)
    b = a + 0.3 * np.random.default_rng(3).standard_normal(a.shape).astype(np.float32)
    assert ev.identity_preservation(a, b) < ev.identity_preservation(a, a)
    assert ev.style_strength(a, b) > ev.style_strength(a, a)

def test_smoothness_bounded():
    s = ev.smoothness(_synthetic(seed=4))
    assert 0.0 < s <= 1.0

def test_generalizability_report_structure():
    rng = np.random.default_rng(5)
    records = []
    for genre in ("ballet", "jazz"):
        for verb in ("walk", "wave"):
            for perf in ("p1", "p2"):
                a = _synthetic(seed=rng.integers(0, 999))
                b = a + 0.1 * rng.standard_normal(a.shape).astype(np.float32)
                records.append({"original": a, "output": b,
                                "genre": genre, "verb": verb, "performer": perf})
    metrics = ev.evaluate_matrix(records)
    rep = ev.generalizability_report(metrics)
    assert rep["n_pairs"] == len(records)
    assert set(rep["by_genre"]) == {"ballet", "jazz"}
    for v in rep["cross_performer_consistency"].values():
        assert 0.0 <= v <= 1.0
    assert isinstance(ev.format_report(rep), str)
