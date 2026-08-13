"""
Stage 2.4 - test outputs for generalizability across genres, verbs and performers.

Given pairs of (original joints, blended output joints), this module scores each pair on
four kinematic metrics and then aggregates them into a generalizability report: how
consistently the pipeline behaves as genre, verb and performer vary. No perceptual study
and no GPU are needed; everything is computed from the joint arrays the pipeline already
produces.

Metrics per (original, output) pair, all in [0, 1] unless noted:
  identity_preservation  how close the output trajectory stays to the original (1 = identical)
  style_strength         how far the output moved from the original in zone-feature space
  smoothness             inverse jerk of the output (1 = perfectly smooth)
  foot_skate             mean horizontal foot slip during ground contact (lower is better, meters/s)

A run generalizes well when identity_preservation, style_strength and smoothness stay high
and stable (low spread) across every genre, verb and performer, and foot_skate stays low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .zones import ZoneConfig
from .zone_features import ZoneFeatureExtractor, FPS

_J_PELVIS = 0
_FOOT_JOINTS = [7, 8, 10, 11]   # ankles and toes


def _root_align(motion: np.ndarray) -> np.ndarray:
    """Remove global translation by subtracting the pelvis position each frame."""
    return motion - motion[:, _J_PELVIS:_J_PELVIS + 1, :]


def identity_preservation(original: np.ndarray, output: np.ndarray) -> float:
    """
    1 = the output stays exactly on the performer's trajectory, 0 = wholly different.
    Global translation is removed first, so this measures posture and limb identity,
    not where the body walked to. exp(-rmse / scale) maps deviation into [0, 1].
    """
    a, b = _root_align(original), _root_align(output)
    T = min(len(a), len(b))
    a, b = a[:T], b[:T]
    rmse = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))
    scale = float(np.sqrt(np.mean(np.sum(a ** 2, axis=-1)))) + 1e-6   # rms body extent
    return float(np.exp(-rmse / scale))


def style_strength(original: np.ndarray, output: np.ndarray,
                   extractor: Optional[ZoneFeatureExtractor] = None) -> float:
    """
    How much the output moved away from the original in per-zone feature space,
    squashed into [0, 1]. 0 = feature-identical to the original (no style imported).
    """
    extractor = extractor or ZoneFeatureExtractor(ZoneConfig("standard"))
    fo, fb = extractor.extract(original), extractor.extract(output)
    dists = []
    for zone in fo:
        d = np.linalg.norm(fb[zone] - fo[zone]) / (np.linalg.norm(fo[zone]) + 1e-6)
        dists.append(d)
    mean_rel = float(np.mean(dists))
    return float(1.0 - np.exp(-mean_rel))   # 0 when unchanged, -> 1 as it diverges


def smoothness(output: np.ndarray) -> float:
    """Inverse mean jerk (third time derivative). 1 = perfectly smooth."""
    if len(output) < 4:
        return 1.0
    jerk = np.diff(output, n=3, axis=0) * (FPS ** 3)
    mean_jerk = float(np.mean(np.linalg.norm(jerk, axis=-1)))
    return float(1.0 / (1.0 + mean_jerk))


def foot_skate(output: np.ndarray, contact_frac: float = 0.15) -> float:
    """
    Mean horizontal foot speed during ground contact (meters/s, lower is better).
    A foot is "in contact" on frames where its height is within contact_frac of the
    per-foot vertical range. Sliding contacts read as physically implausible motion.
    """
    if len(output) < 2:
        return 0.0
    speeds = []
    for j in _FOOT_JOINTS:
        y = output[:, j, 1]
        lo, hi = float(y.min()), float(y.max())
        thresh = lo + contact_frac * (hi - lo + 1e-9)
        contact = y[:-1] <= thresh
        if not np.any(contact):
            continue
        horiz = np.diff(output[:, j, [0, 2]], axis=0) * FPS      # (T-1, 2)
        speed = np.linalg.norm(horiz, axis=-1)
        speeds.append(float(np.mean(speed[contact])))
    return float(np.mean(speeds)) if speeds else 0.0


@dataclass
class PairMetrics:
    genre: str
    verb: str
    performer: str
    identity_preservation: float
    style_strength: float
    smoothness: float
    foot_skate: float

    def as_dict(self) -> Dict[str, float]:
        return {"identity_preservation": self.identity_preservation,
                "style_strength": self.style_strength,
                "smoothness": self.smoothness,
                "foot_skate": self.foot_skate}


_METRICS = ["identity_preservation", "style_strength", "smoothness", "foot_skate"]


def evaluate_pair(original: np.ndarray, output: np.ndarray,
                  genre: str = "", verb: str = "", performer: str = "",
                  extractor: Optional[ZoneFeatureExtractor] = None) -> PairMetrics:
    """Score one (original, output) pair on all four metrics."""
    extractor = extractor or ZoneFeatureExtractor(ZoneConfig("standard"))
    return PairMetrics(
        genre=genre, verb=verb, performer=performer,
        identity_preservation=round(identity_preservation(original, output), 4),
        style_strength=round(style_strength(original, output, extractor), 4),
        smoothness=round(smoothness(output), 4),
        foot_skate=round(foot_skate(output), 4),
    )


def evaluate_matrix(records: List[dict]) -> List[PairMetrics]:
    """
    Score a list of records against every metric.

    Each record: {"original": (T,22,3), "output": (T,22,3),
                  "genre": str, "verb": str, "performer": str}.
    """
    extractor = ZoneFeatureExtractor(ZoneConfig("standard"))
    out = []
    for r in records:
        out.append(evaluate_pair(
            r["original"], r["output"],
            genre=r.get("genre", ""), verb=r.get("verb", ""), performer=r.get("performer", ""),
            extractor=extractor,
        ))
    return out


def _group_stats(metrics: List[PairMetrics], key: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each value of `key` (genre/verb/performer), mean and std of every metric."""
    groups: Dict[str, List[PairMetrics]] = {}
    for m in metrics:
        groups.setdefault(getattr(m, key), []).append(m)
    out = {}
    for val, ms in groups.items():
        out[val] = {}
        for name in _METRICS:
            vals = np.array([getattr(m, name) for m in ms], dtype=float)
            out[val][name] = {"mean": round(float(vals.mean()), 4),
                              "std": round(float(vals.std()), 4), "n": len(ms)}
    return out


def generalizability_report(metrics: List[PairMetrics]) -> dict:
    """
    Aggregate per-pair metrics into a generalizability summary.

    Returns overall mean/std per metric, per-genre / per-verb / per-performer breakdowns,
    and a cross-performer consistency score per metric (1 - normalised spread across
    performers; higher means the metric holds steady as the performer changes).
    """
    if not metrics:
        raise ValueError("No metrics to report")

    overall = {}
    for name in _METRICS:
        vals = np.array([getattr(m, name) for m in metrics], dtype=float)
        overall[name] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}

    by_performer = _group_stats(metrics, "performer")
    consistency = {}
    for name in _METRICS:
        means = np.array([by_performer[p][name]["mean"] for p in by_performer], dtype=float)
        spread = float(means.std())
        denom = float(abs(means.mean())) + 1e-6
        consistency[name] = round(float(max(0.0, 1.0 - spread / denom)), 4)

    return {
        "n_pairs": len(metrics),
        "overall": overall,
        "by_genre": _group_stats(metrics, "genre"),
        "by_verb": _group_stats(metrics, "verb"),
        "by_performer": by_performer,
        "cross_performer_consistency": consistency,
    }


def format_report(report: dict) -> str:
    """Render a generalizability report as a compact text table."""
    lines = [f"Generalizability report  ({report['n_pairs']} pairs)", ""]
    lines.append(f"{'metric':24s}{'mean':>8s}{'std':>8s}{'consistency':>14s}")
    for name in _METRICS:
        o = report["overall"][name]
        c = report["cross_performer_consistency"][name]
        lines.append(f"{name:24s}{o['mean']:>8.3f}{o['std']:>8.3f}{c:>14.3f}")
    for axis in ("by_genre", "by_verb"):
        lines.append("")
        lines.append(axis.replace("by_", "by ") + ":")
        for val, stats in sorted(report[axis].items()):
            ident = stats["identity_preservation"]["mean"]
            style = stats["style_strength"]["mean"]
            lines.append(f"  {val:16s} identity={ident:.3f}  style={style:.3f}  n={stats['identity_preservation']['n']}")
    return "\n".join(lines)
