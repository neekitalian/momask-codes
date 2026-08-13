"""
Stage 2.2 - z estimation and z integration.

Phase 1 conditions the fine-tuned MoMask model on a hand-written spectrum label of
the form  [walk:0.91][dance:0.09]  prepended to a free-form caption. That label (the
"z" coordinate) had to be chosen by hand for every recording. Stage 2.2 estimates it
automatically from the performer's MediaPipe joints, so the conditioning coordinate is
read off the reference motion instead of guessed.

z estimation
  A recording is scored on the six semantic dimensions by the existing
  SpectrumAnalyzer. The verbs are grouped into a base (locomotion) set and a style
  (expressive) set. z is the style fraction on the dominant base<->style axis:

      z = style_score / (style_score + base_score)

  z = 0 is pure base motion (e.g. plain walking), z = 1 is pure style. The label
  keeps the two dominant terms so it sums to 1.00, exactly matching the Phase 1
  prompt format.

z integration
  estimate_prompt() turns a recording (array or .npz) straight into the conditioning
  string that edit_t2m / video_to_spectrum feed to the model, replacing the
  hard-coded [walk:0.91][dance:0.09].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .analyzer import SpectrumAnalyzer

# Verbs the body is fundamentally doing (locomotion / support) ...
BASE_VERBS: List[str] = ["walk", "run", "stand"]
# ... versus the expressive overlay that makes it read as performance.
STYLE_VERBS: List[str] = ["dance", "spin", "jump"]


@dataclass
class ZEstimate:
    """Result of estimating the spectrum coordinate for one recording."""
    scores: Dict[str, float]          # raw per-dimension scores in [0, 1]
    base: str                         # dominant base (locomotion) verb
    style: str                        # dominant style (expressive) verb
    z: float                          # style fraction in [0, 1] on the base<->style axis
    label: Dict[str, float]           # {base: 1-z, style: z}, rounded, summing to 1.00
    prompt: str = field(default="")   # e.g. "[walk:0.91][dance:0.09]"

    def caption(self, text: str) -> str:
        """Prepend the spectrum label to a free-form caption."""
        return f"{self.prompt} {text}".strip()


def _dominant(scores: Dict[str, float], verbs: List[str]) -> str:
    """Verb with the highest score among `verbs` that the analyzer actually produced."""
    present = [v for v in verbs if v in scores]
    if not present:
        raise ValueError(f"None of {verbs} were scored; available: {list(scores)}")
    return max(present, key=lambda v: scores[v])


def format_prompt(base: str, style: str, z: float) -> str:
    """Build the two-term spectrum label, base first, summing to 1.00 after rounding."""
    z = float(np.clip(z, 0.0, 1.0))
    style_w = round(z, 2)
    base_w = round(1.0 - style_w, 2)   # complement of the rounded value so the two sum to 1.00
    return f"[{base}:{base_w:.2f}][{style}:{style_w:.2f}]"


def estimate_z(
    motion: np.ndarray,
    analyzer: Optional[SpectrumAnalyzer] = None,
    base_verbs: Optional[List[str]] = None,
    style_verbs: Optional[List[str]] = None,
    gamma: float = 1.0,
) -> ZEstimate:
    """
    Estimate the spectrum coordinate z from a raw joint recording.

    Parameters
    ----------
    motion : (T, 22, 3) float array, Y-up, meters, 20 fps.
    analyzer : optional SpectrumAnalyzer (a default one is built if omitted).
    base_verbs, style_verbs : override the locomotion / expressive groupings.
    gamma : calibration warp on z, z_out = z**gamma. gamma > 1 pulls toward the base,
            gamma < 1 pushes toward the style. Stage 2.3 fits this from feedback.

    Returns
    -------
    ZEstimate with the raw scores, dominant base and style verbs, z, label and prompt.
    """
    analyzer = analyzer or SpectrumAnalyzer()
    base_verbs = base_verbs or BASE_VERBS
    style_verbs = style_verbs or STYLE_VERBS

    scores = analyzer.analyze(motion)
    base = _dominant(scores, base_verbs)
    style = _dominant(scores, style_verbs)

    b, s = float(scores[base]), float(scores[style])
    z = s / (s + b + 1e-8)
    if gamma != 1.0:
        z = float(z ** gamma)
    z = float(np.clip(z, 0.0, 1.0))

    prompt = format_prompt(base, style, z)
    style_w = round(z, 2)
    label = {base: round(1.0 - style_w, 2), style: style_w}
    return ZEstimate(scores=scores, base=base, style=style, z=z, label=label, prompt=prompt)


# ─────────────────────────────────────────────────────────────────────
# z integration - recording in, conditioning prompt out
# ─────────────────────────────────────────────────────────────────────

def _load_joints(npz_path: str | Path) -> np.ndarray:
    """Load a (T, 22, 3) joint array from an .npz, trying the common key names."""
    data = np.load(npz_path, allow_pickle=True)
    for key in ("joints", "poses", "motion", "pose", "arr_0"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32)
            break
    else:
        raise KeyError(f"No joint array found in {npz_path}; keys: {list(data.keys())}")
    if arr.ndim == 2 and arr.shape[1] == 66:      # (T, 66) flattened -> (T, 22, 3)
        arr = arr.reshape(arr.shape[0], 22, 3)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        raise ValueError(f"Expected (T, 22, 3) joints, got {arr.shape}")
    return arr


def estimate_prompt(
    recording: str | Path | np.ndarray,
    text: str = "",
    analyzer: Optional[SpectrumAnalyzer] = None,
    gamma: float = 1.0,
) -> str:
    """
    Convenience wrapper: recording (array or .npz path) -> full conditioning prompt.

    This is the drop-in replacement for the hard-coded "[walk:0.91][dance:0.09]" in
    video_to_spectrum.py: pass the MediaPipe .npz and, optionally, the caption text.
    """
    motion = recording if isinstance(recording, np.ndarray) else _load_joints(recording)
    est = estimate_z(motion, analyzer=analyzer, gamma=gamma)
    return est.caption(text) if text else est.prompt


if __name__ == "__main__":   # pragma: no cover - thin CLI
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Estimate the spectrum coordinate z from a recording.")
    ap.add_argument("npz", help="Path to a MediaPipe .npz with a (T, 22, 3) joint array.")
    ap.add_argument("--text", default="", help="Caption text to append after the label.")
    ap.add_argument("--gamma", type=float, default=1.0, help="Calibration warp on z (Stage 2.3).")
    args = ap.parse_args()

    joints = _load_joints(args.npz)
    est = estimate_z(joints, gamma=args.gamma)
    print(json.dumps({
        "scores": {k: round(v, 3) for k, v in est.scores.items()},
        "base": est.base, "style": est.style, "z": round(est.z, 3),
        "label": est.label, "prompt": est.caption(args.text) if args.text else est.prompt,
    }, indent=2))
