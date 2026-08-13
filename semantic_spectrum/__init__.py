from .analyzer import SpectrumAnalyzer
from .synthesizer import SpectrumCaptionSynthesizer
from .augment import augment_dataset

# Phase 2 — zone-aware blending
from .zones import ZoneConfig
from .zone_features import ZoneFeatureExtractor
from .blend import FeatureBlender

# ZoneBlendPipeline pulls in torch / MoMask. Guard it so the pure-kinematics modules
# (analyzer, zone_features, blend, estimate_z, calibrate, evaluate) stay importable on
# machines without a working torch/CUDA build.
try:
    from .pipeline import ZoneBlendPipeline
except Exception:   # pragma: no cover - optional heavy dependency
    ZoneBlendPipeline = None
