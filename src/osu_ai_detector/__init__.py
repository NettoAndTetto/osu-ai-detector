"""Explainable osu! beatmap AI-fingerprint detection."""

from .detector import (
    DEFAULT_CONTENT_MODEL,
    DEFAULT_FORENSIC_MODEL,
    DEFAULT_FUSION_MODEL,
    DEFAULT_REVISION_REGISTRY,
    DetectionReport,
    Detector,
    DetectorConfig,
    Verdict,
)
from .forensic import ForensicEnsemble
from .fusion import CheapFusionModel
from .parser import Beatmap, parse_beatmap
from .revision_registry import load_revision_registry, match_revision
from .whitebox import WhiteboxCheckpoint, WhiteboxEngine, WhiteboxOptions, score_whitebox
from .whitebox_model import (
    DEFAULT_MODEL as DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL,
    WhiteboxDiscriminator,
    candidate_map_key,
    flatten_whitebox_result,
    score_whitebox_discriminator,
)

__all__ = [
    "Beatmap",
    "DEFAULT_CONTENT_MODEL",
    "DEFAULT_FORENSIC_MODEL",
    "DEFAULT_FUSION_MODEL",
    "DEFAULT_REVISION_REGISTRY",
    "DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL",
    "DetectionReport",
    "Detector",
    "DetectorConfig",
    "ForensicEnsemble",
    "CheapFusionModel",
    "Verdict",
    "WhiteboxCheckpoint",
    "WhiteboxEngine",
    "WhiteboxOptions",
    "WhiteboxDiscriminator",
    "candidate_map_key",
    "flatten_whitebox_result",
    "load_revision_registry",
    "match_revision",
    "parse_beatmap",
    "score_whitebox",
    "score_whitebox_discriminator",
]
__version__ = "1.0.0"
