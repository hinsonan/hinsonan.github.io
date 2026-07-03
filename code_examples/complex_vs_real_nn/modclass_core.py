"""Backward-compatible API surface for the AMC experiment modules."""

from config import ModClassConfig
from data import (
    CONSTELLATIONS,
    add_awgn,
    constellation,
    generate_burst,
    generate_clean_burst,
    generate_dataset,
    rotate_burst,
)
from models import (
    ComplexConv1d,
    ComplexModClassifier,
    ComplexMomentClassifier,
    RealModClassifier,
    build_model,
    count_parameters,
    modReLU,
)

__all__ = [
    "ModClassConfig",
    "CONSTELLATIONS",
    "constellation",
    "generate_clean_burst",
    "rotate_burst",
    "add_awgn",
    "generate_burst",
    "generate_dataset",
    "ComplexConv1d",
    "modReLU",
    "ComplexModClassifier",
    "ComplexMomentClassifier",
    "RealModClassifier",
    "count_parameters",
    "build_model",
]
