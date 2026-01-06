"""Validation module for comparing rankings against external validators."""

from src.validation.models import (
    AnomalyFactor,
    GamePrediction,
    PatternReport,
    SourcePrediction,
    TeamValidation,
    UpsetAnalysis,
    UpsetStats,
    ValidationReport,
    ValidatorRating,
    VegasUpset,
)
from src.validation.validators import ValidatorService
from src.validation.consensus import (
    assess_confidence,
    calculate_consensus,
    spread_to_probability,
    CONSENSUS_WEIGHTS,
)
from src.validation.upset_analyzer import UpsetAnalyzer

__all__ = [
    # Models
    "AnomalyFactor",
    "GamePrediction",
    "PatternReport",
    "SourcePrediction",
    "TeamValidation",
    "UpsetAnalysis",
    "UpsetStats",
    "ValidationReport",
    "ValidatorRating",
    "VegasUpset",
    # Services
    "ValidatorService",
    "UpsetAnalyzer",
    # Consensus
    "assess_confidence",
    "calculate_consensus",
    "spread_to_probability",
    "CONSENSUS_WEIGHTS",
]
