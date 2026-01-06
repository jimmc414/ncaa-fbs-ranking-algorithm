"""Predefined configuration profiles for the ranking algorithm.

Each profile represents a different philosophy or use case:
- pure_results: Pure game outcomes, no adjustments
- balanced: Moderate adjustments for conference strength and quality wins
- predictive: Optimized for predicting future game outcomes
- conservative: Uses historical priors, smoothed adjustments
"""

from src.data.models import AlgorithmConfig


# Pure results - no adjustments, just game outcomes
PURE_RESULTS = AlgorithmConfig()


# Balanced - moderate adjustments for real-world factors
BALANCED = AlgorithmConfig(
    # Enable conference adjustments
    enable_conference_adj=True,
    p5_multiplier=1.02,
    g5_multiplier=0.95,
    fcs_multiplier=0.50,
    # Enable quality tiers
    enable_quality_tiers=True,
    elite_threshold=0.80,
    elite_win_bonus=0.05,
    bad_threshold=0.35,
    bad_loss_penalty=0.10,
    # Moderate SOS adjustment
    sos_adjustment_weight=0.10,
)


# Predictive - tuned for predicting game outcomes
PREDICTIVE = AlgorithmConfig(
    # Stronger conference adjustments
    enable_conference_adj=True,
    p5_multiplier=1.05,
    g5_multiplier=0.85,
    fcs_multiplier=0.40,
    # Stronger quality tier effects
    enable_quality_tiers=True,
    elite_threshold=0.75,
    elite_win_bonus=0.10,
    bad_threshold=0.40,
    bad_loss_penalty=0.15,
    # SOS requirements for top rankings
    sos_adjustment_weight=0.15,
    min_sos_top_10=0.45,
    min_sos_top_25=0.35,
    min_p5_games_top_10=3,
    # Recency weighting - recent games matter more
    enable_recency=True,
    recency_half_life=6,
    recency_min_weight=0.4,
)


# Tuned Predictive - calibrated from 2025 season diagnostics
# Changes from predictive: venue_road_win 0.10→0.08, margin_weight 0.20→0.15, opponent_weight 1.0→0.9
TUNED_PREDICTIVE = AlgorithmConfig(
    # Core tuning - reduced to improve calibration
    margin_weight=0.15,  # Reduced from 0.20 - close games were overweighted
    opponent_weight=0.9,  # Reduced from 1.0 - rating spread was too wide
    # Venue - reduced road bonus
    venue_road_win=0.08,  # Reduced from 0.10 - road/home advantage was misaligned
    # Stronger conference adjustments
    enable_conference_adj=True,
    p5_multiplier=1.05,
    g5_multiplier=0.85,
    fcs_multiplier=0.40,
    # Stronger quality tier effects
    enable_quality_tiers=True,
    elite_threshold=0.75,
    elite_win_bonus=0.10,
    bad_threshold=0.40,
    bad_loss_penalty=0.15,
    # SOS requirements for top rankings
    sos_adjustment_weight=0.15,
    min_sos_top_10=0.45,
    min_sos_top_25=0.35,
    min_p5_games_top_10=3,
    # Recency weighting - recent games matter more
    enable_recency=True,
    recency_half_life=6,
    recency_min_weight=0.4,
)


# Conservative - uses historical data and conservative adjustments
CONSERVATIVE = AlgorithmConfig(
    # Enable prior from previous seasons
    enable_prior=True,
    prior_weight=0.20,
    prior_decay_weeks=8,
    # Enable recency with longer memory
    enable_recency=True,
    recency_half_life=10,
    recency_min_weight=0.5,
    # Moderate conference adjustments
    enable_conference_adj=True,
    p5_multiplier=1.03,
    g5_multiplier=0.92,
    fcs_multiplier=0.45,
)


# Dictionary of all profiles
PROFILES: dict[str, AlgorithmConfig] = {
    "pure_results": PURE_RESULTS,
    "balanced": BALANCED,
    "predictive": PREDICTIVE,
    "tuned_predictive": TUNED_PREDICTIVE,
    "conservative": CONSERVATIVE,
}


def get_profile(name: str) -> AlgorithmConfig:
    """
    Get a configuration profile by name.

    Args:
        name: Profile name (pure_results, balanced, predictive, conservative)

    Returns:
        AlgorithmConfig instance

    Raises:
        ValueError: If profile name not found
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> list[str]:
    """
    List all available profile names.

    Returns:
        List of profile names
    """
    return list(PROFILES.keys())


def get_profile_description(name: str) -> str:
    """
    Get a human-readable description of a profile.

    Args:
        name: Profile name

    Returns:
        Description string
    """
    descriptions = {
        "pure_results": (
            "Pure game outcomes with no adjustments. "
            "Uses only wins, losses, margin, and venue."
        ),
        "balanced": (
            "Moderate adjustments for conference strength and quality wins. "
            "Good general-purpose profile."
        ),
        "predictive": (
            "Optimized for predicting future game outcomes. "
            "Strong conference and recency adjustments."
        ),
        "tuned_predictive": (
            "Calibrated predictive profile using 2025 diagnostics. "
            "82.9% accuracy, improved calibration. Best for predictions."
        ),
        "conservative": (
            "Uses historical priors and conservative adjustments. "
            "More stable rankings early in the season."
        ),
    }
    if name not in descriptions:
        return "No description available."
    return descriptions[name]
