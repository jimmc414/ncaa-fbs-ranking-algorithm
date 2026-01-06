"""Game grade calculation for NCAA ranking algorithm.

The game grade is a numerical representation of how well a team performed
in a single game, considering:
- Win/loss result
- Margin of victory (logarithmic, capped)
- Venue (road, neutral, home)
- Recency weighting (optional)
- Conference adjustments (optional)
"""

import math
from typing import Literal

from src.data.models import AlgorithmConfig, GameResult

# Default configuration (module-level for convenience)
_DEFAULT_CONFIG = AlgorithmConfig()

# Conference tier classifications
P5_CONFERENCES = {
    "SEC",
    "Big Ten",
    "Big 12",
    "ACC",
    "Pac-12",
    "FBS Independents",
}

G5_CONFERENCES = {
    "American Athletic",
    "Mountain West",
    "Sun Belt",
    "MAC",
    "Mid-American",
    "Conference USA",
}


def compute_margin_bonus(margin: int, is_win: bool) -> float:
    """
    Calculate the logarithmic margin bonus.

    The bonus rewards larger margins of victory but with diminishing returns.
    The first touchdown of margin is worth more than the third.
    Margin is capped at 28 points to avoid incentivizing running up scores.

    Args:
        margin: Absolute point differential (always positive)
        is_win: Whether the team won

    Returns:
        Margin bonus between 0.0 and 0.20
    """
    if not is_win:
        return 0.0

    if margin <= 0:
        return 0.0

    # Cap margin at 28 points
    capped_margin = min(abs(margin), 28)

    # Logarithmic bonus: 0.20 * ln(1 + margin) / ln(29)
    # At margin=28: ln(29)/ln(29) = 1.0, so bonus = 0.20
    return 0.20 * math.log(1 + capped_margin) / math.log(29)


def compute_venue_adjustment(
    location: Literal["home", "away", "neutral"],
    is_win: bool,
    config: AlgorithmConfig | None = None,
) -> float:
    """
    Calculate the venue adjustment.

    Road wins are rewarded, home losses are penalized.

    Args:
        location: Game location from team's perspective
        is_win: Whether the team won
        config: Optional config for custom venue values

    Returns:
        Venue adjustment (positive for bonus, negative for penalty)
    """
    if config is None:
        config = _DEFAULT_CONFIG

    if is_win:
        return {
            "away": config.venue_road_win,
            "neutral": config.venue_neutral_win,
            "home": 0.0,
        }[location]
    else:
        return {
            "away": 0.0,
            "neutral": config.venue_neutral_loss,
            "home": config.venue_home_loss,
        }[location]


def compute_game_grade(
    is_win: bool,
    margin: int,
    location: Literal["home", "away", "neutral"],
    config: AlgorithmConfig | None = None,
) -> float:
    """
    Calculate the complete game grade.

    GameGrade = ResultPoints + MarginBonus + VenueBonus

    Where:
        ResultPoints = 0.70 if win, 0.00 if loss
        MarginBonus = logarithmic bonus (0 to 0.20), wins only
        VenueBonus = venue adjustment (-0.03 to +0.10)

    Args:
        is_win: Whether the team won
        margin: Absolute point differential
        location: Game location from team's perspective
        config: Optional config for custom weights

    Returns:
        Game grade (typically -0.03 to 1.00)
    """
    if config is None:
        config = _DEFAULT_CONFIG

    # Result points: 0.70 for win, 0.00 for loss
    result_points = config.win_base if is_win else 0.0

    # Margin bonus (only for wins)
    margin_bonus = compute_margin_bonus(margin, is_win)
    if config.margin_weight != 0.20:
        # Scale margin bonus if using custom weight
        margin_bonus = margin_bonus * (config.margin_weight / 0.20)

    # Venue adjustment
    venue_bonus = compute_venue_adjustment(location, is_win, config)

    return result_points + margin_bonus + venue_bonus


def compute_game_grade_for_result(
    result: GameResult,
    config: AlgorithmConfig | None = None,
) -> float:
    """
    Calculate game grade from a GameResult object.

    Convenience wrapper that extracts the needed fields from GameResult.

    Args:
        result: GameResult object
        config: Optional config for custom weights

    Returns:
        Game grade
    """
    return compute_game_grade(
        is_win=result.is_win,
        margin=abs(result.margin),
        location=result.location,
        config=config,
    )


def compute_margin_bonus_curved(
    margin: int,
    is_win: bool,
    config: AlgorithmConfig | None = None,
) -> float:
    """
    Calculate margin bonus using configurable curve type.

    Supports three curve types:
    - log: Logarithmic (default, diminishing returns)
    - linear: Proportional to margin
    - sqrt: Square root (between log and linear)

    Args:
        margin: Absolute point differential (always positive)
        is_win: Whether the team won
        config: Config with margin_curve setting

    Returns:
        Margin bonus between 0.0 and margin_weight
    """
    if not is_win:
        return 0.0

    if margin <= 0:
        return 0.0

    if config is None:
        config = _DEFAULT_CONFIG

    # Cap margin at configured max
    cap = config.margin_cap
    capped_margin = min(abs(margin), cap)

    # Calculate bonus based on curve type
    curve = config.margin_curve
    weight = config.margin_weight

    if curve == "linear":
        # Linear: margin/cap * weight
        bonus = (capped_margin / cap) * weight
    elif curve == "sqrt":
        # Square root: sqrt(margin/cap) * weight
        bonus = math.sqrt(capped_margin / cap) * weight
    else:  # "log" (default)
        # Logarithmic: weight * ln(1 + margin) / ln(1 + cap)
        bonus = weight * math.log(1 + capped_margin) / math.log(1 + cap)

    return bonus


def compute_recency_weight(
    weeks_ago: int,
    current_week: int,
    config: AlgorithmConfig | None = None,
) -> float:
    """
    Calculate recency weight for a game.

    Uses exponential decay with half-life to weight recent games more.
    Weight = max(min_weight, 0.5^(weeks_ago / half_life))

    Args:
        weeks_ago: How many weeks ago the game was played
        current_week: Current week of the season
        config: Config with recency settings

    Returns:
        Weight between min_weight and 1.0
    """
    if config is None:
        config = _DEFAULT_CONFIG

    # If recency disabled, all games get full weight
    if not config.enable_recency:
        return 1.0

    # Current week games get full weight
    if weeks_ago <= 0:
        return 1.0

    # Exponential decay: 0.5^(weeks_ago / half_life)
    half_life = config.recency_half_life
    raw_weight = 0.5 ** (weeks_ago / half_life)

    # Apply minimum floor
    return max(config.recency_min_weight, raw_weight)


def compute_conference_multiplier(
    conference: str | None,
    config: AlgorithmConfig | None = None,
) -> float:
    """
    Calculate conference strength multiplier.

    Adjusts opponent quality based on conference tier.

    Args:
        conference: Conference name (e.g., "SEC", "Mountain West")
        config: Config with conference adjustment settings

    Returns:
        Multiplier (1.0 if disabled, or p5/g5/fcs multiplier)
    """
    if config is None:
        config = _DEFAULT_CONFIG

    # If conference adjustment disabled, return 1.0
    if not config.enable_conference_adj:
        return 1.0

    if conference is None:
        return config.g5_multiplier  # Unknown defaults to G5

    # Check for FCS
    if conference.upper() == "FCS" or "FCS" in conference.upper():
        return config.fcs_multiplier

    # Check for P5
    if conference in P5_CONFERENCES:
        return config.p5_multiplier

    # Check for G5
    if conference in G5_CONFERENCES:
        return config.g5_multiplier

    # Unknown conference defaults to G5
    return config.g5_multiplier
