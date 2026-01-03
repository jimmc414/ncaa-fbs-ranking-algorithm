"""Game grade calculation for NCAA ranking algorithm.

The game grade is a numerical representation of how well a team performed
in a single game, considering:
- Win/loss result
- Margin of victory (logarithmic, capped)
- Venue (road, neutral, home)
"""

import math
from typing import Literal

from src.data.models import AlgorithmConfig, GameResult

# Default configuration (module-level for convenience)
_DEFAULT_CONFIG = AlgorithmConfig()


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
