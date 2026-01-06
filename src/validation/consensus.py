"""Consensus prediction logic for blending multiple prediction sources."""

import math
from typing import Literal

from src.validation.models import (
    ConfidenceBreakdown,
    GamePrediction,
    SourcePrediction,
    WeekPredictions,
)


# Default weights for consensus calculation
# Vegas highest as it's historically most accurate
# ALL factors shown in output regardless of weight
CONSENSUS_WEIGHTS = {
    "vegas": 0.35,  # Vegas betting lines
    "our_algorithm": 0.25,  # Our core ranking
    "pregame_wp": 0.20,  # Pre-game win probability from markets
    "sp_implied": 0.10,  # SP+ implied probability
    "elo_implied": 0.10,  # Elo implied probability
}


def spread_to_probability(spread: float, home_advantage: bool = True) -> float:
    """
    Convert Vegas point spread to win probability.

    Based on historical data calibration:
    - -3 points = ~58% win probability
    - -7 points = ~70% win probability
    - -14 points = ~85% win probability
    - -21 points = ~92% win probability

    Args:
        spread: Point spread (negative = favored)
        home_advantage: Whether the favored team is at home

    Returns:
        Win probability for the favored team (0.0 to 1.0)
    """
    # Steepness factor calibrated to historical spread data
    # This maps spread to probability using logistic function
    k = 0.148  # Calibrated so -7 spread ≈ 70%

    # The spread is from home team perspective
    # Negative spread = home favored, positive = away favored
    prob = 1.0 / (1.0 + math.exp(k * spread))

    return prob


def elo_to_win_probability(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 55.0,
) -> float:
    """
    Convert Elo ratings to home team win probability.

    Args:
        home_elo: Home team Elo rating
        away_elo: Away team Elo rating
        home_advantage: Elo points for home field advantage (default 55)

    Returns:
        Home team win probability (0.0 to 1.0)
    """
    # Standard Elo probability formula with home advantage
    elo_diff = home_elo + home_advantage - away_elo
    prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
    return prob


def rating_gap_to_probability(
    rating_gap: float,
    home_advantage: float = 0.03,
) -> float:
    """
    Convert our rating gap to win probability.

    Args:
        rating_gap: Home team rating - Away team rating
        home_advantage: Home field advantage in rating points

    Returns:
        Home team win probability (0.0 to 1.0)
    """
    # Use logistic function, same as in diagnostics
    k = 10.0
    adjusted_gap = rating_gap + home_advantage
    prob = 1.0 / (1.0 + math.exp(-k * adjusted_gap))
    return prob


def calculate_consensus(
    predictions: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Calculate weighted consensus from multiple prediction sources.

    Args:
        predictions: Dict mapping source -> home win probability
        weights: Optional custom weights (uses CONSENSUS_WEIGHTS if None)

    Returns:
        Tuple of (consensus_probability, contributions_dict)
        contributions_dict shows how much each source contributed
    """
    if weights is None:
        weights = CONSENSUS_WEIGHTS

    # Filter to available predictions
    available = {k: v for k, v in predictions.items() if v is not None}

    if not available:
        return 0.5, {}

    # Calculate total weight of available sources
    total_weight = sum(weights.get(source, 0) for source in available)

    if total_weight == 0:
        # Equal weight fallback
        total_weight = len(available)
        rebalanced_weights = {source: 1.0 for source in available}
    else:
        # Rebalance weights to sum to 1.0
        rebalanced_weights = {
            source: weights.get(source, 0) / total_weight for source in available
        }

    # Calculate weighted consensus
    consensus = 0.0
    contributions = {}
    for source, prob in available.items():
        weight = rebalanced_weights.get(source, 0)
        contribution = prob * weight
        consensus += contribution
        contributions[source] = contribution

    return consensus, contributions


def assess_confidence(
    predictions: dict[str, float],
) -> ConfidenceBreakdown:
    """
    Assess prediction confidence based on source agreement.

    Confidence levels:
    - HIGH: All sources agree on winner, spread < 10%
    - MODERATE: All sources agree, spread < 20%
    - LOW: All sources agree, spread > 20%
    - SPLIT: Sources disagree on winner
    - UNKNOWN: Not enough data

    Args:
        predictions: Dict mapping source -> home win probability

    Returns:
        ConfidenceBreakdown with level and explanation
    """
    # Filter to available predictions
    available = {k: v for k, v in predictions.items() if v is not None}
    sources_available = len(available)

    if sources_available == 0:
        return ConfidenceBreakdown(
            level="UNKNOWN",
            sources_agreeing=0,
            sources_available=0,
            probability_spread=0.0,
            reason="No prediction sources available",
        )

    probs = list(available.values())
    prob_spread = max(probs) - min(probs)

    # Count sources agreeing on winner (prob > 0.5 = home, < 0.5 = away)
    favoring_home = sum(1 for p in probs if p > 0.5)
    favoring_away = sources_available - favoring_home

    # Check for tie (exactly 0.5)
    ties = sum(1 for p in probs if p == 0.5)

    if ties == sources_available:
        return ConfidenceBreakdown(
            level="SPLIT",
            sources_agreeing=0,
            sources_available=sources_available,
            probability_spread=prob_spread,
            reason="All sources predict a toss-up",
        )

    # Determine majority and agreement
    if favoring_home > favoring_away:
        majority_count = favoring_home
        minority_count = favoring_away
    else:
        majority_count = favoring_away
        minority_count = favoring_home

    sources_agreeing = majority_count

    # If any meaningful disagreement on winner
    if minority_count > 0:
        return ConfidenceBreakdown(
            level="SPLIT",
            sources_agreeing=sources_agreeing,
            sources_available=sources_available,
            probability_spread=prob_spread,
            reason=f"Sources disagree: {majority_count}/{sources_available} favor one team",
        )

    # All agree on winner - assess confidence based on probability spread
    if prob_spread < 0.10:
        level: Literal["HIGH", "MODERATE", "LOW", "SPLIT", "UNKNOWN"] = "HIGH"
        reason = f"All {sources_available} sources agree, tight spread ({prob_spread:.1%})"
    elif prob_spread < 0.20:
        level = "MODERATE"
        reason = f"All {sources_available} sources agree, moderate spread ({prob_spread:.1%})"
    else:
        level = "LOW"
        reason = f"All {sources_available} sources agree, but wide spread ({prob_spread:.1%})"

    return ConfidenceBreakdown(
        level=level,
        sources_agreeing=sources_agreeing,
        sources_available=sources_available,
        probability_spread=prob_spread,
        reason=reason,
    )


def build_game_prediction(
    home_team_id: str,
    away_team_id: str,
    home_rating: float | None,
    away_rating: float | None,
    vegas_spread: float | None = None,
    pregame_wp: float | None = None,
    home_sp_rating: float | None = None,
    away_sp_rating: float | None = None,
    home_elo: float | None = None,
    away_elo: float | None = None,
    game_id: int | None = None,
    season: int = 0,
    week: int | None = None,
    neutral_site: bool = False,
) -> GamePrediction:
    """
    Build a comprehensive game prediction from all available sources.

    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        home_rating: Our home team rating (0-1 scale)
        away_rating: Our away team rating (0-1 scale)
        vegas_spread: Vegas spread (negative = home favored)
        pregame_wp: Market-derived home win probability
        home_sp_rating: Home team SP+ rating
        away_sp_rating: Away team SP+ rating
        home_elo: Home team Elo rating
        away_elo: Away team Elo rating
        game_id: Game ID
        season: Season year
        week: Week number
        neutral_site: Whether game is at neutral site

    Returns:
        GamePrediction with all sources and consensus
    """
    source_predictions = []
    home_probs: dict[str, float] = {}

    home_advantage = 0.0 if neutral_site else 0.03

    # Our algorithm prediction
    if home_rating is not None and away_rating is not None:
        our_prob = rating_gap_to_probability(
            home_rating - away_rating,
            home_advantage=home_advantage,
        )
        home_probs["our_algorithm"] = our_prob
        source_predictions.append(
            SourcePrediction(
                source="our_algorithm",
                predicted_winner=home_team_id if our_prob > 0.5 else away_team_id,
                win_probability=our_prob if our_prob > 0.5 else 1 - our_prob,
                weight=CONSENSUS_WEIGHTS["our_algorithm"],
                available=True,
            )
        )
    else:
        source_predictions.append(
            SourcePrediction(
                source="our_algorithm",
                predicted_winner=None,
                win_probability=None,
                weight=CONSENSUS_WEIGHTS["our_algorithm"],
                available=False,
            )
        )

    # Vegas spread prediction
    if vegas_spread is not None:
        vegas_prob = spread_to_probability(vegas_spread)
        home_probs["vegas"] = vegas_prob
        source_predictions.append(
            SourcePrediction(
                source="vegas",
                predicted_winner=home_team_id if vegas_prob > 0.5 else away_team_id,
                win_probability=vegas_prob if vegas_prob > 0.5 else 1 - vegas_prob,
                spread=vegas_spread,
                weight=CONSENSUS_WEIGHTS["vegas"],
                available=True,
            )
        )
    else:
        source_predictions.append(
            SourcePrediction(
                source="vegas",
                predicted_winner=None,
                win_probability=None,
                weight=CONSENSUS_WEIGHTS["vegas"],
                available=False,
            )
        )

    # Pre-game win probability (already a probability for home team)
    if pregame_wp is not None:
        home_probs["pregame_wp"] = pregame_wp
        source_predictions.append(
            SourcePrediction(
                source="pregame_wp",
                predicted_winner=home_team_id if pregame_wp > 0.5 else away_team_id,
                win_probability=pregame_wp if pregame_wp > 0.5 else 1 - pregame_wp,
                weight=CONSENSUS_WEIGHTS["pregame_wp"],
                available=True,
            )
        )
    else:
        source_predictions.append(
            SourcePrediction(
                source="pregame_wp",
                predicted_winner=None,
                win_probability=None,
                weight=CONSENSUS_WEIGHTS["pregame_wp"],
                available=False,
            )
        )

    # SP+ implied prediction
    if home_sp_rating is not None and away_sp_rating is not None:
        # SP+ rating is a point-based rating, convert gap to probability
        sp_gap = home_sp_rating - away_sp_rating
        sp_home_advantage = 0 if neutral_site else 2.5  # SP+ uses ~2.5 point home advantage
        sp_adj_gap = sp_gap + sp_home_advantage
        # Use similar logistic as spread
        sp_prob = 1.0 / (1.0 + math.exp(-0.10 * sp_adj_gap))
        home_probs["sp_implied"] = sp_prob
        source_predictions.append(
            SourcePrediction(
                source="sp_implied",
                predicted_winner=home_team_id if sp_prob > 0.5 else away_team_id,
                win_probability=sp_prob if sp_prob > 0.5 else 1 - sp_prob,
                weight=CONSENSUS_WEIGHTS["sp_implied"],
                available=True,
            )
        )
    else:
        source_predictions.append(
            SourcePrediction(
                source="sp_implied",
                predicted_winner=None,
                win_probability=None,
                weight=CONSENSUS_WEIGHTS["sp_implied"],
                available=False,
            )
        )

    # Elo implied prediction
    if home_elo is not None and away_elo is not None:
        elo_home_adv = 0.0 if neutral_site else 55.0
        elo_prob = elo_to_win_probability(home_elo, away_elo, home_advantage=elo_home_adv)
        home_probs["elo_implied"] = elo_prob
        source_predictions.append(
            SourcePrediction(
                source="elo_implied",
                predicted_winner=home_team_id if elo_prob > 0.5 else away_team_id,
                win_probability=elo_prob if elo_prob > 0.5 else 1 - elo_prob,
                weight=CONSENSUS_WEIGHTS["elo_implied"],
                available=True,
            )
        )
    else:
        source_predictions.append(
            SourcePrediction(
                source="elo_implied",
                predicted_winner=None,
                win_probability=None,
                weight=CONSENSUS_WEIGHTS["elo_implied"],
                available=False,
            )
        )

    # Calculate consensus
    consensus_home_prob, _ = calculate_consensus(home_probs)
    confidence_breakdown = assess_confidence(home_probs)

    # Determine consensus winner
    if consensus_home_prob > 0.5:
        consensus_winner = home_team_id
        consensus_prob = consensus_home_prob
    else:
        consensus_winner = away_team_id
        consensus_prob = 1 - consensus_home_prob

    return GamePrediction(
        game_id=game_id,
        season=season,
        week=week,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        source_predictions=source_predictions,
        consensus_winner=consensus_winner,
        consensus_prob=consensus_prob,
        confidence=confidence_breakdown.level,
        sources_agreeing=confidence_breakdown.sources_agreeing,
        sources_available=confidence_breakdown.sources_available,
        home_rating=home_rating,
        away_rating=away_rating,
        rating_gap=home_rating - away_rating if home_rating and away_rating else None,
    )


def organize_by_confidence(predictions: list[GamePrediction]) -> WeekPredictions:
    """
    Organize predictions by confidence level.

    Args:
        predictions: List of GamePrediction objects

    Returns:
        WeekPredictions with games sorted into confidence buckets
    """
    if not predictions:
        return WeekPredictions(season=0, week=0)

    result = WeekPredictions(
        season=predictions[0].season,
        week=predictions[0].week or 0,
    )

    for pred in predictions:
        if pred.confidence == "HIGH":
            result.high_confidence.append(pred)
        elif pred.confidence == "MODERATE":
            result.moderate_confidence.append(pred)
        elif pred.confidence == "LOW":
            result.low_confidence.append(pred)
        else:
            result.split.append(pred)

    # Sort each bucket by consensus probability (most confident first)
    for bucket in [
        result.high_confidence,
        result.moderate_confidence,
        result.low_confidence,
        result.split,
    ]:
        bucket.sort(key=lambda p: p.consensus_prob or 0, reverse=True)

    # Calculate accuracy if games have results
    completed = [p for p in predictions if p.was_correct is not None]
    if completed:
        result.total_predictions = len(completed)
        result.correct_predictions = sum(1 for p in completed if p.was_correct)
        result.accuracy = result.correct_predictions / result.total_predictions

    return result
