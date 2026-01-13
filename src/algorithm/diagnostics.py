"""Diagnostic tools for calibrating and debugging the ranking algorithm.

This module provides:
- Game contribution decomposition (break down each game's impact)
- Win probability calculation from rating gaps
- Prediction tracking and accuracy metrics
- Parameter sensitivity analysis (which lever caused wrong predictions)
"""

import math
from dataclasses import dataclass, field
from typing import Literal

from src.algorithm.convergence import build_game_results, converge, normalize_ratings
from src.algorithm.game_grade import (
    compute_game_grade_for_result,
    compute_margin_bonus_curved,
    compute_venue_adjustment,
)
from src.data.models import AlgorithmConfig, Game, GameResult


@dataclass
class GameContributionBreakdown:
    """Detailed breakdown of a single game's contribution to a team's rating."""

    game_id: int
    team_id: str
    opponent_id: str
    is_win: bool
    margin: int
    location: Literal["home", "away", "neutral"]

    # Component breakdown
    base_result: float  # Win: win_base (0.70), Loss: 0.0
    margin_bonus: float  # 0 to margin_weight (0.20) for wins
    venue_adjustment: float  # ±home_advantage
    opponent_rating: float  # Weighted opponent rating
    opponent_contribution: float  # How opponent rating affected contribution
    quality_tier_adjustment: float  # Elite win bonus or bad loss penalty
    conference_multiplier: float  # P5/G5/FCS adjustment
    recency_weight: float  # Weight based on game timing

    # Final values
    game_grade: float  # base + margin + venue
    total_contribution: float  # Full contribution to rating

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "team_id": self.team_id,
            "opponent_id": self.opponent_id,
            "is_win": self.is_win,
            "margin": self.margin,
            "location": self.location,
            "components": {
                "base_result": self.base_result,
                "margin_bonus": self.margin_bonus,
                "venue_adjustment": self.venue_adjustment,
                "opponent_rating": self.opponent_rating,
                "opponent_contribution": self.opponent_contribution,
                "quality_tier_adjustment": self.quality_tier_adjustment,
                "conference_multiplier": self.conference_multiplier,
                "recency_weight": self.recency_weight,
            },
            "game_grade": self.game_grade,
            "total_contribution": self.total_contribution,
        }


@dataclass
class Prediction:
    """A prediction for a single game."""

    game_id: int
    season: int
    week: int
    home_team_id: str
    away_team_id: str
    home_rating: float
    away_rating: float
    rating_gap: float  # home_rating - away_rating (positive = home favored)
    predicted_winner: str
    win_probability: float  # Probability predicted winner wins
    actual_winner: str | None = None  # None if game not played
    was_correct: bool | None = None
    is_upset: bool | None = None
    upset_magnitude: float | None = None  # How surprising (0-1)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team_id,
            "away_team": self.away_team_id,
            "home_rating": round(self.home_rating, 4),
            "away_rating": round(self.away_rating, 4),
            "rating_gap": round(self.rating_gap, 4),
            "predicted_winner": self.predicted_winner,
            "win_probability": round(self.win_probability, 3),
            "actual_winner": self.actual_winner,
            "was_correct": self.was_correct,
            "is_upset": self.is_upset,
            "upset_magnitude": round(self.upset_magnitude, 3) if self.upset_magnitude else None,
        }


@dataclass
class ParameterAttribution:
    """Attribution of prediction errors to specific parameters."""

    parameter: str
    error_count: int
    pattern_description: str
    affected_games: list[int]  # game_ids
    suggested_adjustment: str
    current_value: float | bool | str
    suggested_value: float | bool | str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "error_count": self.error_count,
            "pattern": self.pattern_description,
            "suggested_adjustment": self.suggested_adjustment,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
        }


@dataclass
class DiagnosticReport:
    """Full diagnostic report for a set of predictions."""

    season: int
    week: int | None
    total_games: int
    correct_predictions: int
    wrong_predictions: int
    accuracy: float
    brier_score: float  # Lower is better (0 = perfect)
    calibration_error: float  # How well probabilities match outcomes
    predictions: list[Prediction]
    upsets: list[Prediction]
    parameter_attributions: list[ParameterAttribution]
    suggestions: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "season": self.season,
            "week": self.week,
            "summary": {
                "total_games": self.total_games,
                "correct": self.correct_predictions,
                "wrong": self.wrong_predictions,
                "accuracy": round(self.accuracy, 4),
                "brier_score": round(self.brier_score, 4),
                "calibration_error": round(self.calibration_error, 4),
            },
            "upsets": [p.to_dict() for p in self.upsets],
            "parameter_attributions": [a.to_dict() for a in self.parameter_attributions],
            "suggestions": self.suggestions,
        }


# =============================================================================
# Win Probability Calculation
# =============================================================================


def rating_gap_to_win_probability(rating_gap: float, home_advantage: float = 0.03) -> float:
    """
    Convert rating gap to win probability using logistic function.

    The logistic curve maps rating differences to probabilities:
    - Gap of 0 → 50% (adjusted for home advantage)
    - Gap of 0.10 → ~65%
    - Gap of 0.20 → ~75%
    - Gap of 0.30 → ~85%
    - Gap of 0.40+ → ~92%+

    Args:
        rating_gap: Difference in ratings (positive = team A favored)
        home_advantage: Home field boost (default 0.03)

    Returns:
        Win probability for the favored team (0.5 to 1.0)
    """
    # Logistic function with steepness calibrated to college football
    # k = 10 gives reasonable spread across typical rating differences
    k = 10.0

    # Apply home advantage to the gap
    adjusted_gap = rating_gap + home_advantage

    # Logistic function: 1 / (1 + e^(-k*x))
    prob = 1.0 / (1.0 + math.exp(-k * adjusted_gap))

    return prob


def compute_upset_magnitude(rating_gap: float, win_prob: float) -> float:
    """
    Calculate how surprising an upset was.

    Args:
        rating_gap: How much higher the loser was rated
        win_prob: Pre-game win probability of the loser

    Returns:
        Upset magnitude (0 = not surprising, 1 = shocking)
    """
    # Magnitude based on how improbable the outcome was
    return win_prob  # Simple: the probability that was "defied"


# =============================================================================
# Game Contribution Decomposition
# =============================================================================


def decompose_game_contribution(
    game_result: GameResult,
    opponent_rating: float,
    config: AlgorithmConfig,
    normalized_opp_rating: float | None = None,
) -> GameContributionBreakdown:
    """
    Break down a game's contribution into its component parts.

    Args:
        game_result: The game from one team's perspective
        opponent_rating: Current rating of opponent
        config: Algorithm configuration
        normalized_opp_rating: Normalized [0,1] opponent rating for quality tiers

    Returns:
        GameContributionBreakdown with all components
    """
    # Base result
    base_result = config.win_base if game_result.is_win else 0.0

    # Margin bonus
    margin_bonus = compute_margin_bonus_curved(
        abs(game_result.margin),
        game_result.is_win,
        config,
    )

    # Venue adjustment
    venue_adjustment = compute_venue_adjustment(
        game_result.location,
        game_result.is_win,
        config,
    )

    # Game grade (base + margin + venue)
    game_grade = base_result + margin_bonus + venue_adjustment

    # Opponent contribution (quality losses hurt less than bad losses)
    weighted_opp_rating = opponent_rating * config.opponent_weight
    if game_result.is_win:
        opponent_contribution = weighted_opp_rating
    else:
        # Penalty proportional to opponent weakness (clamped for stability)
        # Loss to 0.9 team: -0.1, Loss to 0.1 team: -0.9
        opponent_weakness = 1.0 - weighted_opp_rating
        clamped_weakness = max(0.0, min(1.0, opponent_weakness))
        opponent_contribution = -clamped_weakness * config.loss_opponent_factor

    # Quality tier adjustment
    quality_tier_adjustment = 0.0
    if config.enable_quality_tiers and normalized_opp_rating is not None:
        if game_result.is_win and normalized_opp_rating >= config.elite_threshold:
            quality_tier_adjustment = config.elite_win_bonus
        elif not game_result.is_win and normalized_opp_rating <= config.bad_threshold:
            quality_tier_adjustment = -config.bad_loss_penalty

    # Conference multiplier (applied to opponent rating externally, tracked here)
    conference_multiplier = 1.0  # Would need conference info to compute

    # Recency weight (would need week info)
    recency_weight = 1.0

    # Total contribution
    total_contribution = game_grade + opponent_contribution + quality_tier_adjustment

    return GameContributionBreakdown(
        game_id=game_result.game_id,
        team_id=game_result.team_id,
        opponent_id=game_result.opponent_id,
        is_win=game_result.is_win,
        margin=game_result.margin,
        location=game_result.location,
        base_result=base_result,
        margin_bonus=margin_bonus,
        venue_adjustment=venue_adjustment,
        opponent_rating=opponent_rating,
        opponent_contribution=opponent_contribution,
        quality_tier_adjustment=quality_tier_adjustment,
        conference_multiplier=conference_multiplier,
        recency_weight=recency_weight,
        game_grade=game_grade,
        total_contribution=total_contribution,
    )


def decompose_team_rating(
    team_id: str,
    games: list[Game],
    config: AlgorithmConfig,
) -> list[GameContributionBreakdown]:
    """
    Decompose a team's entire rating into game-by-game contributions.

    Args:
        team_id: Team to analyze
        games: All games in the dataset
        config: Algorithm configuration

    Returns:
        List of GameContributionBreakdown for each game
    """
    # Run convergence to get ratings
    result = converge(games, config)
    normalized = normalize_ratings(result.ratings)

    # Build game results
    game_results = build_game_results(games)
    team_results = game_results.get(team_id, [])

    breakdowns = []
    for game_result in team_results:
        opp_rating = result.ratings.get(game_result.opponent_id, 0.5)
        norm_opp_rating = normalized.get(game_result.opponent_id, 0.5)

        breakdown = decompose_game_contribution(
            game_result,
            opp_rating,
            config,
            norm_opp_rating,
        )
        breakdowns.append(breakdown)

    return breakdowns


# =============================================================================
# Prediction Generation and Evaluation
# =============================================================================


def generate_predictions(
    games: list[Game],
    ratings: dict[str, float],
    config: AlgorithmConfig,
) -> list[Prediction]:
    """
    Generate win/loss predictions for games based on ratings.

    Args:
        games: Games to predict
        ratings: Team ratings (normalized 0-1)
        config: Algorithm configuration

    Returns:
        List of Prediction objects
    """
    predictions = []

    for game in games:
        home_rating = ratings.get(game.home_team_id, 0.5)
        away_rating = ratings.get(game.away_team_id, 0.5)

        # Calculate rating gap (positive = home favored)
        # Account for neutral site
        home_boost = 0.0 if game.neutral_site else config.venue_road_win / 2
        rating_gap = (home_rating + home_boost) - away_rating

        # Determine predicted winner
        if rating_gap >= 0:
            predicted_winner = game.home_team_id
            win_prob = rating_gap_to_win_probability(
                abs(rating_gap),
                home_advantage=0.0 if game.neutral_site else config.venue_road_win / 2,
            )
        else:
            predicted_winner = game.away_team_id
            win_prob = rating_gap_to_win_probability(
                abs(rating_gap),
                home_advantage=0.0,  # Away team doesn't get home advantage
            )

        # Determine actual winner if game played
        actual_winner = None
        was_correct = None
        is_upset = None
        upset_magnitude = None

        if game.home_score > 0 or game.away_score > 0:  # Game was played
            if game.home_score > game.away_score:
                actual_winner = game.home_team_id
            elif game.away_score > game.home_score:
                actual_winner = game.away_team_id
            else:
                actual_winner = "tie"

            was_correct = (predicted_winner == actual_winner)

            # Check if upset
            if not was_correct and actual_winner != "tie":
                is_upset = win_prob >= 0.55  # Only count as upset if we were confident
                if is_upset:
                    upset_magnitude = compute_upset_magnitude(abs(rating_gap), win_prob)

        predictions.append(Prediction(
            game_id=game.game_id,
            season=game.season,
            week=game.week,
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            home_rating=home_rating,
            away_rating=away_rating,
            rating_gap=rating_gap,
            predicted_winner=predicted_winner,
            win_probability=win_prob,
            actual_winner=actual_winner,
            was_correct=was_correct,
            is_upset=is_upset,
            upset_magnitude=upset_magnitude,
        ))

    return predictions


def compute_brier_score(predictions: list[Prediction]) -> float:
    """
    Compute Brier score for predictions.

    Brier score = mean((probability - outcome)^2)
    Lower is better. 0 = perfect, 0.25 = random guessing.

    Args:
        predictions: List of predictions with outcomes

    Returns:
        Brier score (0-1)
    """
    completed = [p for p in predictions if p.was_correct is not None]
    if not completed:
        return 0.0

    total = 0.0
    for p in completed:
        # Outcome: 1 if prediction correct, 0 if wrong
        outcome = 1.0 if p.was_correct else 0.0
        total += (p.win_probability - outcome) ** 2

    return total / len(completed)


def compute_calibration_error(predictions: list[Prediction]) -> float:
    """
    Compute calibration error (how well probabilities match actual outcomes).

    Groups predictions by probability bucket and compares predicted vs actual
    win rates.

    Args:
        predictions: List of predictions with outcomes

    Returns:
        Mean absolute calibration error
    """
    completed = [p for p in predictions if p.was_correct is not None]
    if not completed:
        return 0.0

    # Group into probability buckets (0.5-0.6, 0.6-0.7, etc.)
    buckets: dict[str, list[Prediction]] = {
        "0.50-0.60": [],
        "0.60-0.70": [],
        "0.70-0.80": [],
        "0.80-0.90": [],
        "0.90-1.00": [],
    }

    for p in completed:
        prob = p.win_probability
        if prob < 0.60:
            buckets["0.50-0.60"].append(p)
        elif prob < 0.70:
            buckets["0.60-0.70"].append(p)
        elif prob < 0.80:
            buckets["0.70-0.80"].append(p)
        elif prob < 0.90:
            buckets["0.80-0.90"].append(p)
        else:
            buckets["0.90-1.00"].append(p)

    # Compute calibration error per bucket
    errors = []
    bucket_midpoints = {
        "0.50-0.60": 0.55,
        "0.60-0.70": 0.65,
        "0.70-0.80": 0.75,
        "0.80-0.90": 0.85,
        "0.90-1.00": 0.95,
    }

    for bucket_name, bucket_preds in buckets.items():
        if not bucket_preds:
            continue

        expected = bucket_midpoints[bucket_name]
        actual = sum(1 for p in bucket_preds if p.was_correct) / len(bucket_preds)
        errors.append(abs(expected - actual))

    return sum(errors) / len(errors) if errors else 0.0


# =============================================================================
# Parameter Attribution
# =============================================================================


@dataclass
class WrongPredictionAnalysis:
    """Analysis of a single wrong prediction."""

    prediction: Prediction
    winner_breakdown: GameContributionBreakdown | None
    loser_breakdown: GameContributionBreakdown | None
    likely_culprits: list[str]  # Parameter names


def analyze_wrong_prediction(
    prediction: Prediction,
    games: list[Game],
    config: AlgorithmConfig,
) -> WrongPredictionAnalysis:
    """
    Analyze why a prediction was wrong.

    Args:
        prediction: The wrong prediction
        games: All games (for context)
        config: Current configuration

    Returns:
        Analysis with likely culprit parameters
    """
    likely_culprits = []

    # Find the game
    game = next((g for g in games if g.game_id == prediction.game_id), None)
    if not game:
        return WrongPredictionAnalysis(
            prediction=prediction,
            winner_breakdown=None,
            loser_breakdown=None,
            likely_culprits=[],
        )

    # Analyze patterns
    rating_gap = abs(prediction.rating_gap)
    actual_winner = prediction.actual_winner
    predicted_winner = prediction.predicted_winner

    # Pattern: G5 team beat P5 team
    # (Would need team info to determine - use rating as proxy)
    winner_rating = prediction.home_rating if actual_winner == prediction.home_team_id else prediction.away_rating
    loser_rating = prediction.home_rating if actual_winner != prediction.home_team_id else prediction.away_rating

    if winner_rating < 0.4 and loser_rating > 0.6:
        likely_culprits.append("g5_multiplier")

    # Pattern: Home underdog won
    if game.home_team_id == actual_winner and prediction.rating_gap < 0:
        if not game.neutral_site:
            likely_culprits.append("venue_road_win")

    # Pattern: Road team won when expected to lose
    if game.away_team_id == actual_winner and prediction.rating_gap > 0:
        if not game.neutral_site:
            likely_culprits.append("venue_road_win")

    # Pattern: Close game went wrong way (margin matters less)
    if rating_gap < 0.10:
        likely_culprits.append("margin_weight")

    # Pattern: Large upset (ratings too spread)
    if rating_gap > 0.25:
        likely_culprits.append("opponent_weight")

    return WrongPredictionAnalysis(
        prediction=prediction,
        winner_breakdown=None,  # Would compute if needed
        loser_breakdown=None,
        likely_culprits=likely_culprits,
    )


def attribute_errors_to_parameters(
    wrong_predictions: list[Prediction],
    games: list[Game],
    config: AlgorithmConfig,
) -> list[ParameterAttribution]:
    """
    Analyze wrong predictions and attribute errors to parameters.

    Args:
        wrong_predictions: List of wrong predictions
        games: All games
        config: Current configuration

    Returns:
        List of parameter attributions sorted by error count
    """
    # Track errors by parameter
    error_counts: dict[str, list[int]] = {}
    patterns: dict[str, list[str]] = {}

    for pred in wrong_predictions:
        analysis = analyze_wrong_prediction(pred, games, config)
        for param in analysis.likely_culprits:
            if param not in error_counts:
                error_counts[param] = []
                patterns[param] = []
            error_counts[param].append(pred.game_id)

    # Build attributions
    attributions = []

    param_suggestions = {
        "g5_multiplier": ("G5 teams beating P5 opponents", "Increase g5_multiplier", config.g5_multiplier, 0.95),
        "venue_road_win": ("Road/home advantage misaligned", "Adjust venue_road_win", config.venue_road_win, 0.08),
        "margin_weight": ("Close games going wrong", "Reduce margin_weight", config.margin_weight, 0.15),
        "opponent_weight": ("Rating spread too wide", "Reduce opponent_weight", config.opponent_weight, 0.9),
        "bad_loss_penalty": ("Teams with bad losses beating favorites", "Reduce bad_loss_penalty", config.bad_loss_penalty, 0.03),
        "elite_win_bonus": ("Elite win teams overvalued", "Reduce elite_win_bonus", config.elite_win_bonus, 0.03),
    }

    for param, game_ids in error_counts.items():
        if param in param_suggestions:
            pattern, suggestion, current, suggested = param_suggestions[param]
        else:
            pattern = "Unknown pattern"
            suggestion = "Review parameter"
            current = getattr(config, param, "N/A")
            suggested = None

        attributions.append(ParameterAttribution(
            parameter=param,
            error_count=len(game_ids),
            pattern_description=pattern,
            affected_games=game_ids,
            suggested_adjustment=suggestion,
            current_value=current,
            suggested_value=suggested,
        ))

    # Sort by error count descending
    attributions.sort(key=lambda a: a.error_count, reverse=True)

    return attributions


# =============================================================================
# Full Diagnostic Report
# =============================================================================


def generate_diagnostic_report(
    games: list[Game],
    config: AlgorithmConfig,
    week: int | None = None,
) -> DiagnosticReport:
    """
    Generate a full diagnostic report for predictions.

    Args:
        games: All games in the dataset
        config: Algorithm configuration
        week: Specific week to analyze (None = all weeks)

    Returns:
        DiagnosticReport with accuracy, calibration, and parameter attribution
    """
    if not games:
        return DiagnosticReport(
            season=0,
            week=week,
            total_games=0,
            correct_predictions=0,
            wrong_predictions=0,
            accuracy=0.0,
            brier_score=0.0,
            calibration_error=0.0,
            predictions=[],
            upsets=[],
            parameter_attributions=[],
            suggestions=[],
        )

    # Filter by week if specified
    if week is not None:
        games = [g for g in games if g.week == week]

    # Run convergence to get ratings
    result = converge(games, config)
    normalized = normalize_ratings(result.ratings)

    # Generate predictions
    predictions = generate_predictions(games, normalized, config)

    # Filter to completed games
    completed = [p for p in predictions if p.was_correct is not None]

    # Calculate metrics
    correct = sum(1 for p in completed if p.was_correct)
    wrong = len(completed) - correct
    accuracy = correct / len(completed) if completed else 0.0

    brier = compute_brier_score(predictions)
    calibration = compute_calibration_error(predictions)

    # Find upsets
    upsets = [p for p in completed if p.is_upset]
    upsets.sort(key=lambda p: p.upset_magnitude or 0, reverse=True)

    # Attribute errors to parameters
    wrong_preds = [p for p in completed if not p.was_correct]
    attributions = attribute_errors_to_parameters(wrong_preds, games, config)

    # Generate suggestions
    suggestions = []
    for attr in attributions[:3]:  # Top 3 culprits
        suggestions.append(f"{attr.suggested_adjustment}: {attr.current_value} → {attr.suggested_value}")

    if calibration > 0.10:
        suggestions.append("High calibration error - probabilities not matching outcomes")
    if brier > 0.22:
        suggestions.append("Brier score suggests poor prediction quality")

    season = games[0].season if games else 0

    return DiagnosticReport(
        season=season,
        week=week,
        total_games=len(completed),
        correct_predictions=correct,
        wrong_predictions=wrong,
        accuracy=accuracy,
        brier_score=brier,
        calibration_error=calibration,
        predictions=predictions,
        upsets=upsets,
        parameter_attributions=attributions,
        suggestions=suggestions,
    )


def predict_game(
    home_team_id: str,
    away_team_id: str,
    ratings: dict[str, float],
    config: AlgorithmConfig,
    neutral_site: bool = False,
) -> Prediction:
    """
    Predict a single game outcome.

    Args:
        home_team_id: Home team identifier
        away_team_id: Away team identifier
        ratings: Current team ratings
        config: Algorithm configuration
        neutral_site: Whether game is at neutral site

    Returns:
        Prediction object
    """
    home_rating = ratings.get(home_team_id, 0.5)
    away_rating = ratings.get(away_team_id, 0.5)

    # Calculate rating gap
    home_boost = 0.0 if neutral_site else config.venue_road_win / 2
    rating_gap = (home_rating + home_boost) - away_rating

    if rating_gap >= 0:
        predicted_winner = home_team_id
        win_prob = rating_gap_to_win_probability(
            abs(rating_gap),
            home_advantage=0.0 if neutral_site else config.venue_road_win / 2,
        )
    else:
        predicted_winner = away_team_id
        win_prob = rating_gap_to_win_probability(abs(rating_gap), home_advantage=0.0)

    return Prediction(
        game_id=0,
        season=0,
        week=0,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_rating=home_rating,
        away_rating=away_rating,
        rating_gap=rating_gap,
        predicted_winner=predicted_winner,
        win_probability=win_prob,
    )
