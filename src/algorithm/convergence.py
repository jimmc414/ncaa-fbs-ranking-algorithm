"""Iterative convergence algorithm for NCAA team ratings.

This module implements the core ranking algorithm, which uses iterative
convergence (similar to PageRank) to derive team ratings from game outcomes.

Key insight: You can't know how good a team is until you know how good their
opponents are, which depends on _their_ opponents, recursively.

CRITICAL: Losses SUBTRACT opponent rating, not add. This ensures:
- Losing to a good team hurts less than losing to a bad team
- But losing always hurts (no "quality loss" bonus)
"""

from dataclasses import dataclass

from src.algorithm.game_grade import compute_game_grade_for_result
from src.data.models import AlgorithmConfig, Game, GameResult


@dataclass
class ConvergenceResult:
    """Result of the convergence algorithm."""

    ratings: dict[str, float]
    iterations: int
    final_max_delta: float
    delta_history: list[float]


def initialize_ratings(games: list[Game]) -> dict[str, float]:
    """
    Initialize all teams to 0.500 rating.

    Args:
        games: List of games to extract teams from

    Returns:
        Dict mapping team_id to initial rating
    """
    teams = set()
    for game in games:
        teams.add(game.home_team_id)
        teams.add(game.away_team_id)
    return {team: 0.5 for team in teams}


def build_game_results(games: list[Game]) -> dict[str, list[GameResult]]:
    """
    Build a lookup of game results by team.

    Args:
        games: List of games

    Returns:
        Dict mapping team_id to list of GameResult objects
    """
    results: dict[str, list[GameResult]] = {}
    for game in games:
        home_result, away_result = game.to_results()
        results.setdefault(home_result.team_id, []).append(home_result)
        results.setdefault(away_result.team_id, []).append(away_result)
    return results


def iterate_once(
    current_ratings: dict[str, float],
    game_results: dict[str, list[GameResult]],
    config: AlgorithmConfig,
    fcs_teams: set[str] | None = None,
) -> dict[str, float]:
    """
    Perform one iteration of rating updates.

    CRITICAL: Process teams in sorted order for determinism.

    The rating formula for each team is:
        R_t = (1/N) * sum(GameGrade_i + W_i * R_opp_i - L_i * R_opp_i)

    Where:
        - N is number of games
        - W_i is 1 if win, 0 otherwise
        - L_i is 1 if loss, 0 otherwise
        - R_opp_i is opponent's current rating

    Args:
        current_ratings: Current team ratings
        game_results: Game results by team
        config: Algorithm configuration
        fcs_teams: Set of FCS team IDs (optional)

    Returns:
        New ratings dict
    """
    new_ratings = {}
    fcs_teams = fcs_teams or set()

    # CRITICAL: Process teams in sorted order for determinism
    for team_id in sorted(current_ratings.keys()):
        # FCS teams get fixed rating if configured
        if team_id in fcs_teams and config.fcs_fixed_rating is not None:
            new_ratings[team_id] = config.fcs_fixed_rating
            continue

        results = game_results.get(team_id, [])
        if not results:
            new_ratings[team_id] = config.initial_rating
            continue

        total = 0.0
        for result in results:
            game_grade = compute_game_grade_for_result(result, config)
            opp_rating = current_ratings[result.opponent_id]

            # CRITICAL: Losses SUBTRACT opponent rating
            if result.is_win:
                contribution = game_grade + opp_rating
            else:
                contribution = game_grade - opp_rating

            total += contribution

        new_ratings[team_id] = total / len(results)

    return new_ratings


def converge(
    games: list[Game],
    config: AlgorithmConfig,
    fcs_teams: set[str] | None = None,
) -> ConvergenceResult:
    """
    Run iterative convergence until ratings stabilize.

    The algorithm:
    1. Initializes all teams to 0.500
    2. Computes new ratings based on game outcomes and current opponent ratings
    3. Repeats until max change between iterations is below threshold

    Args:
        games: List of games
        config: Algorithm configuration
        fcs_teams: Set of FCS team IDs (optional)

    Returns:
        ConvergenceResult with final ratings and diagnostics
    """
    if not games:
        return ConvergenceResult(
            ratings={},
            iterations=0,
            final_max_delta=0.0,
            delta_history=[],
        )

    # Build game results lookup
    game_results = build_game_results(games)

    # Initialize ratings
    ratings = initialize_ratings(games)
    delta_history: list[float] = []

    for iteration in range(config.max_iterations):
        new_ratings = iterate_once(ratings, game_results, config, fcs_teams)

        # Compute max delta
        max_delta = max(
            abs(new_ratings[t] - ratings[t]) for t in ratings
        )
        delta_history.append(max_delta)

        ratings = new_ratings

        if max_delta < config.convergence_threshold:
            return ConvergenceResult(
                ratings=ratings,
                iterations=iteration + 1,
                final_max_delta=max_delta,
                delta_history=delta_history,
            )

    return ConvergenceResult(
        ratings=ratings,
        iterations=config.max_iterations,
        final_max_delta=delta_history[-1] if delta_history else 0.0,
        delta_history=delta_history,
    )


def normalize_ratings(ratings: dict[str, float]) -> dict[str, float]:
    """
    Scale ratings to [0, 1] range.

    This should only be called AFTER convergence, not during iteration.

    Args:
        ratings: Raw ratings dict

    Returns:
        Normalized ratings dict
    """
    if not ratings:
        return {}

    min_r = min(ratings.values())
    max_r = max(ratings.values())

    if max_r == min_r:
        return {t: 0.5 for t in ratings}

    return {t: (r - min_r) / (max_r - min_r) for t, r in ratings.items()}
