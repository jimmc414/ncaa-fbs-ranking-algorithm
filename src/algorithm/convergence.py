"""Iterative convergence algorithm for NCAA team ratings.

This module implements the core ranking algorithm, which uses iterative
convergence (similar to PageRank) to derive team ratings from game outcomes.

Key insight: You can't know how good a team is until you know how good their
opponents are, which depends on _their_ opponents, recursively.

CRITICAL: Quality losses hurt less than bad losses. The formula:
- Win:  contribution = game_grade + opponent_rating
- Loss: contribution = game_grade - (1 - opponent_rating) * loss_opponent_factor

With default loss_opponent_factor=1.0:
- Loss to 0.9 team: -(1-0.9) = -0.1 (small penalty)
- Loss to 0.1 team: -(1-0.1) = -0.9 (big penalty)
- Losses always hurt, but quality losses hurt less
"""

from dataclasses import dataclass

from src.algorithm.game_grade import compute_game_grade_for_result
from src.data.models import AlgorithmConfig, Game, GameResult, Team


@dataclass
class ConvergenceResult:
    """Result of the convergence algorithm."""

    ratings: dict[str, float]
    iterations: int
    final_max_delta: float
    delta_history: list[float]


def build_conference_multipliers(
    teams: list[Team] | None,
    config: AlgorithmConfig,
) -> dict[str, float]:
    """
    Build a mapping of team_id -> conference multiplier.

    Args:
        teams: List of Team objects with conference info
        config: Algorithm configuration with multiplier values

    Returns:
        Dict mapping team_id to its conference multiplier
    """
    if not teams or not config.enable_conference_adj:
        return {}

    # P5 conferences
    p5_conferences = {
        "SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "FBS Independents",
    }
    g5_conferences = {
        "American Athletic", "Mountain West", "Sun Belt", "MAC",
        "Mid-American", "Conference USA",
    }

    multipliers = {}
    for team in teams:
        if team.division == "fcs":
            multipliers[team.team_id] = config.fcs_multiplier
        elif team.conference in p5_conferences:
            multipliers[team.team_id] = config.p5_multiplier
        elif team.conference in g5_conferences:
            multipliers[team.team_id] = config.g5_multiplier
        else:
            # Unknown conference defaults to G5
            multipliers[team.team_id] = config.g5_multiplier

    return multipliers


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
    normalized_ratings: dict[str, float] | None = None,
    conference_multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Perform one iteration of rating updates.

    CRITICAL: Process teams in sorted order for determinism.

    The rating formula for each team is:
        Win:  contribution = GameGrade + R_opp
        Loss: contribution = GameGrade - (1 - R_opp) * loss_opponent_factor

    With default loss_opponent_factor=1.0:
        Loss to 0.9 team: -(1-0.9) = -0.1 (quality loss = small penalty)
        Loss to 0.1 team: -(1-0.1) = -0.9 (bad loss = big penalty)

    Configurable modifiers:
        - opponent_weight: Multiplier on opponent rating
        - loss_opponent_factor: Scale for loss penalty (default 1.0)
        - quality_tiers: Bonuses for elite wins, penalties for bad losses
        - conference_multipliers: P5/G5/FCS adjustments to opponent rating

    Args:
        current_ratings: Current team ratings
        game_results: Game results by team
        config: Algorithm configuration
        fcs_teams: Set of FCS team IDs (optional)
        normalized_ratings: Normalized [0,1] ratings for quality tier comparison
        conference_multipliers: Dict mapping team_id to conference multiplier

    Returns:
        New ratings dict
    """
    new_ratings = {}
    fcs_teams = fcs_teams or set()
    conference_multipliers = conference_multipliers or {}

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

            # Apply conference multiplier to opponent rating (G5 opponents worth less)
            conf_mult = conference_multipliers.get(result.opponent_id, 1.0)
            opp_rating = opp_rating * conf_mult

            # Apply opponent weight
            weighted_opp_rating = opp_rating * config.opponent_weight

            # CRITICAL: Quality losses hurt less than bad losses
            if result.is_win:
                contribution = game_grade + weighted_opp_rating
            else:
                # Penalty proportional to opponent weakness (clamped for convergence stability)
                # Loss to 0.9 team: penalty = 0.1, Loss to 0.1 team: penalty = 0.9
                opponent_weakness = 1.0 - weighted_opp_rating
                clamped_weakness = max(0.0, min(1.0, opponent_weakness))
                contribution = game_grade - clamped_weakness * config.loss_opponent_factor

            # Apply quality tier bonuses/penalties if enabled
            if config.enable_quality_tiers and normalized_ratings:
                norm_opp = normalized_ratings.get(result.opponent_id, 0.5)
                if result.is_win and norm_opp >= config.elite_threshold:
                    # Bonus for beating elite opponent
                    contribution += config.elite_win_bonus
                elif not result.is_win and norm_opp <= config.bad_threshold:
                    # Penalty for losing to bad opponent
                    contribution -= config.bad_loss_penalty

            total += contribution

        new_ratings[team_id] = total / len(results)

    return new_ratings


def converge(
    games: list[Game],
    config: AlgorithmConfig,
    fcs_teams: set[str] | None = None,
    teams: list[Team] | None = None,
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
        teams: List of Team objects for conference multipliers (optional)

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

    # Build conference multipliers if teams provided
    conference_multipliers = build_conference_multipliers(teams, config)

    # For quality tiers, we need normalized ratings from previous iteration
    normalized_ratings: dict[str, float] | None = None

    for iteration in range(config.max_iterations):
        # If quality tiers enabled, normalize current ratings for tier comparison
        if config.enable_quality_tiers:
            normalized_ratings = normalize_ratings(ratings)

        new_ratings = iterate_once(
            ratings, game_results, config, fcs_teams, normalized_ratings,
            conference_multipliers
        )

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
