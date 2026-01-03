"""Tiebreaker logic for resolving teams with similar ratings.

When two teams have ratings within the threshold (default 0.001), the
tiebreaker chain is applied in order:
1. Head-to-head result
2. Strength of Victory (avg rating of beaten opponents)
3. Common opponent margin differential
4. Away win percentage
"""

from src.data.models import Game


def head_to_head(team_a: str, team_b: str, games: list[Game]) -> int:
    """
    Determine head-to-head winner between two teams.

    Args:
        team_a: First team ID
        team_b: Second team ID
        games: List of all games

    Returns:
        1 if team_a wins H2H, -1 if team_b wins, 0 if no game or split
    """
    a_wins = 0
    b_wins = 0

    for game in games:
        # Check if this game involves both teams
        if not (
            (game.home_team_id == team_a and game.away_team_id == team_b)
            or (game.home_team_id == team_b and game.away_team_id == team_a)
        ):
            continue

        # Determine winner
        if game.home_score > game.away_score:
            winner = game.home_team_id
        elif game.away_score > game.home_score:
            winner = game.away_team_id
        else:
            # Tie game - neither wins
            continue

        if winner == team_a:
            a_wins += 1
        else:
            b_wins += 1

    if a_wins > b_wins:
        return 1
    elif b_wins > a_wins:
        return -1
    return 0


def strength_of_victory(
    team: str,
    games: list[Game],
    ratings: dict[str, float],
) -> float:
    """
    Calculate strength of victory (average rating of beaten opponents).

    Args:
        team: Team ID to calculate SOV for
        games: List of all games
        ratings: Dict mapping team_id to rating

    Returns:
        Average rating of teams this team has beaten, or 0.0 if no wins
    """
    beaten_opponents: list[float] = []

    for game in games:
        # Check if team played in this game
        if game.home_team_id == team:
            is_win = game.home_score > game.away_score
            opponent = game.away_team_id
        elif game.away_team_id == team:
            is_win = game.away_score > game.home_score
            opponent = game.home_team_id
        else:
            continue

        if is_win and opponent in ratings:
            beaten_opponents.append(ratings[opponent])

    if not beaten_opponents:
        return 0.0

    return sum(beaten_opponents) / len(beaten_opponents)


def common_opponent_margin(
    team_a: str,
    team_b: str,
    games: list[Game],
) -> int:
    """
    Calculate margin differential against common opponents.

    For each opponent that both teams played, compute:
    (team_a's margin vs opponent) - (team_b's margin vs opponent)

    Sum across all common opponents.

    Args:
        team_a: First team ID
        team_b: Second team ID
        games: List of all games

    Returns:
        Sum of margin differentials (positive favors team_a, negative favors team_b)
    """
    # Build margin lookup: team -> opponent -> margin
    margins: dict[str, dict[str, int]] = {}

    for game in games:
        home_margin = game.home_score - game.away_score

        # Record from home team's perspective
        margins.setdefault(game.home_team_id, {})[game.away_team_id] = home_margin

        # Record from away team's perspective
        margins.setdefault(game.away_team_id, {})[game.home_team_id] = -home_margin

    # Find common opponents
    a_opponents = set(margins.get(team_a, {}).keys())
    b_opponents = set(margins.get(team_b, {}).keys())
    common = a_opponents & b_opponents

    # Exclude each other from common opponents
    common.discard(team_a)
    common.discard(team_b)

    if not common:
        return 0

    # Sum margin differentials
    total = 0
    for opponent in common:
        a_margin = margins[team_a][opponent]
        b_margin = margins[team_b][opponent]
        total += a_margin - b_margin

    return total


def away_win_percentage(team: str, games: list[Game]) -> float:
    """
    Calculate away game win percentage.

    Only counts true away games (not neutral site).

    Args:
        team: Team ID to calculate for
        games: List of all games

    Returns:
        Percentage of away games won (0.0 to 1.0), or 0.0 if no away games
    """
    away_games = 0
    away_wins = 0

    for game in games:
        # Check if team is away team AND it's not neutral site
        if game.away_team_id == team and not game.neutral_site:
            away_games += 1
            if game.away_score > game.home_score:
                away_wins += 1

    if away_games == 0:
        return 0.0

    return away_wins / away_games


def resolve_ties(
    teams: list[tuple[str, float]],
    games: list[Game],
    ratings: dict[str, float],
    threshold: float = 0.001,
) -> list[str]:
    """
    Resolve ties using the tiebreaker chain.

    Tiebreaker order:
    1. Head-to-head result
    2. Strength of Victory (avg rating of beaten opponents)
    3. Common opponent margin differential
    4. Away win percentage

    Args:
        teams: List of (team_id, rating) tuples sorted by rating descending
        games: List of all games
        ratings: Dict mapping team_id to rating
        threshold: Rating difference threshold to consider teams tied

    Returns:
        Ordered list of team_ids with ties resolved
    """
    if not teams:
        return []

    if len(teams) == 1:
        return [teams[0][0]]

    # Sort by rating descending first
    sorted_teams = sorted(teams, key=lambda x: x[1], reverse=True)

    # Group teams that are within threshold of each other
    groups: list[list[tuple[str, float]]] = []
    current_group: list[tuple[str, float]] = [sorted_teams[0]]

    for i in range(1, len(sorted_teams)):
        # Compare to first team in current group
        if abs(sorted_teams[i][1] - current_group[0][1]) <= threshold:
            current_group.append(sorted_teams[i])
        else:
            groups.append(current_group)
            current_group = [sorted_teams[i]]

    groups.append(current_group)

    # Resolve each group
    result: list[str] = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0][0])
        else:
            resolved = _resolve_tied_group(group, games, ratings)
            result.extend(resolved)

    return result


def _resolve_tied_group(
    group: list[tuple[str, float]],
    games: list[Game],
    ratings: dict[str, float],
) -> list[str]:
    """
    Resolve a group of tied teams using tiebreaker chain.

    For groups > 2, we do pairwise comparisons and track wins/losses.
    """
    if len(group) == 1:
        return [group[0][0]]

    team_ids = [t[0] for t in group]

    if len(group) == 2:
        return _resolve_pair(team_ids[0], team_ids[1], games, ratings)

    # For 3+ teams, use pairwise tiebreaker wins
    # Each team gets a score based on tiebreaker wins against other tied teams
    tiebreaker_scores: dict[str, float] = {t: 0.0 for t in team_ids}

    for i, team_a in enumerate(team_ids):
        for team_b in team_ids[i + 1 :]:
            result = _compare_teams(team_a, team_b, games, ratings)
            if result > 0:
                tiebreaker_scores[team_a] += 1
            elif result < 0:
                tiebreaker_scores[team_b] += 1
            else:
                # Still tied, give half point each
                tiebreaker_scores[team_a] += 0.5
                tiebreaker_scores[team_b] += 0.5

    # Sort by tiebreaker score, then by original rating
    original_ratings = {t[0]: t[1] for t in group}
    return sorted(
        team_ids, key=lambda t: (tiebreaker_scores[t], original_ratings[t]), reverse=True
    )


def _resolve_pair(
    team_a: str,
    team_b: str,
    games: list[Game],
    ratings: dict[str, float],
) -> list[str]:
    """Resolve a two-team tie, returning ordered list."""
    result = _compare_teams(team_a, team_b, games, ratings)
    if result >= 0:
        return [team_a, team_b]
    return [team_b, team_a]


def _compare_teams(
    team_a: str,
    team_b: str,
    games: list[Game],
    ratings: dict[str, float],
) -> int:
    """
    Compare two teams using tiebreaker chain.

    Returns:
        1 if team_a wins, -1 if team_b wins, 0 if still tied
    """
    # 1. Head-to-head
    h2h = head_to_head(team_a, team_b, games)
    if h2h != 0:
        return h2h

    # 2. Strength of Victory
    sov_a = strength_of_victory(team_a, games, ratings)
    sov_b = strength_of_victory(team_b, games, ratings)
    if abs(sov_a - sov_b) > 0.0001:  # Small threshold for float comparison
        return 1 if sov_a > sov_b else -1

    # 3. Common opponent margin
    com = common_opponent_margin(team_a, team_b, games)
    if com != 0:
        return 1 if com > 0 else -1

    # 4. Away win percentage
    away_a = away_win_percentage(team_a, games)
    away_b = away_win_percentage(team_b, games)
    if abs(away_a - away_b) > 0.0001:
        return 1 if away_a > away_b else -1

    # All tiebreakers exhausted, still tied
    return 0
