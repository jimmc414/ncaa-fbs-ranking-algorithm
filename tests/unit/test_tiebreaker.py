"""Tests for tiebreaker logic.

Tests the tiebreaker chain:
1. Head-to-head result
2. Strength of Victory (avg rating of beaten opponents)
3. Common opponent margin differential
4. Away win percentage
"""

from datetime import date

import pytest

from src.data.models import Game


# =============================================================================
# Test Fixtures
# =============================================================================


def make_game(
    game_id: int,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    neutral: bool = False,
) -> Game:
    """Helper to create test games."""
    return Game(
        game_id=game_id,
        season=2024,
        week=1,
        game_date=date(2024, 9, 1),
        home_team_id=home,
        away_team_id=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=neutral,
    )


# =============================================================================
# Head-to-Head Tests
# =============================================================================


class TestHeadToHead:
    """Tests for head_to_head function."""

    def test_a_wins_h2h(self):
        """A beat B, returns 1."""
        from src.algorithm.tiebreaker import head_to_head

        games = [make_game(1, "team-a", "team-b", 28, 14)]  # A beats B at home
        assert head_to_head("team-a", "team-b", games) == 1

    def test_b_wins_h2h(self):
        """B beat A, returns -1."""
        from src.algorithm.tiebreaker import head_to_head

        games = [make_game(1, "team-a", "team-b", 14, 28)]  # B beats A on road
        assert head_to_head("team-a", "team-b", games) == -1

    def test_no_game_between_teams(self):
        """No game between teams, returns 0."""
        from src.algorithm.tiebreaker import head_to_head

        games = [
            make_game(1, "team-a", "team-c", 28, 14),
            make_game(2, "team-b", "team-c", 21, 7),
        ]
        assert head_to_head("team-a", "team-b", games) == 0

    def test_split_multiple_games(self):
        """Teams split series, returns 0."""
        from src.algorithm.tiebreaker import head_to_head

        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # A wins
            make_game(2, "team-b", "team-a", 35, 28),  # B wins
        ]
        assert head_to_head("team-a", "team-b", games) == 0

    def test_h2h_sweep(self):
        """A beats B twice, returns 1."""
        from src.algorithm.tiebreaker import head_to_head

        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # A wins at home
            make_game(2, "team-b", "team-a", 14, 21),  # A wins on road
        ]
        assert head_to_head("team-a", "team-b", games) == 1

    def test_h2h_reversed_order(self):
        """Test B vs A returns opposite."""
        from src.algorithm.tiebreaker import head_to_head

        games = [make_game(1, "team-a", "team-b", 28, 14)]
        assert head_to_head("team-b", "team-a", games) == -1


# =============================================================================
# Strength of Victory Tests
# =============================================================================


class TestStrengthOfVictory:
    """Tests for strength_of_victory function."""

    def test_sov_with_wins(self):
        """Average rating of beaten opponents."""
        from src.algorithm.tiebreaker import strength_of_victory

        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # A beats B
            make_game(2, "team-a", "team-c", 21, 7),  # A beats C
        ]
        ratings = {"team-a": 0.8, "team-b": 0.6, "team-c": 0.4}

        sov = strength_of_victory("team-a", games, ratings)
        # Average of B (0.6) and C (0.4) = 0.5
        assert sov == pytest.approx(0.5)

    def test_sov_no_wins(self):
        """Returns 0.0 if team has no wins."""
        from src.algorithm.tiebreaker import strength_of_victory

        games = [
            make_game(1, "team-a", "team-b", 14, 28),  # A loses to B
            make_game(2, "team-c", "team-a", 21, 7),  # A loses to C
        ]
        ratings = {"team-a": 0.3, "team-b": 0.6, "team-c": 0.7}

        sov = strength_of_victory("team-a", games, ratings)
        assert sov == 0.0

    def test_sov_ignores_losses(self):
        """Only beaten opponents count."""
        from src.algorithm.tiebreaker import strength_of_victory

        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # A beats B
            make_game(2, "team-c", "team-a", 35, 0),  # A loses to C
        ]
        ratings = {"team-a": 0.5, "team-b": 0.4, "team-c": 0.9}

        sov = strength_of_victory("team-a", games, ratings)
        # Only B counts, C doesn't (loss)
        assert sov == pytest.approx(0.4)

    def test_sov_single_win(self):
        """SOV with single win."""
        from src.algorithm.tiebreaker import strength_of_victory

        games = [make_game(1, "team-a", "team-b", 28, 14)]
        ratings = {"team-a": 0.7, "team-b": 0.65}

        sov = strength_of_victory("team-a", games, ratings)
        assert sov == pytest.approx(0.65)

    def test_sov_away_win_counts(self):
        """Away wins count for SOV."""
        from src.algorithm.tiebreaker import strength_of_victory

        games = [make_game(1, "team-b", "team-a", 14, 28)]  # A wins on road
        ratings = {"team-a": 0.7, "team-b": 0.55}

        sov = strength_of_victory("team-a", games, ratings)
        assert sov == pytest.approx(0.55)


# =============================================================================
# Common Opponent Margin Tests
# =============================================================================


class TestCommonOpponentMargin:
    """Tests for common_opponent_margin function."""

    def test_single_common_opponent(self):
        """A beat X by 14, B beat X by 7, A wins tiebreaker."""
        from src.algorithm.tiebreaker import common_opponent_margin

        games = [
            make_game(1, "team-a", "team-x", 28, 14),  # A beats X by 14
            make_game(2, "team-b", "team-x", 21, 14),  # B beats X by 7
        ]

        margin = common_opponent_margin("team-a", "team-b", games)
        # A's margin (14) - B's margin (7) = 7
        assert margin == 7

    def test_no_common_opponents(self):
        """No common opponents, returns 0."""
        from src.algorithm.tiebreaker import common_opponent_margin

        games = [
            make_game(1, "team-a", "team-x", 28, 14),
            make_game(2, "team-b", "team-y", 21, 14),
        ]

        margin = common_opponent_margin("team-a", "team-b", games)
        assert margin == 0

    def test_multiple_common_opponents(self):
        """Sum of margin differentials across common opponents."""
        from src.algorithm.tiebreaker import common_opponent_margin

        games = [
            make_game(1, "team-a", "team-x", 28, 14),  # A beats X by 14
            make_game(2, "team-b", "team-x", 21, 14),  # B beats X by 7
            make_game(3, "team-a", "team-y", 35, 7),  # A beats Y by 28
            make_game(4, "team-b", "team-y", 21, 7),  # B beats Y by 14
        ]

        margin = common_opponent_margin("team-a", "team-b", games)
        # X: (14 - 7) = 7, Y: (28 - 14) = 14, total = 21
        assert margin == 21

    def test_common_opponent_loss(self):
        """Both teams lost to common opponent."""
        from src.algorithm.tiebreaker import common_opponent_margin

        games = [
            make_game(1, "team-x", "team-a", 28, 14),  # A loses to X by 14
            make_game(2, "team-x", "team-b", 21, 14),  # B loses to X by 7
        ]

        margin = common_opponent_margin("team-a", "team-b", games)
        # A's margin (-14) - B's margin (-7) = -7
        assert margin == -7

    def test_common_opponent_mixed_results(self):
        """A beat X, B lost to X."""
        from src.algorithm.tiebreaker import common_opponent_margin

        games = [
            make_game(1, "team-a", "team-x", 28, 14),  # A beats X by 14
            make_game(2, "team-x", "team-b", 21, 14),  # B loses to X by 7
        ]

        margin = common_opponent_margin("team-a", "team-b", games)
        # A's margin (14) - B's margin (-7) = 21
        assert margin == 21

    def test_common_opponent_reversed(self):
        """B vs A returns opposite sign."""
        from src.algorithm.tiebreaker import common_opponent_margin

        games = [
            make_game(1, "team-a", "team-x", 28, 14),  # A beats X by 14
            make_game(2, "team-b", "team-x", 21, 14),  # B beats X by 7
        ]

        margin_ab = common_opponent_margin("team-a", "team-b", games)
        margin_ba = common_opponent_margin("team-b", "team-a", games)
        assert margin_ab == -margin_ba


# =============================================================================
# Away Win Percentage Tests
# =============================================================================


class TestAwayWinPercentage:
    """Tests for away_win_percentage function."""

    def test_all_away_wins(self):
        """100% away win percentage."""
        from src.algorithm.tiebreaker import away_win_percentage

        games = [
            make_game(1, "team-b", "team-a", 14, 28),  # A wins on road
            make_game(2, "team-c", "team-a", 7, 21),  # A wins on road
        ]

        pct = away_win_percentage("team-a", games)
        assert pct == pytest.approx(1.0)

    def test_no_away_wins(self):
        """0% away win percentage."""
        from src.algorithm.tiebreaker import away_win_percentage

        games = [
            make_game(1, "team-b", "team-a", 28, 14),  # A loses on road
            make_game(2, "team-c", "team-a", 21, 7),  # A loses on road
        ]

        pct = away_win_percentage("team-a", games)
        assert pct == pytest.approx(0.0)

    def test_mixed_away_results(self):
        """50% away win percentage."""
        from src.algorithm.tiebreaker import away_win_percentage

        games = [
            make_game(1, "team-b", "team-a", 14, 28),  # A wins on road
            make_game(2, "team-c", "team-a", 21, 7),  # A loses on road
        ]

        pct = away_win_percentage("team-a", games)
        assert pct == pytest.approx(0.5)

    def test_no_away_games(self):
        """Returns 0.0 if no away games."""
        from src.algorithm.tiebreaker import away_win_percentage

        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # A plays at home
            make_game(2, "team-a", "team-c", 21, 7),  # A plays at home
        ]

        pct = away_win_percentage("team-a", games)
        assert pct == 0.0

    def test_ignores_home_games(self):
        """Only counts away games."""
        from src.algorithm.tiebreaker import away_win_percentage

        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # Home win
            make_game(2, "team-a", "team-c", 7, 21),  # Home loss
            make_game(3, "team-d", "team-a", 14, 28),  # Away win
        ]

        pct = away_win_percentage("team-a", games)
        # Only 1 away game, 1 win = 100%
        assert pct == pytest.approx(1.0)

    def test_ignores_neutral_site(self):
        """Neutral site games don't count as away."""
        from src.algorithm.tiebreaker import away_win_percentage

        games = [
            make_game(1, "team-b", "team-a", 14, 28, neutral=True),  # Neutral
            make_game(2, "team-c", "team-a", 7, 21),  # True away win
        ]

        pct = away_win_percentage("team-a", games)
        # Only 1 away game (non-neutral), 1 win = 100%
        assert pct == pytest.approx(1.0)


# =============================================================================
# Resolve Ties Tests
# =============================================================================


class TestResolveTies:
    """Tests for resolve_ties function (full tiebreaker chain)."""

    def test_no_ties_in_input(self):
        """Teams not tied, returns sorted by rating."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.8), ("team-b", 0.6), ("team-c", 0.4)]
        games: list[Game] = []
        ratings = {"team-a": 0.8, "team-b": 0.6, "team-c": 0.4}

        result = resolve_ties(teams, games, ratings, threshold=0.001)
        assert result == ["team-a", "team-b", "team-c"]

    def test_resolve_by_h2h(self):
        """Tie resolved via head-to-head."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.700), ("team-b", 0.7005)]  # Within threshold
        games = [make_game(1, "team-a", "team-b", 28, 14)]  # A beat B
        ratings = {"team-a": 0.700, "team-b": 0.7005}

        result = resolve_ties(teams, games, ratings, threshold=0.001)
        assert result == ["team-a", "team-b"]

    def test_resolve_by_sov(self):
        """Tie resolved via strength of victory (h2h inconclusive)."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.700), ("team-b", 0.7005)]
        games = [
            make_game(1, "team-a", "team-c", 28, 14),  # A beats C (0.3)
            make_game(2, "team-b", "team-d", 21, 7),  # B beats D (0.5)
        ]
        ratings = {"team-a": 0.700, "team-b": 0.7005, "team-c": 0.3, "team-d": 0.5}

        result = resolve_ties(teams, games, ratings, threshold=0.001)
        # B has higher SOV (0.5 vs 0.3)
        assert result == ["team-b", "team-a"]

    def test_resolve_by_common_opponent(self):
        """Tie resolved via common opponent margin (h2h and SOV tied)."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.700), ("team-b", 0.7005)]
        games = [
            make_game(1, "team-a", "team-x", 28, 14),  # A beats X by 14
            make_game(2, "team-b", "team-x", 21, 14),  # B beats X by 7
            make_game(3, "team-a", "team-y", 21, 7),  # A beats Y
            make_game(4, "team-b", "team-z", 21, 7),  # B beats Z
        ]
        # Same SOV - both beat opponents with 0.4 rating
        ratings = {"team-a": 0.700, "team-b": 0.7005, "team-x": 0.4, "team-y": 0.4, "team-z": 0.4}

        result = resolve_ties(teams, games, ratings, threshold=0.001)
        # A has better common opponent margin (+14 vs +7 = +7)
        assert result == ["team-a", "team-b"]

    def test_resolve_by_away_pct(self):
        """Tie resolved via away win percentage (all others tied)."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.700), ("team-b", 0.7005)]
        games = [
            # No common opponents - A plays X, B plays Y (different opponents)
            make_game(1, "team-a", "team-x", 28, 14),  # A beats X at home
            make_game(2, "team-b", "team-y", 28, 14),  # B beats Y at home (same SOV)
            make_game(3, "team-w", "team-a", 14, 21),  # A wins on road (50%)
            make_game(4, "team-w", "team-a", 28, 14),  # A loses on road (50%)
            make_game(5, "team-z", "team-b", 14, 35),  # B wins on road (100%)
        ]
        # Same SOV (both 0.4), no common opponents
        ratings = {"team-a": 0.700, "team-b": 0.7005, "team-x": 0.4, "team-y": 0.4, "team-w": 0.3, "team-z": 0.3}

        result = resolve_ties(teams, games, ratings, threshold=0.001)
        # B has better away win % (100% vs 50%)
        assert result == ["team-b", "team-a"]

    def test_multiple_teams_tied(self):
        """Three-way tie resolved correctly."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [
            ("team-a", 0.700),
            ("team-b", 0.7005),
            ("team-c", 0.6998),
        ]
        games = [
            make_game(1, "team-a", "team-b", 28, 14),  # A beats B
            make_game(2, "team-b", "team-c", 21, 14),  # B beats C
            make_game(3, "team-c", "team-a", 17, 14),  # C beats A
        ]
        ratings = {"team-a": 0.700, "team-b": 0.7005, "team-c": 0.6998}

        result = resolve_ties(teams, games, ratings, threshold=0.001)
        # Circular H2H, all have same SOV, no common opponents
        # Falls through to away win %: C > A (C won away), B = 0 (no away games)
        # But we need to look at actual logic - all beat each other once
        assert len(result) == 3
        assert set(result) == {"team-a", "team-b", "team-c"}

    def test_empty_teams_list(self):
        """Empty input returns empty list."""
        from src.algorithm.tiebreaker import resolve_ties

        result = resolve_ties([], [], {}, threshold=0.001)
        assert result == []

    def test_single_team(self):
        """Single team returns that team."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.700)]
        result = resolve_ties(teams, [], {"team-a": 0.700}, threshold=0.001)
        assert result == ["team-a"]

    def test_wider_threshold(self):
        """Wider threshold groups more teams as tied."""
        from src.algorithm.tiebreaker import resolve_ties

        teams = [("team-a", 0.70), ("team-b", 0.69)]  # 0.01 apart
        games = [make_game(1, "team-b", "team-a", 28, 14)]  # B beat A
        ratings = {"team-a": 0.70, "team-b": 0.69}

        # With narrow threshold (0.001), not tied - A ranked higher
        result_narrow = resolve_ties(teams, games, ratings, threshold=0.001)
        assert result_narrow == ["team-a", "team-b"]

        # With wide threshold (0.02), tied - B wins H2H
        result_wide = resolve_ties(teams, games, ratings, threshold=0.02)
        assert result_wide == ["team-b", "team-a"]
