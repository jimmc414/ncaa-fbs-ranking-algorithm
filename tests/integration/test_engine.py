"""Integration tests for the RankingEngine.

Tests the orchestration layer that ties together:
- Convergence algorithm
- Tiebreaker logic
- TeamRating generation
"""

from datetime import date

import pytest

from src.data.models import AlgorithmConfig, Game, TeamRating, RatingExplanation


# =============================================================================
# Test Fixtures
# =============================================================================


def make_game(
    game_id: int,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    week: int = 1,
    neutral: bool = False,
    postseason: bool = False,
) -> Game:
    """Helper to create test games."""
    return Game(
        game_id=game_id,
        season=2024,
        week=week,
        game_date=date(2024, 9, 1 + week),
        home_team_id=home,
        away_team_id=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=neutral,
        postseason=postseason,
    )


@pytest.fixture
def four_team_games() -> list[Game]:
    """Canonical 4-team round robin test case.

    A beats B, C, D
    B beats C, D
    C beats D
    Expected order: A > B > C > D
    """
    return [
        make_game(1, "team-a", "team-b", 28, 14, week=1),  # A beats B
        make_game(2, "team-a", "team-c", 35, 7, week=2),   # A beats C
        make_game(3, "team-a", "team-d", 42, 0, week=3),   # A beats D
        make_game(4, "team-b", "team-c", 21, 14, week=1),  # B beats C
        make_game(5, "team-b", "team-d", 28, 7, week=2),   # B beats D
        make_game(6, "team-c", "team-d", 17, 10, week=3),  # C beats D
    ]


@pytest.fixture
def games_with_postseason() -> list[Game]:
    """Games including postseason bowl game."""
    return [
        make_game(1, "team-a", "team-b", 28, 14, week=1),
        make_game(2, "team-a", "team-c", 35, 7, week=2),
        make_game(3, "team-b", "team-c", 21, 14, week=3),
        # Bowl game
        make_game(4, "team-a", "team-b", 31, 28, week=15, postseason=True),
    ]


@pytest.fixture
def tied_teams_games() -> list[Game]:
    """Games that produce tied ratings (circular results)."""
    return [
        make_game(1, "team-a", "team-b", 21, 14, week=1),  # A beats B
        make_game(2, "team-b", "team-c", 21, 14, week=2),  # B beats C
        make_game(3, "team-c", "team-a", 21, 14, week=3),  # C beats A (circular)
        # Common opponent for tiebreaker
        make_game(4, "team-a", "team-x", 35, 7, week=4),   # A beats X by 28
        make_game(5, "team-b", "team-x", 28, 14, week=4),  # B beats X by 14
        make_game(6, "team-c", "team-x", 21, 14, week=4),  # C beats X by 7
    ]


# =============================================================================
# Basic Ranking Tests
# =============================================================================


class TestRankSeason:
    """Tests for rank_season method."""

    @pytest.mark.asyncio
    async def test_returns_team_ratings(self, four_team_games: list[Game]):
        """Returns list of TeamRating objects."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        assert len(rankings) == 4
        assert all(isinstance(r, TeamRating) for r in rankings)

    @pytest.mark.asyncio
    async def test_ordered_by_rank(self, four_team_games: list[Game]):
        """Rankings are ordered by rank (1, 2, 3, ...)."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        ranks = [r.rank for r in rankings]
        assert ranks == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_highest_rating_is_rank_one(self, four_team_games: list[Game]):
        """Team with highest rating gets rank 1."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        # Ratings should decrease as rank increases
        ratings = [r.rating for r in rankings]
        assert ratings == sorted(ratings, reverse=True)

    @pytest.mark.asyncio
    async def test_canonical_order(self, four_team_games: list[Game]):
        """4-team round robin produces A > B > C > D order."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        team_order = [r.team_id for r in rankings]
        assert team_order == ["team-a", "team-b", "team-c", "team-d"]

    @pytest.mark.asyncio
    async def test_includes_win_loss_record(self, four_team_games: list[Game]):
        """TeamRating includes wins and losses."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        # Team A: 3 wins, 0 losses
        team_a = next(r for r in rankings if r.team_id == "team-a")
        assert team_a.wins == 3
        assert team_a.losses == 0

        # Team D: 0 wins, 3 losses
        team_d = next(r for r in rankings if r.team_id == "team-d")
        assert team_d.wins == 0
        assert team_d.losses == 3

    @pytest.mark.asyncio
    async def test_games_played_count(self, four_team_games: list[Game]):
        """TeamRating includes games_played."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        # All teams play 3 games in round robin
        for r in rankings:
            assert r.games_played == 3


# =============================================================================
# Week-by-Week Tests
# =============================================================================


class TestRankAsOfWeek:
    """Tests for rank_as_of_week method."""

    @pytest.mark.asyncio
    async def test_filters_games_through_week(self, four_team_games: list[Game]):
        """Only includes games through specified week."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_as_of_week(2024, week=1, games=four_team_games)

        # Week 1 only has 2 games (A vs B, B vs C)
        total_games = sum(r.games_played for r in rankings) // 2  # Each game counted twice
        assert total_games == 2

    @pytest.mark.asyncio
    async def test_week_progression(self, four_team_games: list[Game]):
        """Rankings change as weeks progress."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()

        week1 = await engine.rank_as_of_week(2024, week=1, games=four_team_games)
        week3 = await engine.rank_as_of_week(2024, week=3, games=four_team_games)

        # More games played by week 3
        week1_total = sum(r.games_played for r in week1)
        week3_total = sum(r.games_played for r in week3)
        assert week3_total > week1_total


class TestRankAllWeeks:
    """Tests for rank_all_weeks method."""

    @pytest.mark.asyncio
    async def test_returns_dict_by_week(self, four_team_games: list[Game]):
        """Returns dict mapping week to rankings."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        all_weeks = await engine.rank_all_weeks(2024, games=four_team_games)

        assert isinstance(all_weeks, dict)
        assert 1 in all_weeks
        assert 3 in all_weeks

    @pytest.mark.asyncio
    async def test_each_week_has_rankings(self, four_team_games: list[Game]):
        """Each week has a list of TeamRating."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        all_weeks = await engine.rank_all_weeks(2024, games=four_team_games)

        for week, rankings in all_weeks.items():
            assert isinstance(rankings, list)
            assert all(isinstance(r, TeamRating) for r in rankings)


# =============================================================================
# Postseason Tests
# =============================================================================


class TestPostseasonHandling:
    """Tests for postseason game handling."""

    @pytest.mark.asyncio
    async def test_include_postseason_true(self, games_with_postseason: list[Game]):
        """Postseason games included when flag is True."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(
            2024, include_postseason=True, games=games_with_postseason
        )

        # Team A plays 3 regular + 1 postseason = 4 games
        team_a = next(r for r in rankings if r.team_id == "team-a")
        assert team_a.games_played == 3  # 3 games involving A

    @pytest.mark.asyncio
    async def test_include_postseason_false(self, games_with_postseason: list[Game]):
        """Postseason games excluded when flag is False."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(
            2024, include_postseason=False, games=games_with_postseason
        )

        # Without postseason, team A plays 2 games
        team_a = next(r for r in rankings if r.team_id == "team-a")
        assert team_a.games_played == 2


# =============================================================================
# Strength of Schedule / Victory Tests
# =============================================================================


class TestStrengthMetrics:
    """Tests for SOS and SOV computation."""

    @pytest.mark.asyncio
    async def test_sos_computed(self, four_team_games: list[Game]):
        """Strength of schedule is computed."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        for r in rankings:
            assert r.strength_of_schedule is not None

    @pytest.mark.asyncio
    async def test_sov_computed(self, four_team_games: list[Game]):
        """Strength of victory is computed."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        # Teams with wins should have SOV
        team_a = next(r for r in rankings if r.team_id == "team-a")
        assert team_a.strength_of_victory is not None
        assert team_a.strength_of_victory > 0

    @pytest.mark.asyncio
    async def test_sov_zero_for_winless(self, four_team_games: list[Game]):
        """Team with no wins has SOV of 0."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=four_team_games)

        team_d = next(r for r in rankings if r.team_id == "team-d")
        assert team_d.strength_of_victory == 0.0


# =============================================================================
# Tiebreaker Tests
# =============================================================================


class TestTiebreakerApplication:
    """Tests for tiebreaker resolution."""

    @pytest.mark.asyncio
    async def test_tiebreaker_resolves_circular(self, tied_teams_games: list[Game]):
        """Circular results are resolved via tiebreaker."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=tied_teams_games)

        # Should have unique ranks (no ties in final output)
        ranks = [r.rank for r in rankings]
        assert len(ranks) == len(set(ranks))


# =============================================================================
# Rating Explanation Tests
# =============================================================================


class TestExplainRating:
    """Tests for explain_rating method."""

    @pytest.mark.asyncio
    async def test_returns_explanation(self, four_team_games: list[Game]):
        """Returns RatingExplanation object."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        explanation = await engine.explain_rating(
            2024, "team-a", games=four_team_games
        )

        assert isinstance(explanation, RatingExplanation)

    @pytest.mark.asyncio
    async def test_includes_game_breakdown(self, four_team_games: list[Game]):
        """Explanation includes game-by-game contributions."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        explanation = await engine.explain_rating(
            2024, "team-a", games=four_team_games
        )

        assert len(explanation.games) == 3  # A plays 3 games

    @pytest.mark.asyncio
    async def test_includes_convergence_info(self, four_team_games: list[Game]):
        """Explanation includes convergence iterations."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        explanation = await engine.explain_rating(
            2024, "team-a", games=four_team_games
        )

        assert explanation.iterations_to_converge > 0

    @pytest.mark.asyncio
    async def test_includes_rank(self, four_team_games: list[Game]):
        """Explanation includes team's rank."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        explanation = await engine.explain_rating(
            2024, "team-a", games=four_team_games
        )

        assert explanation.rank == 1  # Team A should be #1


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for custom algorithm configuration."""

    @pytest.mark.asyncio
    async def test_custom_config(self, four_team_games: list[Game]):
        """Uses provided AlgorithmConfig."""
        from src.ranking.engine import RankingEngine

        config = AlgorithmConfig(
            convergence_threshold=0.001,  # Different threshold
            margin_cap=21,  # Different cap
        )
        engine = RankingEngine(config=config)
        rankings = await engine.rank_season(2024, games=four_team_games)

        assert len(rankings) == 4

    @pytest.mark.asyncio
    async def test_default_config(self, four_team_games: list[Game]):
        """Uses default config when none provided."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()  # No config
        rankings = await engine.rank_season(2024, games=four_team_games)

        assert len(rankings) == 4


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_games_list(self):
        """Empty games list returns empty rankings."""
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=[])

        assert rankings == []

    @pytest.mark.asyncio
    async def test_single_game(self):
        """Single game produces 2-team rankings."""
        from src.ranking.engine import RankingEngine

        games = [make_game(1, "team-a", "team-b", 28, 14)]
        engine = RankingEngine()
        rankings = await engine.rank_season(2024, games=games)

        assert len(rankings) == 2
        # Winner should be ranked higher
        assert rankings[0].team_id == "team-a"
