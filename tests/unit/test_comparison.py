"""Unit tests for the RankingComparator.

Tests the historical comparison module that compares
algorithm rankings to AP/CFP polls.
"""

from datetime import date

import pytest

from src.data.models import (
    AlgorithmConfig,
    ComparisonResult,
    Game,
    TeamRating,
)


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
        neutral_site=False,
        postseason=False,
    )


def make_team_rating(
    team_id: str,
    rank: int,
    rating: float = 0.5,
) -> TeamRating:
    """Helper to create test TeamRating."""
    return TeamRating(
        team_id=team_id,
        season=2024,
        week=15,
        rating=rating,
        rank=rank,
        games_played=12,
        wins=10,
        losses=2,
    )


@pytest.fixture
def sample_our_rankings() -> list[TeamRating]:
    """Sample algorithm rankings (25 teams)."""
    return [
        make_team_rating("team-a", 1, 0.95),
        make_team_rating("team-b", 2, 0.90),
        make_team_rating("team-c", 3, 0.85),
        make_team_rating("team-d", 4, 0.80),
        make_team_rating("team-e", 5, 0.75),
        make_team_rating("team-f", 6, 0.70),
        make_team_rating("team-g", 7, 0.65),
        make_team_rating("team-h", 8, 0.60),
        make_team_rating("team-i", 9, 0.55),
        make_team_rating("team-j", 10, 0.50),
    ]


@pytest.fixture
def sample_poll_rankings() -> list[dict]:
    """Sample poll rankings (matching teams, different order)."""
    return [
        {"team_id": "team-a", "rank": 1, "points": 1500},
        {"team_id": "team-c", "rank": 2, "points": 1450},  # We rank 3
        {"team_id": "team-b", "rank": 3, "points": 1400},  # We rank 2
        {"team_id": "team-e", "rank": 4, "points": 1350},  # We rank 5
        {"team_id": "team-d", "rank": 5, "points": 1300},  # We rank 4
        {"team_id": "team-f", "rank": 6, "points": 1250},
        {"team_id": "team-h", "rank": 7, "points": 1200},  # We rank 8
        {"team_id": "team-g", "rank": 8, "points": 1150},  # We rank 7
        {"team_id": "team-j", "rank": 9, "points": 1100},  # We rank 10
        {"team_id": "team-i", "rank": 10, "points": 1050},  # We rank 9
    ]


@pytest.fixture
def four_team_games() -> list[Game]:
    """Canonical 4-team round robin for engine tests."""
    return [
        make_game(1, "team-a", "team-b", 28, 14, week=1),
        make_game(2, "team-a", "team-c", 35, 7, week=2),
        make_game(3, "team-a", "team-d", 42, 0, week=3),
        make_game(4, "team-b", "team-c", 21, 14, week=1),
        make_game(5, "team-b", "team-d", 28, 7, week=2),
        make_game(6, "team-c", "team-d", 17, 10, week=3),
    ]


# =============================================================================
# Spearman Correlation Tests
# =============================================================================


class TestSpearmanCorrelation:
    """Tests for Spearman rank correlation calculation."""

    def test_identical_rankings_returns_one(self):
        """Identical rankings should return correlation of 1.0."""
        from src.ranking.comparison import RankingComparator

        our_ranks = [1, 2, 3, 4, 5]
        poll_ranks = [1, 2, 3, 4, 5]

        correlation = RankingComparator._compute_spearman(our_ranks, poll_ranks)

        assert correlation == pytest.approx(1.0)

    def test_reverse_rankings_returns_negative_one(self):
        """Reversed rankings should return correlation of -1.0."""
        from src.ranking.comparison import RankingComparator

        our_ranks = [1, 2, 3, 4, 5]
        poll_ranks = [5, 4, 3, 2, 1]

        correlation = RankingComparator._compute_spearman(our_ranks, poll_ranks)

        assert correlation == pytest.approx(-1.0)

    def test_partial_correlation(self):
        """Partial correlation should be between -1 and 1."""
        from src.ranking.comparison import RankingComparator

        our_ranks = [1, 2, 3, 4, 5]
        poll_ranks = [1, 3, 2, 5, 4]  # Slight reordering

        correlation = RankingComparator._compute_spearman(our_ranks, poll_ranks)

        assert -1.0 < correlation < 1.0
        assert correlation > 0  # Still mostly similar

    def test_single_element_returns_one(self):
        """Single element lists should return NaN or handle gracefully."""
        from src.ranking.comparison import RankingComparator

        our_ranks = [1]
        poll_ranks = [1]

        # scipy returns NaN for single element, we should handle this
        correlation = RankingComparator._compute_spearman(our_ranks, poll_ranks)

        # Could be 1.0 or NaN depending on implementation
        # We'll accept either a valid float or 1.0 as default
        assert correlation == pytest.approx(1.0) or correlation != correlation  # NaN check

    def test_empty_lists_returns_zero(self):
        """Empty lists should return 0.0 correlation."""
        from src.ranking.comparison import RankingComparator

        correlation = RankingComparator._compute_spearman([], [])

        assert correlation == 0.0


# =============================================================================
# Compare to Poll Tests
# =============================================================================


class TestCompareToPoll:
    """Tests for compare_to_poll method."""

    @pytest.mark.asyncio
    async def test_returns_comparison_result(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Returns a ComparisonResult object."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_type="ap",
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        assert isinstance(result, ComparisonResult)

    @pytest.mark.asyncio
    async def test_includes_correlation(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Result includes Spearman correlation."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_type="ap",
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        assert -1.0 <= result.spearman_correlation <= 1.0

    @pytest.mark.asyncio
    async def test_includes_teams_compared_count(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Result includes count of compared teams."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_type="ap",
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        assert result.teams_compared == 10  # All 10 teams match

    @pytest.mark.asyncio
    async def test_includes_season_and_week(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Result includes season and week."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_type="ap",
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        assert result.season == 2024
        assert result.week == 15
        assert result.poll_type == "ap"

    @pytest.mark.asyncio
    async def test_identical_rankings_high_correlation(self):
        """Identical rankings should have correlation near 1.0."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        # Create matching rankings
        our_rankings = [make_team_rating(f"team-{i}", i) for i in range(1, 11)]
        poll_rankings = [{"team_id": f"team-{i}", "rank": i} for i in range(1, 11)]

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=poll_rankings,
            our_rankings=our_rankings,
        )

        assert result.spearman_correlation == pytest.approx(1.0)


# =============================================================================
# Overranked/Underranked Tests
# =============================================================================


class TestOverrankedUnderranked:
    """Tests for identifying overranked and underranked teams."""

    @pytest.mark.asyncio
    async def test_finds_overranked_teams(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Identifies teams we rank higher than the poll."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        # Overranked: our rank < poll rank (we rank them higher)
        # team-b: we rank 2, poll ranks 3 → overranked
        # team-d: we rank 4, poll ranks 5 → overranked
        # team-g: we rank 7, poll ranks 8 → overranked
        # team-i: we rank 9, poll ranks 10 → overranked
        assert len(result.overranked) > 0

        overranked_ids = [t["team_id"] for t in result.overranked]
        assert "team-b" in overranked_ids

    @pytest.mark.asyncio
    async def test_finds_underranked_teams(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Identifies teams we rank lower than the poll."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        # Underranked: our rank > poll rank (we rank them lower)
        # team-c: we rank 3, poll ranks 2 → underranked
        # team-e: we rank 5, poll ranks 4 → underranked
        # team-h: we rank 8, poll ranks 7 → underranked
        # team-j: we rank 10, poll ranks 9 → underranked
        assert len(result.underranked) > 0

        underranked_ids = [t["team_id"] for t in result.underranked]
        assert "team-c" in underranked_ids

    @pytest.mark.asyncio
    async def test_overranked_includes_rank_details(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Overranked entries include rank details."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        if result.overranked:
            entry = result.overranked[0]
            assert "team_id" in entry
            assert "our_rank" in entry
            assert "poll_rank" in entry
            assert "difference" in entry


# =============================================================================
# Find Biggest Differences Tests
# =============================================================================


class TestFindBiggestDifferences:
    """Tests for find_biggest_differences method."""

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Returns a list of dicts with difference info."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        differences = await comparator.find_biggest_differences(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
            limit=5,
        )

        assert isinstance(differences, list)
        assert len(differences) <= 5

    @pytest.mark.asyncio
    async def test_sorted_by_absolute_difference(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Results are sorted by absolute difference (largest first)."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        differences = await comparator.find_biggest_differences(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
            limit=10,
        )

        if len(differences) > 1:
            abs_diffs = [abs(d["difference"]) for d in differences]
            assert abs_diffs == sorted(abs_diffs, reverse=True)

    @pytest.mark.asyncio
    async def test_includes_required_fields(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Each result includes required fields."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        differences = await comparator.find_biggest_differences(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
            limit=5,
        )

        for diff in differences:
            assert "team_id" in diff
            assert "our_rank" in diff
            assert "poll_rank" in diff
            assert "difference" in diff

    @pytest.mark.asyncio
    async def test_difference_calculation(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Difference = poll_rank - our_rank (negative = we rank higher)."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        differences = await comparator.find_biggest_differences(
            season=2024,
            week=15,
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
            limit=10,
        )

        for diff in differences:
            expected_diff = diff["poll_rank"] - diff["our_rank"]
            assert diff["difference"] == expected_diff

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Returns at most 'limit' results."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        # Create 20 teams with differences
        our_rankings = [make_team_rating(f"team-{i}", i) for i in range(1, 21)]
        poll_rankings = [{"team_id": f"team-{i}", "rank": 21 - i} for i in range(1, 21)]

        differences = await comparator.find_biggest_differences(
            season=2024,
            week=15,
            poll_rankings=poll_rankings,
            our_rankings=our_rankings,
            limit=5,
        )

        assert len(differences) == 5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_no_matching_teams(self):
        """Handles case where no teams match between rankings."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        our_rankings = [make_team_rating(f"team-{i}", i) for i in range(1, 6)]
        poll_rankings = [{"team_id": f"other-{i}", "rank": i} for i in range(1, 6)]

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=poll_rankings,
            our_rankings=our_rankings,
        )

        assert result.teams_compared == 0
        assert result.spearman_correlation == 0.0
        assert len(result.overranked) == 0
        assert len(result.underranked) == 0

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        """Only compares teams that appear in both rankings."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        our_rankings = [make_team_rating(f"team-{i}", i) for i in range(1, 11)]
        # Poll only has 5 matching teams
        poll_rankings = [
            {"team_id": "team-1", "rank": 1},
            {"team_id": "team-2", "rank": 2},
            {"team_id": "team-3", "rank": 3},
            {"team_id": "team-4", "rank": 4},
            {"team_id": "team-5", "rank": 5},
            {"team_id": "other-1", "rank": 6},  # Not in our rankings
            {"team_id": "other-2", "rank": 7},
        ]

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=poll_rankings,
            our_rankings=our_rankings,
        )

        assert result.teams_compared == 5

    @pytest.mark.asyncio
    async def test_empty_poll_rankings(self):
        """Handles empty poll rankings."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        our_rankings = [make_team_rating(f"team-{i}", i) for i in range(1, 6)]

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=[],
            our_rankings=our_rankings,
        )

        assert result.teams_compared == 0

    @pytest.mark.asyncio
    async def test_empty_our_rankings(self):
        """Handles empty algorithm rankings."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        poll_rankings = [{"team_id": f"team-{i}", "rank": i} for i in range(1, 6)]

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_rankings=poll_rankings,
            our_rankings=[],
        )

        assert result.teams_compared == 0

    @pytest.mark.asyncio
    async def test_handles_cfp_poll_type(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Works with CFP poll type."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_type="cfp",
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        assert result.poll_type == "cfp"

    @pytest.mark.asyncio
    async def test_handles_coaches_poll_type(
        self,
        sample_our_rankings: list[TeamRating],
        sample_poll_rankings: list[dict],
    ):
        """Works with Coaches poll type."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        result = await comparator.compare_to_poll(
            season=2024,
            week=15,
            poll_type="coaches",
            poll_rankings=sample_poll_rankings,
            our_rankings=sample_our_rankings,
        )

        assert result.poll_type == "coaches"


# =============================================================================
# Integration with RankingEngine Tests
# =============================================================================


class TestEngineIntegration:
    """Tests for integration with RankingEngine."""

    @pytest.mark.asyncio
    async def test_uses_engine_for_rankings(self, four_team_games: list[Game]):
        """Can use engine to generate rankings."""
        from src.ranking.comparison import RankingComparator
        from src.ranking.engine import RankingEngine

        engine = RankingEngine()
        comparator = RankingComparator(engine)

        # Generate rankings using engine
        our_rankings = await engine.rank_season(2024, games=four_team_games)

        # Create poll with same order
        poll_rankings = [
            {"team_id": r.team_id, "rank": r.rank}
            for r in our_rankings
        ]

        result = await comparator.compare_to_poll(
            season=2024,
            week=3,
            poll_rankings=poll_rankings,
            our_rankings=our_rankings,
        )

        # Should have perfect correlation since they match
        assert result.spearman_correlation == pytest.approx(1.0)
        assert result.teams_compared == 4
