"""Integration tests for the FastAPI web dashboard.

Tests the web routes and HTMX partial updates.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.data.models import (
    ComparisonResult,
    RatingExplanation,
    TeamRating,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_team_rating(
    team_id: str,
    rank: int,
    rating: float = 0.5,
    wins: int = 10,
    losses: int = 2,
) -> TeamRating:
    """Helper to create test TeamRating."""
    return TeamRating(
        team_id=team_id,
        season=2024,
        week=15,
        rating=rating,
        rank=rank,
        games_played=wins + losses,
        wins=wins,
        losses=losses,
        strength_of_schedule=0.5,
        strength_of_victory=0.4,
    )


@pytest.fixture
def sample_rankings() -> list[TeamRating]:
    """Sample rankings for testing."""
    return [
        make_team_rating("ohio-state", 1, 0.95, 12, 0),
        make_team_rating("georgia", 2, 0.92, 11, 1),
        make_team_rating("michigan", 3, 0.89, 11, 1),
        make_team_rating("alabama", 4, 0.86, 10, 2),
        make_team_rating("texas", 5, 0.83, 10, 2),
    ]


@pytest.fixture
def sample_explanation() -> RatingExplanation:
    """Sample rating explanation."""
    return RatingExplanation(
        team_id="ohio-state",
        season=2024,
        week=15,
        final_rating=0.95,
        normalized_rating=1.0,
        rank=1,
        games=[
            {
                "game_id": 1,
                "opponent_id": "michigan",
                "is_win": True,
                "margin": 14,
                "location": "home",
                "game_grade": 0.85,
                "opponent_rating": 0.89,
                "contribution": 1.74,
            },
            {
                "game_id": 2,
                "opponent_id": "penn-state",
                "is_win": True,
                "margin": 21,
                "location": "away",
                "game_grade": 0.92,
                "opponent_rating": 0.75,
                "contribution": 1.67,
            },
        ],
        total_contribution=3.41,
        iterations_to_converge=44,
    )


@pytest.fixture
def sample_comparison() -> ComparisonResult:
    """Sample comparison result."""
    return ComparisonResult(
        season=2024,
        week=15,
        poll_type="ap",
        spearman_correlation=0.85,
        teams_compared=25,
        overranked=[
            {"team_id": "ohio-state", "our_rank": 1, "poll_rank": 3, "difference": 2},
        ],
        underranked=[
            {"team_id": "georgia", "our_rank": 4, "poll_rank": 1, "difference": -3},
        ],
    )


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    from src.web.app import app
    return TestClient(app)


# =============================================================================
# Home Page Tests
# =============================================================================


class TestHomePage:
    """Tests for home page."""

    def test_home_redirects_to_rankings(self, client: TestClient):
        """Home page redirects to rankings."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (301, 302, 307, 308)
        assert "/rankings/" in response.headers.get("location", "")


# =============================================================================
# Rankings Page Tests
# =============================================================================


class TestRankingsPage:
    """Tests for rankings page."""

    def test_rankings_page_renders(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Rankings page returns 200."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_rankings_page_shows_teams(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Rankings page shows team names."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024")

            assert "ohio-state" in response.text.lower()
            assert "georgia" in response.text.lower()

    def test_rankings_page_with_week(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Rankings page accepts week parameter."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024?week=12")

            assert response.status_code == 200
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("week") == 12

    def test_rankings_page_shows_week_selector(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Rankings page has week selector."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024")

            assert "select" in response.text.lower()


# =============================================================================
# Rankings Table Partial Tests
# =============================================================================


class TestRankingsTablePartial:
    """Tests for HTMX partial updates."""

    def test_rankings_table_partial_returns_html(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Table partial returns HTML fragment."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024/table")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_rankings_table_partial_shows_teams(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Table partial shows team names."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024/table")

            assert "ohio-state" in response.text.lower()

    def test_rankings_table_partial_with_week(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """Table partial accepts week parameter."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024/table?week=10")

            assert response.status_code == 200
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("week") == 10


# =============================================================================
# Team Detail Page Tests
# =============================================================================


class TestTeamDetailPage:
    """Tests for team detail page."""

    def test_team_detail_page_renders(
        self, client: TestClient, sample_explanation: RatingExplanation
    ):
        """Team detail page returns 200."""
        with patch("src.web.app.get_team_detail") as mock_get:
            mock_get.return_value = sample_explanation

            response = client.get("/teams/ohio-state/2024")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_team_detail_page_shows_team_name(
        self, client: TestClient, sample_explanation: RatingExplanation
    ):
        """Team detail page shows team name."""
        with patch("src.web.app.get_team_detail") as mock_get:
            mock_get.return_value = sample_explanation

            response = client.get("/teams/ohio-state/2024")

            assert "ohio-state" in response.text.lower()

    def test_team_detail_page_shows_games(
        self, client: TestClient, sample_explanation: RatingExplanation
    ):
        """Team detail page shows game breakdown."""
        with patch("src.web.app.get_team_detail") as mock_get:
            mock_get.return_value = sample_explanation

            response = client.get("/teams/ohio-state/2024")

            assert "michigan" in response.text.lower()

    def test_team_detail_page_shows_rank(
        self, client: TestClient, sample_explanation: RatingExplanation
    ):
        """Team detail page shows rank."""
        with patch("src.web.app.get_team_detail") as mock_get:
            mock_get.return_value = sample_explanation

            response = client.get("/teams/ohio-state/2024")

            # Should show rank #1 or "1"
            assert "#1" in response.text or "rank" in response.text.lower()

    def test_team_detail_page_handles_unknown_team(self, client: TestClient):
        """Team detail page handles unknown team."""
        with patch("src.web.app.get_team_detail") as mock_get:
            mock_get.return_value = None

            response = client.get("/teams/unknown-team/2024")

            assert response.status_code == 200
            assert "not found" in response.text.lower() or "no data" in response.text.lower()


# =============================================================================
# Compare Page Tests
# =============================================================================


class TestComparePage:
    """Tests for comparison page."""

    def test_compare_page_renders(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare page returns 200."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_compare_page_shows_correlation(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare page shows Spearman correlation."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024")

            # Should show correlation (0.85 = 85%)
            assert "85" in response.text or "0.85" in response.text

    def test_compare_page_shows_overranked(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare page shows overranked teams."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024")

            assert "ohio-state" in response.text.lower()

    def test_compare_page_shows_underranked(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare page shows underranked teams."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024")

            assert "georgia" in response.text.lower()

    def test_compare_page_with_poll_option(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare page accepts poll parameter."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024?poll=cfp")

            assert response.status_code == 200
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("poll") == "cfp"

    def test_compare_page_with_week_option(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare page accepts week parameter."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024?week=14")

            assert response.status_code == 200
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("week") == 14


# =============================================================================
# Compare Table Partial Tests
# =============================================================================


class TestCompareTablePartial:
    """Tests for comparison HTMX partial."""

    def test_compare_table_partial_renders(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare table partial returns HTML."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024/table")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_compare_table_partial_shows_data(
        self, client: TestClient, sample_comparison: ComparisonResult
    ):
        """Compare table partial shows comparison data."""
        with patch("src.web.app.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            response = client.get("/compare/2024/table?poll=ap")

            assert "ohio-state" in response.text.lower() or "georgia" in response.text.lower()


# =============================================================================
# Static Assets Tests
# =============================================================================


class TestStaticAssets:
    """Tests for static files and assets."""

    def test_htmx_script_included(
        self, client: TestClient, sample_rankings: list[TeamRating]
    ):
        """HTMX script is included in pages."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            response = client.get("/rankings/2024")

            assert "htmx" in response.text.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_season_format(self, client: TestClient):
        """Invalid season returns 422."""
        response = client.get("/rankings/invalid")
        assert response.status_code == 422

    def test_empty_rankings_handled(self, client: TestClient):
        """Empty rankings are handled gracefully."""
        with patch("src.web.app.get_rankings") as mock_get:
            mock_get.return_value = []

            response = client.get("/rankings/2024")

            assert response.status_code == 200
