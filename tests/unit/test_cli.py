"""Unit tests for the CLI interface.

Tests the Typer CLI commands for ranking, explaining,
comparing, and exporting rankings.
"""

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from src.data.models import (
    ComparisonResult,
    Game,
    RatingExplanation,
    TeamRating,
)


# =============================================================================
# Test Fixtures
# =============================================================================


runner = CliRunner()


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


# =============================================================================
# Rank Command Tests
# =============================================================================


class TestRankCommand:
    """Tests for the rank command."""

    def test_rank_displays_table(self, sample_rankings: list[TeamRating]):
        """Rank command displays a table of rankings."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            result = runner.invoke(app, ["rank", "2024"])

            assert result.exit_code == 0
            assert "ohio-state" in result.output
            assert "georgia" in result.output

    def test_rank_with_week(self, sample_rankings: list[TeamRating]):
        """Rank command accepts --week option."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            result = runner.invoke(app, ["rank", "2024", "--week", "12"])

            assert result.exit_code == 0
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("week") == 12

    def test_rank_with_top_limit(self, sample_rankings: list[TeamRating]):
        """Rank command accepts --top option to limit results."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings[:3]

            result = runner.invoke(app, ["rank", "2024", "--top", "3"])

            assert result.exit_code == 0
            assert "ohio-state" in result.output
            assert "michigan" in result.output

    def test_rank_shows_record(self, sample_rankings: list[TeamRating]):
        """Rank command shows win-loss record."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            result = runner.invoke(app, ["rank", "2024"])

            assert result.exit_code == 0
            # Should show record like "12-0" or "11-1"
            assert "12-0" in result.output or "12" in result.output

    def test_rank_handles_empty_results(self):
        """Rank command handles no results gracefully."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = []

            result = runner.invoke(app, ["rank", "2024"])

            assert result.exit_code == 0
            assert "No rankings" in result.output or "no rankings" in result.output.lower()


# =============================================================================
# Explain Command Tests
# =============================================================================


class TestExplainCommand:
    """Tests for the explain command."""

    def test_explain_shows_rating_breakdown(self, sample_explanation: RatingExplanation):
        """Explain command shows rating breakdown."""
        from src.cli.main import app

        with patch("src.cli.main.get_explanation") as mock_get:
            mock_get.return_value = sample_explanation

            result = runner.invoke(app, ["explain", "2024", "ohio-state"])

            assert result.exit_code == 0
            assert "ohio-state" in result.output.lower()

    def test_explain_shows_games(self, sample_explanation: RatingExplanation):
        """Explain command shows game-by-game contributions."""
        from src.cli.main import app

        with patch("src.cli.main.get_explanation") as mock_get:
            mock_get.return_value = sample_explanation

            result = runner.invoke(app, ["explain", "2024", "ohio-state"])

            assert result.exit_code == 0
            # Should show opponent names
            assert "michigan" in result.output.lower()

    def test_explain_with_week(self, sample_explanation: RatingExplanation):
        """Explain command accepts --week option."""
        from src.cli.main import app

        with patch("src.cli.main.get_explanation") as mock_get:
            mock_get.return_value = sample_explanation

            result = runner.invoke(app, ["explain", "2024", "ohio-state", "--week", "12"])

            assert result.exit_code == 0
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("week") == 12

    def test_explain_shows_rank(self, sample_explanation: RatingExplanation):
        """Explain command shows team's rank."""
        from src.cli.main import app

        with patch("src.cli.main.get_explanation") as mock_get:
            mock_get.return_value = sample_explanation

            result = runner.invoke(app, ["explain", "2024", "ohio-state"])

            assert result.exit_code == 0
            assert "#1" in result.output or "Rank: 1" in result.output or "rank" in result.output.lower()

    def test_explain_handles_unknown_team(self):
        """Explain command handles unknown team gracefully."""
        from src.cli.main import app

        with patch("src.cli.main.get_explanation") as mock_get:
            mock_get.return_value = None

            result = runner.invoke(app, ["explain", "2024", "unknown-team"])

            assert result.exit_code == 0
            assert "not found" in result.output.lower() or "no data" in result.output.lower()


# =============================================================================
# Compare Command Tests
# =============================================================================


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_shows_correlation(self, sample_comparison: ComparisonResult):
        """Compare command shows Spearman correlation."""
        from src.cli.main import app

        with patch("src.cli.main.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            result = runner.invoke(app, ["compare", "2024"])

            assert result.exit_code == 0
            assert "0.85" in result.output or "85" in result.output

    def test_compare_shows_overranked(self, sample_comparison: ComparisonResult):
        """Compare command shows overranked teams."""
        from src.cli.main import app

        with patch("src.cli.main.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            result = runner.invoke(app, ["compare", "2024"])

            assert result.exit_code == 0
            assert "ohio-state" in result.output.lower()

    def test_compare_shows_underranked(self, sample_comparison: ComparisonResult):
        """Compare command shows underranked teams."""
        from src.cli.main import app

        with patch("src.cli.main.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            result = runner.invoke(app, ["compare", "2024"])

            assert result.exit_code == 0
            assert "georgia" in result.output.lower()

    def test_compare_with_poll_option(self, sample_comparison: ComparisonResult):
        """Compare command accepts --poll option."""
        from src.cli.main import app

        with patch("src.cli.main.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            result = runner.invoke(app, ["compare", "2024", "--poll", "cfp"])

            assert result.exit_code == 0
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("poll_type") == "cfp"

    def test_compare_with_week_option(self, sample_comparison: ComparisonResult):
        """Compare command accepts --week option."""
        from src.cli.main import app

        with patch("src.cli.main.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            result = runner.invoke(app, ["compare", "2024", "--week", "14"])

            assert result.exit_code == 0
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1].get("week") == 14


# =============================================================================
# Export Command Tests
# =============================================================================


class TestExportCommand:
    """Tests for the export command."""

    def test_export_csv(self, sample_rankings: list[TeamRating]):
        """Export command creates CSV file."""
        from src.cli.main import app

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rankings.csv"

            with patch("src.cli.main.get_rankings") as mock_get:
                mock_get.return_value = sample_rankings

                result = runner.invoke(
                    app, ["export", "2024", "--format", "csv", "--output", str(output_path)]
                )

                assert result.exit_code == 0
                assert output_path.exists()
                content = output_path.read_text()
                assert "ohio-state" in content
                assert "georgia" in content

    def test_export_json(self, sample_rankings: list[TeamRating]):
        """Export command creates JSON file."""
        from src.cli.main import app

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rankings.json"

            with patch("src.cli.main.get_rankings") as mock_get:
                mock_get.return_value = sample_rankings

                result = runner.invoke(
                    app, ["export", "2024", "--format", "json", "--output", str(output_path)]
                )

                assert result.exit_code == 0
                assert output_path.exists()
                data = json.loads(output_path.read_text())
                assert isinstance(data, list)
                assert len(data) == 5

    def test_export_default_filename(self, sample_rankings: list[TeamRating]):
        """Export command uses default filename if not specified."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            # Use temp directory as working directory
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("src.cli.main.get_default_output_path") as mock_path:
                    output_path = Path(tmpdir) / "rankings_2024.csv"
                    mock_path.return_value = output_path

                    result = runner.invoke(app, ["export", "2024", "--format", "csv"])

                    assert result.exit_code == 0

    def test_export_with_week(self, sample_rankings: list[TeamRating]):
        """Export command accepts --week option."""
        from src.cli.main import app

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rankings.csv"

            with patch("src.cli.main.get_rankings") as mock_get:
                mock_get.return_value = sample_rankings

                result = runner.invoke(
                    app,
                    ["export", "2024", "--week", "12", "--format", "csv", "--output", str(output_path)],
                )

                assert result.exit_code == 0
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert call_args[1].get("week") == 12


# =============================================================================
# Help and Version Tests
# =============================================================================


class TestHelpAndVersion:
    """Tests for help and version display."""

    def test_help_display(self):
        """--help shows usage information."""
        from src.cli.main import app

        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "rank" in result.output
        assert "explain" in result.output
        assert "compare" in result.output
        assert "export" in result.output

    def test_rank_help(self):
        """rank --help shows command help."""
        from src.cli.main import app

        result = runner.invoke(app, ["rank", "--help"])

        assert result.exit_code == 0
        assert "season" in result.output.lower()

    def test_explain_help(self):
        """explain --help shows command help."""
        from src.cli.main import app

        result = runner.invoke(app, ["explain", "--help"])

        assert result.exit_code == 0
        assert "team" in result.output.lower()

    def test_compare_help(self):
        """compare --help shows command help."""
        from src.cli.main import app

        result = runner.invoke(app, ["compare", "--help"])

        assert result.exit_code == 0
        assert "poll" in result.output.lower()

    def test_export_help(self):
        """export --help shows command help."""
        from src.cli.main import app

        result = runner.invoke(app, ["export", "--help"])

        assert result.exit_code == 0
        assert "format" in result.output.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_season(self):
        """Invalid season shows error."""
        from src.cli.main import app

        result = runner.invoke(app, ["rank", "invalid"])

        # Typer should catch this
        assert result.exit_code != 0

    def test_invalid_poll_type(self, sample_comparison: ComparisonResult):
        """Invalid poll type is handled."""
        from src.cli.main import app

        with patch("src.cli.main.get_comparison") as mock_get:
            mock_get.return_value = sample_comparison

            # Should accept any string, validation happens in comparator
            result = runner.invoke(app, ["compare", "2024", "--poll", "invalid"])

            # Should still work (comparator handles validation)
            assert result.exit_code == 0

    def test_invalid_export_format(self, sample_rankings: list[TeamRating]):
        """Invalid export format shows error."""
        from src.cli.main import app

        with patch("src.cli.main.get_rankings") as mock_get:
            mock_get.return_value = sample_rankings

            result = runner.invoke(app, ["export", "2024", "--format", "invalid"])

            # Should fail or show error
            assert result.exit_code != 0 or "invalid" in result.output.lower()
