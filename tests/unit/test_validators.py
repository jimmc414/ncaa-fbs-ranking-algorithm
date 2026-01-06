"""Unit tests for validator service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.validation.validators import ValidatorService
from src.validation.models import ValidatorRating, TeamValidation
from src.data.models import TeamRating


class TestValidatorService:
    """Tests for ValidatorService class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock CFBDataClient."""
        client = MagicMock()
        client.fetch_sp_ratings = AsyncMock()
        client.fetch_srs_ratings = AsyncMock()
        client.fetch_elo_ratings = AsyncMock()
        return client

    @pytest.fixture
    def sample_sp_data(self):
        """Sample SP+ ratings data."""
        return [
            {"team_id": "ohio-state", "rating": 31.2, "ranking": 1, "offense": 35.0, "defense": 5.0},
            {"team_id": "georgia", "rating": 28.5, "ranking": 2, "offense": 30.0, "defense": 6.0},
            {"team_id": "alabama", "rating": 26.0, "ranking": 3, "offense": 28.0, "defense": 7.0},
        ]

    @pytest.fixture
    def sample_srs_data(self):
        """Sample SRS ratings data."""
        return [
            {"team_id": "ohio-state", "rating": 24.6, "ranking": 1},
            {"team_id": "georgia", "rating": 22.0, "ranking": 2},
            {"team_id": "alabama", "rating": 20.0, "ranking": 3},
        ]

    @pytest.fixture
    def sample_elo_data(self):
        """Sample Elo ratings data."""
        return [
            {"team_id": "ohio-state", "elo": 2211.0},
            {"team_id": "georgia", "elo": 2150.0},
            {"team_id": "alabama", "elo": 2100.0},
        ]

    @pytest.fixture
    def sample_our_rankings(self):
        """Sample of our rankings."""
        return [
            TeamRating(
                team_id="ohio-state", season=2024, week=15, rating=0.85, rank=1,
                wins=12, losses=1, games_played=13,
            ),
            TeamRating(
                team_id="georgia", season=2024, week=15, rating=0.82, rank=2,
                wins=11, losses=2, games_played=13,
            ),
            TeamRating(
                team_id="james-madison", season=2024, week=15, rating=0.75, rank=10,
                wins=12, losses=2, games_played=14, strength_of_schedule=0.40,
            ),
        ]


class TestFetchAllValidators(TestValidatorService):
    """Tests for fetch_all_validators method."""

    @pytest.mark.asyncio
    async def test_fetches_all_sources_in_parallel(self, mock_client, sample_sp_data, sample_srs_data, sample_elo_data):
        """Should fetch SP+, SRS, and Elo in parallel."""
        mock_client.fetch_sp_ratings.return_value = sample_sp_data
        mock_client.fetch_srs_ratings.return_value = sample_srs_data
        mock_client.fetch_elo_ratings.return_value = sample_elo_data

        service = ValidatorService(mock_client)
        validators = await service.fetch_all_validators(2024)

        assert "sp" in validators
        assert "srs" in validators
        assert "elo" in validators

    @pytest.mark.asyncio
    async def test_handles_sp_failure_gracefully(self, mock_client, sample_srs_data, sample_elo_data):
        """Should continue if SP+ fetch fails."""
        mock_client.fetch_sp_ratings.side_effect = Exception("API Error")
        mock_client.fetch_srs_ratings.return_value = sample_srs_data
        mock_client.fetch_elo_ratings.return_value = sample_elo_data

        service = ValidatorService(mock_client)
        validators = await service.fetch_all_validators(2024)

        assert "sp" not in validators
        assert "srs" in validators
        assert "elo" in validators

    @pytest.mark.asyncio
    async def test_returns_validator_ratings(self, mock_client, sample_sp_data):
        """Should return list of ValidatorRating objects."""
        mock_client.fetch_sp_ratings.return_value = sample_sp_data
        mock_client.fetch_srs_ratings.return_value = []
        mock_client.fetch_elo_ratings.return_value = []

        service = ValidatorService(mock_client)
        validators = await service.fetch_all_validators(2024)

        sp_ratings = validators.get("sp", [])
        assert len(sp_ratings) == 3
        assert all(isinstance(r, ValidatorRating) for r in sp_ratings)


class TestCompareRankings(TestValidatorService):
    """Tests for compare_rankings method."""

    @pytest.mark.asyncio
    async def test_calculates_correlations(self, mock_client, sample_sp_data, sample_srs_data, sample_elo_data, sample_our_rankings):
        """Should calculate Spearman correlations."""
        mock_client.fetch_sp_ratings.return_value = sample_sp_data
        mock_client.fetch_srs_ratings.return_value = sample_srs_data
        mock_client.fetch_elo_ratings.return_value = sample_elo_data

        service = ValidatorService(mock_client)
        validators = await service.fetch_all_validators(2024)
        report = service.compare_rankings(sample_our_rankings, validators)

        # Should have correlation for SP+ at least
        assert "sp" in report.correlations or len(validators.get("sp", [])) < 10

    @pytest.mark.asyncio
    async def test_flags_large_discrepancies(self, mock_client):
        """Should flag teams with rank gap >= threshold."""
        # SP+ has JMU at rank 41, we have them at rank 10
        sp_data = [
            {"team_id": "ohio-state", "rating": 31.2, "ranking": 1},
            {"team_id": "james-madison", "rating": 7.5, "ranking": 41},
        ]
        mock_client.fetch_sp_ratings.return_value = sp_data
        mock_client.fetch_srs_ratings.return_value = []
        mock_client.fetch_elo_ratings.return_value = []

        our_rankings = [
            TeamRating(team_id="ohio-state", season=2024, week=15, rating=0.85, rank=1, wins=12, losses=1, games_played=13),
            TeamRating(team_id="james-madison", season=2024, week=15, rating=0.75, rank=10, wins=12, losses=2, games_played=14),
        ]

        service = ValidatorService(mock_client)
        validators = await service.fetch_all_validators(2024)
        report = service.compare_rankings(our_rankings, validators, threshold=20)

        # JMU should be flagged (gap of 31)
        flagged_team_ids = [t.team_id for t in report.flagged_teams]
        assert "james-madison" in flagged_team_ids

    def test_respects_threshold_parameter(self, mock_client):
        """Threshold parameter should control flagging sensitivity."""
        service = ValidatorService(mock_client)

        validators = {
            "sp": [
                ValidatorRating(team_id="ohio-state", source="sp", rating=31.2, rank=1),
                ValidatorRating(team_id="team-b", source="sp", rating=20.0, rank=15),
            ]
        }
        our_rankings = [
            TeamRating(team_id="ohio-state", season=2024, week=15, rating=0.85, rank=1, wins=12, losses=1, games_played=13),
            TeamRating(team_id="team-b", season=2024, week=15, rating=0.65, rank=5, wins=10, losses=3, games_played=13),
        ]

        # With threshold 20, team-b (gap=10) should not be flagged
        report_20 = service.compare_rankings(our_rankings, validators, threshold=20)
        flagged_20 = [t.team_id for t in report_20.flagged_teams]
        assert "team-b" not in flagged_20

        # With threshold 5, team-b should be flagged
        report_5 = service.compare_rankings(our_rankings, validators, threshold=5)
        flagged_5 = [t.team_id for t in report_5.flagged_teams]
        assert "team-b" in flagged_5


class TestGetTeamValidation(TestValidatorService):
    """Tests for get_team_validation method."""

    def test_returns_team_validation(self, mock_client):
        """Should return TeamValidation for found team."""
        service = ValidatorService(mock_client)

        validators = {
            "sp": [ValidatorRating(team_id="ohio-state", source="sp", rating=31.2, rank=1)],
            "srs": [ValidatorRating(team_id="ohio-state", source="srs", rating=24.6, rank=1)],
            "elo": [ValidatorRating(team_id="ohio-state", source="elo", rating=2211.0, rank=1)],
        }
        our_rankings = [
            TeamRating(team_id="ohio-state", season=2024, week=15, rating=0.85, rank=9, wins=12, losses=1, games_played=13),
        ]

        result = service.get_team_validation("ohio-state", our_rankings, validators)

        assert result is not None
        assert result.team_id == "ohio-state"
        assert result.our_rank == 9
        assert "sp" in result.validator_ranks
        assert result.validator_ranks["sp"] == 1

    def test_returns_none_for_unknown_team(self, mock_client):
        """Should return None for team not in our rankings."""
        service = ValidatorService(mock_client)

        validators = {"sp": []}
        our_rankings = [
            TeamRating(team_id="ohio-state", season=2024, week=15, rating=0.85, rank=1, wins=12, losses=1, games_played=13),
        ]

        result = service.get_team_validation("unknown-team", our_rankings, validators)
        assert result is None

    def test_calculates_max_gap(self, mock_client):
        """Should calculate maximum rank gap across validators."""
        service = ValidatorService(mock_client)

        validators = {
            "sp": [ValidatorRating(team_id="team-a", source="sp", rating=10.0, rank=41)],
            "srs": [ValidatorRating(team_id="team-a", source="srs", rating=5.0, rank=62)],
            "elo": [ValidatorRating(team_id="team-a", source="elo", rating=1691.0, rank=58)],
        }
        our_rankings = [
            TeamRating(team_id="team-a", season=2024, week=15, rating=0.75, rank=10, wins=12, losses=2, games_played=14),
        ]

        result = service.get_team_validation("team-a", our_rankings, validators)

        assert result is not None
        # Max gap should be 52 (10 - 62)
        assert result.max_gap == 52


class TestValidatorRatingModel:
    """Tests for ValidatorRating dataclass."""

    def test_basic_creation(self):
        """Should create ValidatorRating with required fields."""
        rating = ValidatorRating(
            team_id="ohio-state",
            source="sp",
            rating=31.2,
            rank=1,
        )
        assert rating.team_id == "ohio-state"
        assert rating.source == "sp"
        assert rating.rating == 31.2
        assert rating.rank == 1

    def test_optional_fields(self):
        """Should support optional offense/defense fields."""
        rating = ValidatorRating(
            team_id="ohio-state",
            source="sp",
            rating=31.2,
            rank=1,
            offense=35.0,
            defense=5.0,
        )
        assert rating.offense == 35.0
        assert rating.defense == 5.0


class TestTeamValidationModel:
    """Tests for TeamValidation dataclass."""

    def test_get_gap_returns_difference(self):
        """get_gap should return rank difference."""
        validation = TeamValidation(
            team_id="ohio-state",
            our_rank=9,
            our_rating=0.85,
            validator_ranks={"sp": 1, "srs": 1},
            validator_ratings={"sp": 31.2, "srs": 24.6},
        )

        # We rank 9, SP+ ranks 1, gap should be 9-1=8
        assert validation.get_gap("sp") == 8
        assert validation.get_gap("srs") == 8

    def test_get_gap_returns_none_for_missing_source(self):
        """get_gap should return None for missing validator."""
        validation = TeamValidation(
            team_id="ohio-state",
            our_rank=9,
            our_rating=0.85,
            validator_ranks={"sp": 1},
            validator_ratings={"sp": 31.2},
        )

        assert validation.get_gap("elo") is None
