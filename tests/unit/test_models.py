"""Tests for Pydantic data models."""

import pytest
from datetime import date
from pydantic import ValidationError


class TestTeamModel:
    """Tests for the Team model."""

    def test_team_valid_fbs(self):
        """Team with FBS division validates correctly."""
        from src.data.models import Team

        team = Team(
            team_id="ohio-state",
            name="Ohio State Buckeyes",
            conference="Big Ten",
            division="fbs",
        )
        assert team.team_id == "ohio-state"
        assert team.name == "Ohio State Buckeyes"
        assert team.conference == "Big Ten"
        assert team.division == "fbs"

    def test_team_valid_fcs(self):
        """Team with FCS division validates correctly."""
        from src.data.models import Team

        team = Team(
            team_id="north-dakota-state",
            name="North Dakota State Bison",
            conference="Missouri Valley",
            division="fcs",
        )
        assert team.division == "fcs"

    def test_team_invalid_division(self):
        """Team with invalid division raises ValidationError."""
        from src.data.models import Team

        with pytest.raises(ValidationError) as exc_info:
            Team(
                team_id="test",
                name="Test Team",
                conference="Test",
                division="invalid",
            )
        assert "division" in str(exc_info.value)

    def test_team_optional_conference(self):
        """Team without conference is valid (independents)."""
        from src.data.models import Team

        team = Team(
            team_id="notre-dame",
            name="Notre Dame Fighting Irish",
            conference=None,
            division="fbs",
        )
        assert team.conference is None

    def test_team_serialization(self):
        """Team can be serialized to dict and back."""
        from src.data.models import Team

        team = Team(
            team_id="alabama",
            name="Alabama Crimson Tide",
            conference="SEC",
            division="fbs",
        )
        data = team.model_dump()
        team2 = Team(**data)
        assert team == team2


class TestGameModel:
    """Tests for the Game model."""

    def test_game_valid(self):
        """Valid game data creates Game instance."""
        from src.data.models import Game

        game = Game(
            game_id=401628456,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="ohio-state",
            away_team_id="akron",
            home_score=52,
            away_score=6,
            neutral_site=False,
            postseason=False,
        )
        assert game.game_id == 401628456
        assert game.season == 2024
        assert game.home_score == 52
        assert game.away_score == 6

    def test_game_negative_home_score_rejected(self):
        """Negative home score raises ValidationError."""
        from src.data.models import Game

        with pytest.raises(ValidationError) as exc_info:
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="team-a",
                away_team_id="team-b",
                home_score=-1,
                away_score=14,
            )
        assert "home_score" in str(exc_info.value)

    def test_game_negative_away_score_rejected(self):
        """Negative away score raises ValidationError."""
        from src.data.models import Game

        with pytest.raises(ValidationError) as exc_info:
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="team-a",
                away_team_id="team-b",
                home_score=21,
                away_score=-7,
            )
        assert "away_score" in str(exc_info.value)

    def test_game_same_team_rejected(self):
        """Home and away same team raises ValidationError."""
        from src.data.models import Game

        with pytest.raises(ValidationError) as exc_info:
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="ohio-state",
                away_team_id="ohio-state",
                home_score=21,
                away_score=14,
            )
        assert "must be different" in str(exc_info.value).lower()

    def test_game_defaults(self):
        """Game has correct default values."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="team-a",
            away_team_id="team-b",
            home_score=21,
            away_score=14,
        )
        assert game.neutral_site is False
        assert game.postseason is False

    def test_game_neutral_site(self):
        """Neutral site game is valid."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="georgia",
            away_team_id="clemson",
            home_score=24,
            away_score=21,
            neutral_site=True,
        )
        assert game.neutral_site is True

    def test_game_postseason(self):
        """Postseason game is valid."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=16,
            game_date=date(2025, 1, 1),
            home_team_id="team-a",
            away_team_id="team-b",
            home_score=35,
            away_score=28,
            postseason=True,
        )
        assert game.postseason is True


class TestGameResultDerivation:
    """Tests for GameResult derived from Game."""

    def test_game_result_from_home_perspective_win(self):
        """GameResult correctly computed for home team win."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="ohio-state",
            away_team_id="akron",
            home_score=52,
            away_score=6,
        )
        home_result, away_result = game.to_results()

        assert home_result.team_id == "ohio-state"
        assert home_result.opponent_id == "akron"
        assert home_result.team_score == 52
        assert home_result.opponent_score == 6
        assert home_result.location == "home"
        assert home_result.is_win is True
        assert home_result.margin == 46

    def test_game_result_from_away_perspective_loss(self):
        """GameResult correctly computed for away team loss."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="ohio-state",
            away_team_id="akron",
            home_score=52,
            away_score=6,
        )
        home_result, away_result = game.to_results()

        assert away_result.team_id == "akron"
        assert away_result.opponent_id == "ohio-state"
        assert away_result.team_score == 6
        assert away_result.opponent_score == 52
        assert away_result.location == "away"
        assert away_result.is_win is False
        assert away_result.margin == -46

    def test_game_result_neutral_site(self):
        """Neutral site games report 'neutral' location for both teams."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="georgia",
            away_team_id="clemson",
            home_score=24,
            away_score=21,
            neutral_site=True,
        )
        home_result, away_result = game.to_results()

        assert home_result.location == "neutral"
        assert away_result.location == "neutral"

    def test_game_result_away_team_wins(self):
        """Away team win correctly computed."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="team-a",
            away_team_id="team-b",
            home_score=14,
            away_score=21,
        )
        home_result, away_result = game.to_results()

        assert home_result.is_win is False
        assert home_result.margin == -7
        assert away_result.is_win is True
        assert away_result.margin == 7

    def test_game_result_tie(self):
        """Tie game (historical) handled correctly."""
        from src.data.models import Game

        game = Game(
            game_id=1,
            season=1970,
            week=1,
            game_date=date(1970, 9, 1),
            home_team_id="team-a",
            away_team_id="team-b",
            home_score=14,
            away_score=14,
        )
        home_result, away_result = game.to_results()

        assert home_result.is_win is False
        assert home_result.margin == 0
        assert away_result.is_win is False
        assert away_result.margin == 0


class TestGameResultModel:
    """Tests for the GameResult model directly."""

    def test_game_result_valid(self):
        """Valid GameResult creates instance."""
        from src.data.models import GameResult

        result = GameResult(
            game_id=1,
            team_id="ohio-state",
            opponent_id="michigan",
            team_score=42,
            opponent_score=27,
            location="home",
            is_win=True,
            margin=15,
        )
        assert result.is_win is True
        assert result.margin == 15

    def test_game_result_invalid_location(self):
        """Invalid location raises ValidationError."""
        from src.data.models import GameResult

        with pytest.raises(ValidationError) as exc_info:
            GameResult(
                game_id=1,
                team_id="ohio-state",
                opponent_id="michigan",
                team_score=42,
                opponent_score=27,
                location="invalid",
                is_win=True,
                margin=15,
            )
        assert "location" in str(exc_info.value)


class TestTeamRatingModel:
    """Tests for the TeamRating model."""

    def test_team_rating_valid(self):
        """Valid TeamRating creates instance."""
        from src.data.models import TeamRating

        rating = TeamRating(
            team_id="georgia",
            season=2024,
            week=12,
            rating=0.9432,
            rank=1,
            games_played=11,
            wins=11,
            losses=0,
            strength_of_schedule=0.71,
            strength_of_victory=0.68,
            average_game_grade=0.89,
        )
        assert rating.rating == 0.9432
        assert rating.rank == 1

    def test_team_rating_optional_fields(self):
        """Optional fields can be None."""
        from src.data.models import TeamRating

        rating = TeamRating(
            team_id="georgia",
            season=2024,
            week=12,
            rating=0.9432,
            rank=1,
            games_played=11,
            wins=11,
            losses=0,
        )
        assert rating.strength_of_schedule is None
        assert rating.strength_of_victory is None
        assert rating.average_game_grade is None


class TestAlgorithmConfig:
    """Tests for the AlgorithmConfig model."""

    def test_config_defaults(self):
        """Default config matches spec values."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.max_iterations == 100
        assert config.convergence_threshold == 0.0001
        assert config.win_base == 0.70
        assert config.margin_weight == 0.20
        assert config.margin_cap == 28
        assert config.venue_road_win == 0.10
        assert config.venue_neutral_win == 0.05
        assert config.venue_home_loss == -0.03
        assert config.venue_neutral_loss == -0.01
        assert config.initial_rating == 0.500
        assert config.fcs_fixed_rating == 0.20

    def test_config_custom_values(self):
        """Custom config values accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(
            max_iterations=50,
            convergence_threshold=0.001,
            fcs_fixed_rating=None,
        )
        assert config.max_iterations == 50
        assert config.convergence_threshold == 0.001
        assert config.fcs_fixed_rating is None

    def test_config_validation_positive_iterations(self):
        """Max iterations must be positive."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(max_iterations=0)

    def test_config_validation_positive_threshold(self):
        """Convergence threshold must be positive."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(convergence_threshold=0)

    def test_config_validation_margin_cap_positive(self):
        """Margin cap must be positive."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(margin_cap=0)
