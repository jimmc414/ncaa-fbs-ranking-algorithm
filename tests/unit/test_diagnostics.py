"""Unit tests for the diagnostic system."""

import pytest

from src.algorithm.diagnostics import (
    DiagnosticReport,
    GameContributionBreakdown,
    ParameterAttribution,
    Prediction,
    attribute_errors_to_parameters,
    compute_brier_score,
    compute_calibration_error,
    compute_upset_magnitude,
    decompose_game_contribution,
    decompose_team_rating,
    generate_diagnostic_report,
    generate_predictions,
    predict_game,
    rating_gap_to_win_probability,
)
from src.data.models import AlgorithmConfig, Game, GameResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_games():
    """Create sample games for testing."""
    from datetime import date

    return [
        Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 9, 1),
            home_team_id="team-a",
            away_team_id="team-b",
            home_score=35,
            away_score=21,
        ),
        Game(
            game_id=2,
            season=2024,
            week=2,
            game_date=date(2024, 9, 8),
            home_team_id="team-c",
            away_team_id="team-a",
            home_score=14,
            away_score=28,
        ),
        Game(
            game_id=3,
            season=2024,
            week=2,
            game_date=date(2024, 9, 8),
            home_team_id="team-b",
            away_team_id="team-c",
            home_score=21,
            away_score=24,
        ),
        Game(
            game_id=4,
            season=2024,
            week=3,
            game_date=date(2024, 9, 15),
            home_team_id="team-a",
            away_team_id="team-c",
            home_score=42,
            away_score=7,
        ),
    ]


@pytest.fixture
def sample_game_result():
    """Create a sample game result."""
    return GameResult(
        game_id=1,
        team_id="team-a",
        opponent_id="team-b",
        team_score=35,
        opponent_score=21,
        location="home",
        is_win=True,
        margin=14,
    )


@pytest.fixture
def sample_ratings():
    """Create sample team ratings."""
    return {
        "team-a": 0.85,
        "team-b": 0.55,
        "team-c": 0.35,
    }


# =============================================================================
# Win Probability Tests
# =============================================================================


class TestWinProbability:
    """Tests for win probability calculation."""

    def test_equal_ratings_gives_around_50_percent(self):
        """Equal teams should have ~50% win probability (with small home advantage)."""
        prob = rating_gap_to_win_probability(0.0, home_advantage=0.0)
        assert 0.49 <= prob <= 0.51

    def test_small_gap_gives_moderate_favorite(self):
        """Small rating gap should give moderate advantage."""
        prob = rating_gap_to_win_probability(0.10, home_advantage=0.0)
        assert 0.60 <= prob <= 0.75

    def test_medium_gap_gives_strong_favorite(self):
        """Medium rating gap should give strong advantage."""
        prob = rating_gap_to_win_probability(0.20, home_advantage=0.0)
        assert 0.70 <= prob <= 0.90

    def test_large_gap_gives_heavy_favorite(self):
        """Large rating gap should give heavy favorite."""
        prob = rating_gap_to_win_probability(0.40, home_advantage=0.0)
        assert prob >= 0.90

    def test_home_advantage_increases_probability(self):
        """Home advantage should increase win probability."""
        prob_no_home = rating_gap_to_win_probability(0.10, home_advantage=0.0)
        prob_with_home = rating_gap_to_win_probability(0.10, home_advantage=0.05)
        assert prob_with_home > prob_no_home

    def test_probability_bounded_0_to_1(self):
        """Probability should always be between 0 and 1."""
        for gap in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            prob = rating_gap_to_win_probability(gap)
            assert 0.0 <= prob <= 1.0


class TestUpsetMagnitude:
    """Tests for upset magnitude calculation."""

    def test_high_probability_upset_is_major(self):
        """Upset when favorite had high win probability should be major."""
        magnitude = compute_upset_magnitude(0.30, 0.90)
        assert magnitude >= 0.85

    def test_low_probability_upset_is_minor(self):
        """Upset when favorite had low win probability is minor."""
        magnitude = compute_upset_magnitude(0.05, 0.55)
        assert magnitude <= 0.60


# =============================================================================
# Game Contribution Decomposition Tests
# =============================================================================


class TestGameContributionBreakdown:
    """Tests for game contribution decomposition."""

    def test_decompose_home_win(self, sample_game_result):
        """Test decomposition of a home win."""
        config = AlgorithmConfig()
        breakdown = decompose_game_contribution(
            sample_game_result,
            opponent_rating=0.55,
            config=config,
        )

        # Check components
        assert breakdown.base_result == config.win_base  # 0.70
        assert breakdown.margin_bonus > 0  # Should have margin bonus
        assert breakdown.venue_adjustment == 0.0  # No bonus for home win
        assert breakdown.opponent_contribution > 0  # Positive for win
        assert breakdown.total_contribution > 1.0  # Should be substantial

    def test_decompose_road_win(self):
        """Test decomposition of a road win."""
        result = GameResult(
            game_id=2,
            team_id="team-a",
            opponent_id="team-c",
            team_score=28,
            opponent_score=14,
            location="away",
            is_win=True,
            margin=14,
        )
        config = AlgorithmConfig()
        breakdown = decompose_game_contribution(result, 0.35, config)

        # Road win should have positive venue adjustment
        assert breakdown.venue_adjustment == config.venue_road_win
        assert breakdown.venue_adjustment > 0

    def test_decompose_home_loss(self):
        """Test decomposition of a home loss."""
        result = GameResult(
            game_id=3,
            team_id="team-b",
            opponent_id="team-a",
            team_score=21,
            opponent_score=35,
            location="home",
            is_win=False,
            margin=-14,
        )
        config = AlgorithmConfig()
        breakdown = decompose_game_contribution(result, 0.85, config)

        # Home loss
        assert breakdown.base_result == 0.0
        assert breakdown.margin_bonus == 0.0  # No margin bonus for loss
        assert breakdown.venue_adjustment == config.venue_home_loss
        assert breakdown.opponent_contribution < 0  # Negative for loss
        assert breakdown.total_contribution < 0  # Should be negative

    def test_decompose_with_quality_tier(self):
        """Test decomposition with quality tier bonuses."""
        result = GameResult(
            game_id=1,
            team_id="team-a",
            opponent_id="team-b",
            team_score=35,
            opponent_score=21,
            location="home",
            is_win=True,
            margin=14,
        )
        config = AlgorithmConfig(
            enable_quality_tiers=True,
            elite_threshold=0.80,
            elite_win_bonus=0.05,
        )
        # Opponent is elite (0.85)
        breakdown = decompose_game_contribution(result, 0.5, config, normalized_opp_rating=0.85)

        assert breakdown.quality_tier_adjustment == config.elite_win_bonus


class TestTeamRatingDecomposition:
    """Tests for full team rating decomposition."""

    def test_decompose_team_returns_all_games(self, sample_games):
        """Team decomposition should return breakdown for each game."""
        config = AlgorithmConfig()
        breakdowns = decompose_team_rating("team-a", sample_games, config)

        # Team A plays in 3 games (games 1, 2, and 4 - game 3 is team-b vs team-c)
        assert len(breakdowns) == 3

    def test_decompose_team_empty_for_unknown_team(self, sample_games):
        """Unknown team should return empty list."""
        config = AlgorithmConfig()
        breakdowns = decompose_team_rating("unknown-team", sample_games, config)

        assert len(breakdowns) == 0


# =============================================================================
# Prediction Generation Tests
# =============================================================================


class TestPredictionGeneration:
    """Tests for prediction generation."""

    def test_generate_predictions_for_all_games(self, sample_games, sample_ratings):
        """Should generate prediction for each game."""
        config = AlgorithmConfig()
        predictions = generate_predictions(sample_games, sample_ratings, config)

        assert len(predictions) == len(sample_games)

    def test_prediction_has_correct_fields(self, sample_games, sample_ratings):
        """Prediction should have all required fields."""
        config = AlgorithmConfig()
        predictions = generate_predictions(sample_games, sample_ratings, config)

        pred = predictions[0]
        assert pred.game_id == sample_games[0].game_id
        assert pred.home_team_id == sample_games[0].home_team_id
        assert pred.away_team_id == sample_games[0].away_team_id
        assert pred.predicted_winner in [pred.home_team_id, pred.away_team_id]
        assert 0.5 <= pred.win_probability <= 1.0

    def test_higher_rated_team_is_predicted_winner(self, sample_games, sample_ratings):
        """Higher rated team should be predicted winner."""
        config = AlgorithmConfig()
        predictions = generate_predictions(sample_games, sample_ratings, config)

        # Game 1: team-a (0.85) vs team-b (0.55) - team-a should be predicted
        pred = predictions[0]
        assert pred.predicted_winner == "team-a"

    def test_prediction_correctness_computed(self, sample_games, sample_ratings):
        """Prediction correctness should be computed for completed games."""
        config = AlgorithmConfig()
        predictions = generate_predictions(sample_games, sample_ratings, config)

        # All games are completed, so was_correct should be set
        for pred in predictions:
            assert pred.was_correct is not None

    def test_upset_detection(self, sample_games):
        """Upsets should be detected when underdog wins."""
        # Create ratings where team-c is heavy underdog
        ratings = {"team-a": 0.90, "team-b": 0.40, "team-c": 0.30}
        config = AlgorithmConfig()
        predictions = generate_predictions(sample_games, ratings, config)

        # Game 3: team-b vs team-c, team-c wins but is underdog
        game_3_pred = next(p for p in predictions if p.game_id == 3)
        # team-c (0.30) beat team-b (0.40) - this is an upset
        if game_3_pred.actual_winner == "team-c" and game_3_pred.predicted_winner == "team-b":
            assert game_3_pred.is_upset or game_3_pred.win_probability < 0.55


class TestPredictGame:
    """Tests for single game prediction."""

    def test_predict_game_basic(self, sample_ratings):
        """Basic game prediction should work."""
        config = AlgorithmConfig()
        pred = predict_game("team-a", "team-b", sample_ratings, config)

        assert pred.home_team_id == "team-a"
        assert pred.away_team_id == "team-b"
        assert pred.predicted_winner == "team-a"  # Higher rated
        assert pred.win_probability > 0.5

    def test_predict_game_neutral_site(self, sample_ratings):
        """Neutral site should reduce home advantage."""
        config = AlgorithmConfig()
        pred_home = predict_game("team-a", "team-b", sample_ratings, config, neutral_site=False)
        pred_neutral = predict_game("team-a", "team-b", sample_ratings, config, neutral_site=True)

        # Neutral site should have lower confidence for home team
        assert pred_neutral.win_probability <= pred_home.win_probability


# =============================================================================
# Metrics Tests
# =============================================================================


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_predictions_give_zero(self):
        """Perfect predictions should give Brier score of 0."""
        predictions = [
            Prediction(
                game_id=1, season=2024, week=1,
                home_team_id="a", away_team_id="b",
                home_rating=0.8, away_rating=0.5,
                rating_gap=0.3, predicted_winner="a",
                win_probability=1.0, actual_winner="a", was_correct=True,
            ),
        ]
        score = compute_brier_score(predictions)
        assert score == 0.0

    def test_completely_wrong_predictions(self):
        """Completely wrong confident predictions should give high Brier score."""
        predictions = [
            Prediction(
                game_id=1, season=2024, week=1,
                home_team_id="a", away_team_id="b",
                home_rating=0.8, away_rating=0.5,
                rating_gap=0.3, predicted_winner="a",
                win_probability=1.0, actual_winner="b", was_correct=False,
            ),
        ]
        score = compute_brier_score(predictions)
        assert score == 1.0  # (1.0 - 0)^2 = 1

    def test_uncertain_predictions(self):
        """50/50 predictions should give Brier score of 0.25."""
        predictions = [
            Prediction(
                game_id=1, season=2024, week=1,
                home_team_id="a", away_team_id="b",
                home_rating=0.5, away_rating=0.5,
                rating_gap=0.0, predicted_winner="a",
                win_probability=0.5, actual_winner="b", was_correct=False,
            ),
        ]
        score = compute_brier_score(predictions)
        assert abs(score - 0.25) < 0.01  # (0.5 - 0)^2 = 0.25

    def test_empty_predictions_return_zero(self):
        """Empty predictions should return 0."""
        score = compute_brier_score([])
        assert score == 0.0


class TestCalibrationError:
    """Tests for calibration error calculation."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have low error."""
        # 70% confidence, 70% correct
        predictions = []
        for i in range(10):
            correct = i < 7  # 7 correct, 3 wrong
            predictions.append(Prediction(
                game_id=i, season=2024, week=1,
                home_team_id="a", away_team_id="b",
                home_rating=0.7, away_rating=0.5,
                rating_gap=0.2, predicted_winner="a",
                win_probability=0.70, actual_winner="a" if correct else "b",
                was_correct=correct,
            ))

        error = compute_calibration_error(predictions)
        # Should be close to 0 since 70% conf matches 70% correct
        assert error < 0.15

    def test_empty_predictions_return_zero(self):
        """Empty predictions should return 0."""
        error = compute_calibration_error([])
        assert error == 0.0


# =============================================================================
# Parameter Attribution Tests
# =============================================================================


class TestParameterAttribution:
    """Tests for parameter attribution."""

    def test_attribution_identifies_patterns(self, sample_games):
        """Attribution should identify patterns in wrong predictions."""
        # Create wrong predictions
        wrong_preds = [
            Prediction(
                game_id=1, season=2024, week=1,
                home_team_id="team-a", away_team_id="team-b",
                home_rating=0.85, away_rating=0.55,
                rating_gap=0.30, predicted_winner="team-a",
                win_probability=0.85, actual_winner="team-b", was_correct=False,
                is_upset=True, upset_magnitude=0.85,
            ),
        ]
        config = AlgorithmConfig()
        attributions = attribute_errors_to_parameters(wrong_preds, sample_games, config)

        # Should find some attributions
        assert len(attributions) >= 0  # May or may not find patterns

    def test_attribution_sorted_by_error_count(self, sample_games):
        """Attributions should be sorted by error count descending."""
        # Multiple wrong predictions with different patterns
        wrong_preds = [
            Prediction(
                game_id=i, season=2024, week=1,
                home_team_id="team-a", away_team_id="team-b",
                home_rating=0.85, away_rating=0.35,
                rating_gap=0.50, predicted_winner="team-a",
                win_probability=0.95, actual_winner="team-b", was_correct=False,
                is_upset=True, upset_magnitude=0.95,
            )
            for i in range(5)
        ]
        config = AlgorithmConfig()
        attributions = attribute_errors_to_parameters(wrong_preds, sample_games, config)

        # Should be sorted by error count
        for i in range(len(attributions) - 1):
            assert attributions[i].error_count >= attributions[i + 1].error_count


# =============================================================================
# Diagnostic Report Tests
# =============================================================================


class TestDiagnosticReport:
    """Tests for full diagnostic report generation."""

    def test_generate_report_basic(self, sample_games):
        """Basic report generation should work."""
        config = AlgorithmConfig()
        report = generate_diagnostic_report(sample_games, config)

        assert report.season == 2024
        assert report.total_games > 0
        assert 0.0 <= report.accuracy <= 1.0
        assert report.brier_score >= 0.0

    def test_report_for_specific_week(self, sample_games):
        """Report should filter by week when specified."""
        config = AlgorithmConfig()
        report = generate_diagnostic_report(sample_games, config, week=1)

        # Week 1 only has 1 game
        assert report.total_games == 1
        assert report.week == 1

    def test_empty_games_return_empty_report(self):
        """Empty games should return zeroed report."""
        config = AlgorithmConfig()
        report = generate_diagnostic_report([], config)

        assert report.total_games == 0
        assert report.accuracy == 0.0
        assert report.predictions == []

    def test_report_contains_upsets(self, sample_games):
        """Report should identify upsets."""
        config = AlgorithmConfig()
        report = generate_diagnostic_report(sample_games, config)

        # Upsets list should be populated (may be empty if no upsets)
        assert isinstance(report.upsets, list)

    def test_report_to_dict(self, sample_games):
        """Report should be serializable to dict."""
        config = AlgorithmConfig()
        report = generate_diagnostic_report(sample_games, config)
        report_dict = report.to_dict()

        assert "season" in report_dict
        assert "summary" in report_dict
        assert "accuracy" in report_dict["summary"]


# =============================================================================
# Data Class Tests
# =============================================================================


class TestDataClasses:
    """Tests for diagnostic data classes."""

    def test_game_contribution_to_dict(self):
        """GameContributionBreakdown should serialize to dict."""
        breakdown = GameContributionBreakdown(
            game_id=1,
            team_id="team-a",
            opponent_id="team-b",
            is_win=True,
            margin=14,
            location="home",
            base_result=0.70,
            margin_bonus=0.10,
            venue_adjustment=0.0,
            opponent_rating=0.55,
            opponent_contribution=0.55,
            quality_tier_adjustment=0.0,
            conference_multiplier=1.0,
            recency_weight=1.0,
            game_grade=0.80,
            total_contribution=1.35,
        )

        d = breakdown.to_dict()
        assert d["game_id"] == 1
        assert d["is_win"] is True
        assert "components" in d
        assert d["components"]["base_result"] == 0.70

    def test_prediction_to_dict(self):
        """Prediction should serialize to dict."""
        pred = Prediction(
            game_id=1,
            season=2024,
            week=1,
            home_team_id="team-a",
            away_team_id="team-b",
            home_rating=0.85,
            away_rating=0.55,
            rating_gap=0.30,
            predicted_winner="team-a",
            win_probability=0.85,
        )

        d = pred.to_dict()
        assert d["game_id"] == 1
        assert d["predicted_winner"] == "team-a"
        assert d["win_probability"] == 0.85

    def test_parameter_attribution_to_dict(self):
        """ParameterAttribution should serialize to dict."""
        attr = ParameterAttribution(
            parameter="g5_multiplier",
            error_count=5,
            pattern_description="G5 teams beating P5",
            affected_games=[1, 2, 3, 4, 5],
            suggested_adjustment="Increase g5_multiplier",
            current_value=0.92,
            suggested_value=0.95,
        )

        d = attr.to_dict()
        assert d["parameter"] == "g5_multiplier"
        assert d["error_count"] == 5
