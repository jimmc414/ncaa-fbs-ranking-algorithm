"""Unit tests for consensus prediction logic."""

import pytest

from src.validation.consensus import (
    assess_confidence,
    build_game_prediction,
    calculate_consensus,
    elo_to_win_probability,
    organize_by_confidence,
    rating_gap_to_probability,
    spread_to_probability,
    CONSENSUS_WEIGHTS,
)
from src.validation.models import GamePrediction, SourcePrediction


class TestSpreadToProbability:
    """Tests for spread_to_probability function."""

    def test_home_favorite_negative_spread(self):
        """Negative spread means home team is favored."""
        prob = spread_to_probability(-7.0)
        assert 0.65 < prob < 0.75  # -7 should be around 70%

    def test_away_favorite_positive_spread(self):
        """Positive spread means away team is favored."""
        prob = spread_to_probability(7.0)
        assert 0.25 < prob < 0.35  # +7 should be around 30% for home

    def test_even_spread(self):
        """Zero spread should be close to 50%."""
        prob = spread_to_probability(0.0)
        assert 0.48 < prob < 0.52

    def test_large_favorite(self):
        """Large favorites should have high probability."""
        prob = spread_to_probability(-14.0)
        assert prob > 0.80

    def test_large_underdog(self):
        """Large underdogs should have low probability."""
        prob = spread_to_probability(14.0)
        assert prob < 0.20

    def test_probability_bounds(self):
        """Probability should always be between 0 and 1."""
        for spread in [-50, -21, -7, 0, 7, 21, 50]:
            prob = spread_to_probability(spread)
            assert 0.0 <= prob <= 1.0


class TestEloToWinProbability:
    """Tests for elo_to_win_probability function."""

    def test_equal_elo(self):
        """Equal Elo with home advantage should favor home."""
        prob = elo_to_win_probability(1500, 1500)
        assert 0.55 < prob < 0.60  # Home advantage ~55 points

    def test_higher_home_elo(self):
        """Higher home Elo should increase win probability."""
        prob = elo_to_win_probability(1700, 1500)
        assert prob > 0.70

    def test_higher_away_elo(self):
        """Higher away Elo should decrease home win probability."""
        prob = elo_to_win_probability(1500, 1700)
        assert prob < 0.40

    def test_neutral_site(self):
        """Zero home advantage at neutral site."""
        prob = elo_to_win_probability(1500, 1500, home_advantage=0.0)
        assert 0.48 < prob < 0.52  # Should be close to 50%

    def test_probability_bounds(self):
        """Probability should always be between 0 and 1."""
        test_cases = [
            (2000, 1200),  # Strong home team
            (1200, 2000),  # Strong away team
            (1500, 1500),  # Equal
        ]
        for home_elo, away_elo in test_cases:
            prob = elo_to_win_probability(home_elo, away_elo)
            assert 0.0 <= prob <= 1.0


class TestRatingGapToProbability:
    """Tests for rating_gap_to_probability function."""

    def test_positive_gap_favors_home(self):
        """Positive rating gap should favor home team."""
        prob = rating_gap_to_probability(0.10)
        assert prob > 0.60

    def test_negative_gap_favors_away(self):
        """Negative rating gap should favor away team."""
        prob = rating_gap_to_probability(-0.10)
        assert prob < 0.40

    def test_zero_gap_with_home_advantage(self):
        """Zero gap with home advantage should slightly favor home."""
        prob = rating_gap_to_probability(0.0, home_advantage=0.03)
        assert 0.55 < prob < 0.65

    def test_neutral_site_zero_gap(self):
        """Zero gap at neutral site should be close to 50%."""
        prob = rating_gap_to_probability(0.0, home_advantage=0.0)
        assert 0.48 < prob < 0.52


class TestCalculateConsensus:
    """Tests for calculate_consensus function."""

    def test_single_source(self):
        """Single source should return that probability."""
        predictions = {"our_algorithm": 0.70}
        consensus, _ = calculate_consensus(predictions)
        assert 0.69 < consensus < 0.71

    def test_weighted_average(self):
        """Multiple sources should return weighted average."""
        predictions = {
            "vegas": 0.80,
            "our_algorithm": 0.60,
        }
        consensus, contributions = calculate_consensus(predictions)
        # Vegas has 0.35 weight, our_algo has 0.25 weight
        # Normalized: vegas 0.35/(0.35+0.25)=0.583, our_algo 0.25/0.6=0.417
        # Expected: 0.80*0.583 + 0.60*0.417 = 0.716
        assert 0.70 < consensus < 0.75

    def test_empty_predictions(self):
        """Empty predictions should return 0.5."""
        predictions = {}
        consensus, _ = calculate_consensus(predictions)
        assert consensus == 0.5

    def test_none_values_filtered(self):
        """None values should be filtered out."""
        predictions = {
            "vegas": 0.70,
            "our_algorithm": None,
            "pregame_wp": 0.60,
        }
        consensus, contributions = calculate_consensus(predictions)
        assert "our_algorithm" not in contributions
        assert "vegas" in contributions
        assert "pregame_wp" in contributions

    def test_custom_weights(self):
        """Custom weights should be used when provided."""
        predictions = {"vegas": 0.80, "our_algorithm": 0.60}
        custom_weights = {"vegas": 0.5, "our_algorithm": 0.5}
        consensus, _ = calculate_consensus(predictions, weights=custom_weights)
        # Equal weights: (0.80 + 0.60) / 2 = 0.70
        assert 0.69 < consensus < 0.71


class TestAssessConfidence:
    """Tests for assess_confidence function."""

    def test_high_confidence_all_agree_tight(self):
        """All sources agree with tight spread should be HIGH."""
        predictions = {
            "vegas": 0.72,
            "our_algorithm": 0.70,
            "pregame_wp": 0.68,
        }
        breakdown = assess_confidence(predictions)
        assert breakdown.level == "HIGH"
        assert breakdown.sources_agreeing == 3

    def test_moderate_confidence_all_agree_moderate_spread(self):
        """All sources agree with moderate spread should be MODERATE."""
        predictions = {
            "vegas": 0.75,
            "our_algorithm": 0.60,
            "pregame_wp": 0.65,
        }
        breakdown = assess_confidence(predictions)
        assert breakdown.level == "MODERATE"

    def test_low_confidence_all_agree_wide_spread(self):
        """All sources agree with wide spread should be LOW."""
        predictions = {
            "vegas": 0.85,
            "our_algorithm": 0.55,
        }
        breakdown = assess_confidence(predictions)
        assert breakdown.level == "LOW"

    def test_split_sources_disagree(self):
        """Sources disagreeing on winner should be SPLIT."""
        predictions = {
            "vegas": 0.55,  # Favors home
            "our_algorithm": 0.45,  # Favors away
        }
        breakdown = assess_confidence(predictions)
        assert breakdown.level == "SPLIT"

    def test_unknown_no_sources(self):
        """No sources should return UNKNOWN."""
        predictions = {}
        breakdown = assess_confidence(predictions)
        assert breakdown.level == "UNKNOWN"

    def test_probability_spread_calculated(self):
        """Probability spread should be calculated correctly."""
        predictions = {
            "vegas": 0.80,
            "our_algorithm": 0.60,
        }
        breakdown = assess_confidence(predictions)
        assert abs(breakdown.probability_spread - 0.20) < 0.01


class TestBuildGamePrediction:
    """Tests for build_game_prediction function."""

    def test_basic_prediction(self):
        """Build basic prediction with our algorithm only."""
        prediction = build_game_prediction(
            home_team_id="ohio-state",
            away_team_id="michigan",
            home_rating=0.80,
            away_rating=0.70,
            season=2024,
            week=12,
        )
        assert prediction.home_team_id == "ohio-state"
        assert prediction.away_team_id == "michigan"
        assert prediction.consensus_winner is not None
        assert prediction.consensus_prob is not None

    def test_with_vegas_spread(self):
        """Build prediction with Vegas spread."""
        prediction = build_game_prediction(
            home_team_id="ohio-state",
            away_team_id="michigan",
            home_rating=0.80,
            away_rating=0.70,
            vegas_spread=-7.5,
            season=2024,
        )
        vegas_source = prediction.get_source("vegas")
        assert vegas_source is not None
        assert vegas_source.available is True
        assert vegas_source.spread == -7.5

    def test_with_all_sources(self):
        """Build prediction with all sources available."""
        prediction = build_game_prediction(
            home_team_id="ohio-state",
            away_team_id="michigan",
            home_rating=0.80,
            away_rating=0.70,
            vegas_spread=-7.5,
            pregame_wp=0.72,
            home_sp_rating=25.0,
            away_sp_rating=18.0,
            home_elo=1800.0,
            away_elo=1700.0,
            season=2024,
        )
        assert prediction.sources_available == 5
        for source in ["our_algorithm", "vegas", "pregame_wp", "sp_implied", "elo_implied"]:
            src = prediction.get_source(source)
            assert src is not None
            assert src.available is True

    def test_missing_sources_marked_unavailable(self):
        """Missing sources should be marked as unavailable."""
        prediction = build_game_prediction(
            home_team_id="ohio-state",
            away_team_id="michigan",
            home_rating=0.80,
            away_rating=0.70,
            season=2024,
        )
        vegas_source = prediction.get_source("vegas")
        assert vegas_source is not None
        assert vegas_source.available is False

    def test_neutral_site(self):
        """Neutral site should adjust probabilities."""
        home_pred = build_game_prediction(
            home_team_id="ohio-state",
            away_team_id="michigan",
            home_rating=0.75,
            away_rating=0.75,
            season=2024,
            neutral_site=False,
        )
        neutral_pred = build_game_prediction(
            home_team_id="ohio-state",
            away_team_id="michigan",
            home_rating=0.75,
            away_rating=0.75,
            season=2024,
            neutral_site=True,
        )
        # Home advantage should make home team slightly favored when not neutral
        home_source = home_pred.get_source("our_algorithm")
        neutral_source = neutral_pred.get_source("our_algorithm")
        assert home_source.win_probability > neutral_source.win_probability


class TestOrganizeByConfidence:
    """Tests for organize_by_confidence function."""

    def test_empty_list(self):
        """Empty list should return empty week predictions."""
        result = organize_by_confidence([])
        assert len(result.high_confidence) == 0
        assert len(result.moderate_confidence) == 0

    def test_sorts_into_buckets(self):
        """Predictions should be sorted into confidence buckets."""
        predictions = [
            GamePrediction(
                game_id=1, season=2024, week=12, home_team_id="a", away_team_id="b",
                confidence="HIGH", consensus_prob=0.85, sources_agreeing=5, sources_available=5,
            ),
            GamePrediction(
                game_id=2, season=2024, week=12, home_team_id="c", away_team_id="d",
                confidence="MODERATE", consensus_prob=0.70, sources_agreeing=4, sources_available=4,
            ),
            GamePrediction(
                game_id=3, season=2024, week=12, home_team_id="e", away_team_id="f",
                confidence="SPLIT", consensus_prob=0.52, sources_agreeing=2, sources_available=4,
            ),
        ]
        result = organize_by_confidence(predictions)
        assert len(result.high_confidence) == 1
        assert len(result.moderate_confidence) == 1
        assert len(result.split) == 1

    def test_sorted_by_probability(self):
        """Each bucket should be sorted by consensus probability."""
        predictions = [
            GamePrediction(
                game_id=1, season=2024, week=12, home_team_id="a", away_team_id="b",
                confidence="HIGH", consensus_prob=0.70, sources_agreeing=5, sources_available=5,
            ),
            GamePrediction(
                game_id=2, season=2024, week=12, home_team_id="c", away_team_id="d",
                confidence="HIGH", consensus_prob=0.90, sources_agreeing=5, sources_available=5,
            ),
        ]
        result = organize_by_confidence(predictions)
        assert result.high_confidence[0].consensus_prob > result.high_confidence[1].consensus_prob


class TestConsensusWeights:
    """Tests for CONSENSUS_WEIGHTS configuration."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        total = sum(CONSENSUS_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_vegas_highest_weight(self):
        """Vegas should have the highest weight."""
        assert CONSENSUS_WEIGHTS["vegas"] == max(CONSENSUS_WEIGHTS.values())

    def test_all_weights_positive(self):
        """All weights should be positive."""
        for weight in CONSENSUS_WEIGHTS.values():
            assert weight > 0
