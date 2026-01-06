"""Tests for expanded AlgorithmConfig with 45 configurable levers."""

import pytest
from pydantic import ValidationError


class TestAlgorithmConfigOpponentInfluence:
    """Tests for opponent influence configuration fields."""

    def test_opponent_weight_default(self):
        """Default opponent weight is 1.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.opponent_weight == 1.0

    def test_opponent_weight_custom(self):
        """Custom opponent weight accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(opponent_weight=1.2)
        assert config.opponent_weight == 1.2

    def test_loss_opponent_factor_default(self):
        """Default loss opponent factor is -1.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.loss_opponent_factor == -1.0

    def test_loss_opponent_factor_custom(self):
        """Custom loss opponent factor accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(loss_opponent_factor=-0.8)
        assert config.loss_opponent_factor == -0.8

    def test_second_order_weight_default(self):
        """Default second order weight is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.second_order_weight == 0.0

    def test_second_order_weight_custom(self):
        """Custom second order weight accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(second_order_weight=0.15)
        assert config.second_order_weight == 0.15


class TestAlgorithmConfigConference:
    """Tests for conference adjustment configuration fields."""

    def test_enable_conference_adj_default(self):
        """Conference adjustment disabled by default."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.enable_conference_adj is False

    def test_enable_conference_adj_enabled(self):
        """Conference adjustment can be enabled."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(enable_conference_adj=True)
        assert config.enable_conference_adj is True

    def test_conference_method_default(self):
        """Default conference method is empirical."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.conference_method == "empirical"

    def test_conference_method_manual(self):
        """Manual conference method accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(conference_method="manual")
        assert config.conference_method == "manual"

    def test_conference_method_invalid(self):
        """Invalid conference method rejected."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(conference_method="invalid")

    def test_p5_multiplier_default(self):
        """Default P5 multiplier is 1.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.p5_multiplier == 1.0

    def test_g5_multiplier_default(self):
        """Default G5 multiplier is 1.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.g5_multiplier == 1.0

    def test_fcs_multiplier_default(self):
        """Default FCS multiplier is 0.5."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.fcs_multiplier == 0.5

    def test_conference_multipliers_custom(self):
        """Custom conference multipliers accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(
            p5_multiplier=1.05,
            g5_multiplier=0.85,
            fcs_multiplier=0.40,
        )
        assert config.p5_multiplier == 1.05
        assert config.g5_multiplier == 0.85
        assert config.fcs_multiplier == 0.40


class TestAlgorithmConfigScheduleStrength:
    """Tests for schedule strength configuration fields."""

    def test_sos_adjustment_weight_default(self):
        """Default SOS adjustment weight is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.sos_adjustment_weight == 0.0

    def test_sos_adjustment_weight_custom(self):
        """Custom SOS adjustment weight accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(sos_adjustment_weight=0.15)
        assert config.sos_adjustment_weight == 0.15

    def test_sos_method_default(self):
        """Default SOS method is mean."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.sos_method == "mean"

    def test_sos_method_median(self):
        """Median SOS method accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(sos_method="median")
        assert config.sos_method == "median"

    def test_sos_method_invalid(self):
        """Invalid SOS method rejected."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(sos_method="invalid")

    def test_min_sos_top_10_default(self):
        """Default min SOS for top 10 is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.min_sos_top_10 == 0.0

    def test_min_sos_top_25_default(self):
        """Default min SOS for top 25 is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.min_sos_top_25 == 0.0

    def test_min_p5_games_top_10_default(self):
        """Default min P5 games for top 10 is 0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.min_p5_games_top_10 == 0

    def test_schedule_strength_thresholds_custom(self):
        """Custom schedule strength thresholds accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(
            min_sos_top_10=0.50,
            min_sos_top_25=0.40,
            min_p5_games_top_10=4,
        )
        assert config.min_sos_top_10 == 0.50
        assert config.min_sos_top_25 == 0.40
        assert config.min_p5_games_top_10 == 4


class TestAlgorithmConfigQualityTiers:
    """Tests for quality tier configuration fields."""

    def test_enable_quality_tiers_default(self):
        """Quality tiers disabled by default."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.enable_quality_tiers is False

    def test_enable_quality_tiers_enabled(self):
        """Quality tiers can be enabled."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(enable_quality_tiers=True)
        assert config.enable_quality_tiers is True

    def test_elite_threshold_default(self):
        """Default elite threshold is 0.80."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.elite_threshold == 0.80

    def test_good_threshold_default(self):
        """Default good threshold is 0.55."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.good_threshold == 0.55

    def test_bad_threshold_default(self):
        """Default bad threshold is 0.35."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.bad_threshold == 0.35

    def test_elite_win_bonus_default(self):
        """Default elite win bonus is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.elite_win_bonus == 0.0

    def test_bad_loss_penalty_default(self):
        """Default bad loss penalty is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.bad_loss_penalty == 0.0

    def test_quality_tier_thresholds_custom(self):
        """Custom quality tier thresholds accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(
            elite_threshold=0.85,
            good_threshold=0.60,
            bad_threshold=0.30,
            elite_win_bonus=0.10,
            bad_loss_penalty=0.15,
        )
        assert config.elite_threshold == 0.85
        assert config.good_threshold == 0.60
        assert config.bad_threshold == 0.30
        assert config.elite_win_bonus == 0.10
        assert config.bad_loss_penalty == 0.15


class TestAlgorithmConfigRecency:
    """Tests for recency weighting configuration fields."""

    def test_enable_recency_default(self):
        """Recency weighting disabled by default."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.enable_recency is False

    def test_enable_recency_enabled(self):
        """Recency weighting can be enabled."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(enable_recency=True)
        assert config.enable_recency is True

    def test_recency_half_life_default(self):
        """Default recency half-life is 8 weeks."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.recency_half_life == 8

    def test_recency_half_life_custom(self):
        """Custom recency half-life accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(recency_half_life=6)
        assert config.recency_half_life == 6

    def test_recency_half_life_positive(self):
        """Recency half-life must be positive."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(recency_half_life=0)

    def test_recency_min_weight_default(self):
        """Default recency min weight is 0.5."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.recency_min_weight == 0.5

    def test_recency_min_weight_custom(self):
        """Custom recency min weight accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(recency_min_weight=0.3)
        assert config.recency_min_weight == 0.3


class TestAlgorithmConfigPrior:
    """Tests for historical prior configuration fields."""

    def test_enable_prior_default(self):
        """Prior disabled by default."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.enable_prior is False

    def test_enable_prior_enabled(self):
        """Prior can be enabled."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(enable_prior=True)
        assert config.enable_prior is True

    def test_prior_weight_default(self):
        """Default prior weight is 0.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.prior_weight == 0.0

    def test_prior_weight_custom(self):
        """Custom prior weight accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(prior_weight=0.20)
        assert config.prior_weight == 0.20

    def test_prior_decay_weeks_default(self):
        """Default prior decay weeks is 8."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.prior_decay_weeks == 8

    def test_prior_decay_weeks_custom(self):
        """Custom prior decay weeks accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(prior_decay_weeks=6)
        assert config.prior_decay_weeks == 6

    def test_prior_decay_weeks_positive(self):
        """Prior decay weeks must be positive."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(prior_decay_weeks=0)


class TestAlgorithmConfigMarginCurve:
    """Tests for margin curve configuration fields."""

    def test_margin_curve_default(self):
        """Default margin curve is log."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.margin_curve == "log"

    def test_margin_curve_linear(self):
        """Linear margin curve accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(margin_curve="linear")
        assert config.margin_curve == "linear"

    def test_margin_curve_sqrt(self):
        """Sqrt margin curve accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(margin_curve="sqrt")
        assert config.margin_curve == "sqrt"

    def test_margin_curve_invalid(self):
        """Invalid margin curve rejected."""
        from src.data.models import AlgorithmConfig

        with pytest.raises(ValidationError):
            AlgorithmConfig(margin_curve="invalid")


class TestAlgorithmConfigTiebreakers:
    """Tests for tiebreaker configuration fields."""

    def test_tie_threshold_default(self):
        """Default tie threshold is 0.001."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.tie_threshold == 0.001

    def test_tie_threshold_custom(self):
        """Custom tie threshold accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(tie_threshold=0.005)
        assert config.tie_threshold == 0.005

    def test_tiebreaker_order_default(self):
        """Default tiebreaker order is h2h, sov, common, away."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.tiebreaker_order == ["h2h", "sov", "common", "away"]

    def test_tiebreaker_order_custom(self):
        """Custom tiebreaker order accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(tiebreaker_order=["sov", "h2h", "away"])
        assert config.tiebreaker_order == ["sov", "h2h", "away"]


class TestAlgorithmConfigFilters:
    """Tests for game filter configuration fields."""

    def test_include_postseason_default(self):
        """Postseason included by default."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.include_postseason is True

    def test_include_postseason_disabled(self):
        """Postseason can be excluded."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(include_postseason=False)
        assert config.include_postseason is False

    def test_postseason_weight_default(self):
        """Default postseason weight is 1.0."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.postseason_weight == 1.0

    def test_postseason_weight_custom(self):
        """Custom postseason weight accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(postseason_weight=1.5)
        assert config.postseason_weight == 1.5

    def test_include_fcs_games_default(self):
        """FCS games included by default."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.include_fcs_games is True

    def test_include_fcs_games_disabled(self):
        """FCS games can be excluded."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(include_fcs_games=False)
        assert config.include_fcs_games is False

    def test_min_games_to_rank_default(self):
        """Default min games to rank is 1."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.min_games_to_rank == 1

    def test_min_games_to_rank_custom(self):
        """Custom min games to rank accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(min_games_to_rank=6)
        assert config.min_games_to_rank == 6

    def test_min_games_to_rank_zero(self):
        """Min games to rank of 0 accepted."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(min_games_to_rank=0)
        assert config.min_games_to_rank == 0


class TestAlgorithmConfigSerialization:
    """Tests for config serialization and JSON export."""

    def test_config_to_dict(self):
        """Config can be serialized to dict."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(
            enable_quality_tiers=True,
            elite_win_bonus=0.10,
        )
        data = config.model_dump()
        assert data["enable_quality_tiers"] is True
        assert data["elite_win_bonus"] == 0.10

    def test_config_from_dict(self):
        """Config can be created from dict."""
        from src.data.models import AlgorithmConfig

        data = {
            "enable_conference_adj": True,
            "g5_multiplier": 0.85,
            "min_games_to_rank": 6,
        }
        config = AlgorithmConfig(**data)
        assert config.enable_conference_adj is True
        assert config.g5_multiplier == 0.85
        assert config.min_games_to_rank == 6

    def test_config_json_roundtrip(self):
        """Config survives JSON serialization roundtrip."""
        from src.data.models import AlgorithmConfig
        import json

        config = AlgorithmConfig(
            enable_recency=True,
            recency_half_life=6,
            tiebreaker_order=["sov", "h2h"],
        )
        json_str = config.model_dump_json()
        data = json.loads(json_str)
        config2 = AlgorithmConfig(**data)
        assert config2.enable_recency is True
        assert config2.recency_half_life == 6
        assert config2.tiebreaker_order == ["sov", "h2h"]

    def test_config_all_43_fields(self):
        """Config has all 43 expected fields (11 original + 32 new)."""
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig()
        data = config.model_dump()

        expected_fields = [
            # Core (5)
            "max_iterations",
            "convergence_threshold",
            "win_base",
            "margin_weight",
            "margin_cap",
            # Venue (4)
            "venue_road_win",
            "venue_neutral_win",
            "venue_home_loss",
            "venue_neutral_loss",
            # Initialization (2)
            "initial_rating",
            "fcs_fixed_rating",
            # Opponent Influence (3)
            "opponent_weight",
            "loss_opponent_factor",
            "second_order_weight",
            # Conference (5)
            "enable_conference_adj",
            "conference_method",
            "p5_multiplier",
            "g5_multiplier",
            "fcs_multiplier",
            # Schedule Strength (5)
            "sos_adjustment_weight",
            "sos_method",
            "min_sos_top_10",
            "min_sos_top_25",
            "min_p5_games_top_10",
            # Quality Tiers (6)
            "enable_quality_tiers",
            "elite_threshold",
            "good_threshold",
            "bad_threshold",
            "elite_win_bonus",
            "bad_loss_penalty",
            # Recency (3)
            "enable_recency",
            "recency_half_life",
            "recency_min_weight",
            # Prior (3)
            "enable_prior",
            "prior_weight",
            "prior_decay_weeks",
            # Margin Curve (1)
            "margin_curve",
            # Tiebreakers (2)
            "tie_threshold",
            "tiebreaker_order",
            # Filters (4)
            "include_postseason",
            "postseason_weight",
            "include_fcs_games",
            "min_games_to_rank",
        ]

        assert len(expected_fields) == 43
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
