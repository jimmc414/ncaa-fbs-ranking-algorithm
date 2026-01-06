"""Tests for configuration profiles."""

import pytest


class TestProfiles:
    """Tests for predefined configuration profiles."""

    def test_pure_results_profile(self):
        """Pure results profile uses all defaults."""
        from src.data.profiles import PURE_RESULTS
        from src.data.models import AlgorithmConfig

        default = AlgorithmConfig()
        assert PURE_RESULTS.enable_conference_adj == default.enable_conference_adj
        assert PURE_RESULTS.enable_quality_tiers == default.enable_quality_tiers
        assert PURE_RESULTS.enable_recency == default.enable_recency

    def test_balanced_profile(self):
        """Balanced profile has moderate adjustments."""
        from src.data.profiles import BALANCED

        # Conference adjustments enabled
        assert BALANCED.enable_conference_adj is True
        assert BALANCED.p5_multiplier > 1.0
        assert BALANCED.g5_multiplier < 1.0

        # Quality tiers enabled
        assert BALANCED.enable_quality_tiers is True
        assert BALANCED.elite_win_bonus > 0
        assert BALANCED.bad_loss_penalty > 0

        # Moderate SOS adjustment
        assert BALANCED.sos_adjustment_weight > 0

    def test_predictive_profile(self):
        """Predictive profile has strong adjustments for prediction."""
        from src.data.profiles import PREDICTIVE

        # Strong conference adjustments
        assert PREDICTIVE.enable_conference_adj is True
        assert PREDICTIVE.g5_multiplier < 0.90  # Stronger than balanced

        # SOS requirements for top rankings
        assert PREDICTIVE.min_sos_top_10 > 0
        assert PREDICTIVE.min_p5_games_top_10 >= 3

        # Recency enabled
        assert PREDICTIVE.enable_recency is True
        assert PREDICTIVE.recency_half_life <= 8

    def test_conservative_profile(self):
        """Conservative profile uses priors and long memory."""
        from src.data.profiles import CONSERVATIVE

        # Prior enabled
        assert CONSERVATIVE.enable_prior is True
        assert CONSERVATIVE.prior_weight > 0

        # Recency with longer half-life
        assert CONSERVATIVE.enable_recency is True
        assert CONSERVATIVE.recency_half_life >= 8


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_valid_profile(self):
        """Get profile by valid name."""
        from src.data.profiles import get_profile, BALANCED

        config = get_profile("balanced")
        assert config == BALANCED

    def test_get_all_profiles(self):
        """All profiles can be retrieved."""
        from src.data.profiles import get_profile, list_profiles

        for name in list_profiles():
            config = get_profile(name)
            assert config is not None

    def test_get_invalid_profile(self):
        """Invalid profile name raises ValueError."""
        from src.data.profiles import get_profile

        with pytest.raises(ValueError) as exc_info:
            get_profile("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert "Available" in str(exc_info.value)


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_list_includes_all_profiles(self):
        """List includes all expected profiles."""
        from src.data.profiles import list_profiles

        profiles = list_profiles()
        assert "pure_results" in profiles
        assert "balanced" in profiles
        assert "predictive" in profiles
        assert "conservative" in profiles

    def test_list_returns_list_of_strings(self):
        """List returns list of strings."""
        from src.data.profiles import list_profiles

        profiles = list_profiles()
        assert isinstance(profiles, list)
        assert all(isinstance(p, str) for p in profiles)


class TestProfileDescriptions:
    """Tests for profile descriptions."""

    def test_all_profiles_have_descriptions(self):
        """All profiles have descriptions."""
        from src.data.profiles import list_profiles, get_profile_description

        for name in list_profiles():
            desc = get_profile_description(name)
            assert len(desc) > 10  # Meaningful description

    def test_unknown_profile_description(self):
        """Unknown profile returns default description."""
        from src.data.profiles import get_profile_description

        desc = get_profile_description("nonexistent")
        assert "No description" in desc


class TestProfilesAreValid:
    """Tests that profiles can be used with the algorithm."""

    def test_profiles_work_with_convergence(self, four_team_round_robin):
        """All profiles work with the convergence algorithm."""
        from src.data.profiles import list_profiles, get_profile
        from src.algorithm.convergence import converge

        for name in list_profiles():
            config = get_profile(name)
            result = converge(four_team_round_robin, config)
            # Should produce valid rankings
            assert len(result.ratings) == 4
            assert result.iterations > 0
