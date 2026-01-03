"""Tests for game grade calculation."""

import math

import pytest


class TestMarginBonus:
    """Tests for margin bonus calculation."""

    def test_margin_zero(self):
        """Zero margin gives zero bonus."""
        from src.algorithm.game_grade import compute_margin_bonus

        assert compute_margin_bonus(0, is_win=True) == 0.0

    def test_margin_one_point(self):
        """1-point win gives small bonus."""
        from src.algorithm.game_grade import compute_margin_bonus

        expected = 0.20 * math.log(2) / math.log(29)
        result = compute_margin_bonus(1, is_win=True)
        assert abs(result - expected) < 0.001

    def test_margin_seven_points(self):
        """7-point win (one TD) gives moderate bonus."""
        from src.algorithm.game_grade import compute_margin_bonus

        expected = 0.20 * math.log(8) / math.log(29)
        result = compute_margin_bonus(7, is_win=True)
        assert abs(result - expected) < 0.001

    def test_margin_14_points(self):
        """14-point win gives more bonus but logarithmic."""
        from src.algorithm.game_grade import compute_margin_bonus

        bonus_7 = compute_margin_bonus(7, is_win=True)
        bonus_14 = compute_margin_bonus(14, is_win=True)

        # 14 points should give more than 7, but not double
        assert bonus_14 > bonus_7
        assert bonus_14 < 2 * bonus_7

    def test_margin_28_points(self):
        """28-point win gives maximum bonus."""
        from src.algorithm.game_grade import compute_margin_bonus

        expected = 0.20  # Full margin weight
        result = compute_margin_bonus(28, is_win=True)
        assert abs(result - expected) < 0.001

    def test_margin_capped_at_28(self):
        """Margin beyond 28 gives same as 28."""
        from src.algorithm.game_grade import compute_margin_bonus

        bonus_28 = compute_margin_bonus(28, is_win=True)
        bonus_35 = compute_margin_bonus(35, is_win=True)
        bonus_56 = compute_margin_bonus(56, is_win=True)
        bonus_100 = compute_margin_bonus(100, is_win=True)

        assert bonus_28 == bonus_35
        assert bonus_28 == bonus_56
        assert bonus_28 == bonus_100

    def test_margin_loss_is_zero(self):
        """Losses get no margin bonus regardless of margin."""
        from src.algorithm.game_grade import compute_margin_bonus

        assert compute_margin_bonus(1, is_win=False) == 0.0
        assert compute_margin_bonus(14, is_win=False) == 0.0
        assert compute_margin_bonus(28, is_win=False) == 0.0
        assert compute_margin_bonus(50, is_win=False) == 0.0

    def test_margin_logarithmic_curve(self):
        """First TD of margin worth more than third."""
        from src.algorithm.game_grade import compute_margin_bonus

        # First 7 points of margin
        first_td = compute_margin_bonus(7, is_win=True) - compute_margin_bonus(0, is_win=True)

        # Going from 14 to 21 points (third TD)
        third_td = compute_margin_bonus(21, is_win=True) - compute_margin_bonus(14, is_win=True)

        # First TD should be worth more than third TD
        assert first_td > third_td


class TestVenueAdjustment:
    """Tests for venue adjustment calculation."""

    def test_road_win(self):
        """Road win gets +0.10."""
        from src.algorithm.game_grade import compute_venue_adjustment

        assert compute_venue_adjustment("away", is_win=True) == 0.10

    def test_neutral_win(self):
        """Neutral win gets +0.05."""
        from src.algorithm.game_grade import compute_venue_adjustment

        assert compute_venue_adjustment("neutral", is_win=True) == 0.05

    def test_home_win(self):
        """Home win gets no adjustment."""
        from src.algorithm.game_grade import compute_venue_adjustment

        assert compute_venue_adjustment("home", is_win=True) == 0.0

    def test_road_loss(self):
        """Road loss gets no penalty."""
        from src.algorithm.game_grade import compute_venue_adjustment

        assert compute_venue_adjustment("away", is_win=False) == 0.0

    def test_home_loss(self):
        """Home loss gets -0.03 penalty."""
        from src.algorithm.game_grade import compute_venue_adjustment

        assert compute_venue_adjustment("home", is_win=False) == -0.03

    def test_neutral_loss(self):
        """Neutral loss gets -0.01 penalty."""
        from src.algorithm.game_grade import compute_venue_adjustment

        assert compute_venue_adjustment("neutral", is_win=False) == -0.01


class TestGameGrade:
    """Tests for complete game grade calculation."""

    def test_road_blowout_win(self):
        """Road win by 28: 0.70 + 0.20 + 0.10 = 1.00."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=True, margin=28, location="away")
        assert abs(grade - 1.00) < 0.01

    def test_home_squeaker_win(self):
        """Home win by 1: 0.70 + ~0.041 + 0.00 = ~0.741."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=True, margin=1, location="home")
        # 0.70 + 0.20 * ln(2)/ln(29) + 0.00 = ~0.741
        assert 0.74 < grade < 0.75

    def test_neutral_close_win(self):
        """Neutral win by 3: 0.70 + ~0.082 + 0.05 = ~0.832."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=True, margin=3, location="neutral")
        # 0.70 + 0.20 * ln(4)/ln(29) + 0.05 = ~0.832
        assert 0.83 < grade < 0.84

    def test_road_loss(self):
        """Road loss: 0.00 + 0.00 + 0.00 = 0.00."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=False, margin=14, location="away")
        assert grade == 0.00

    def test_home_loss(self):
        """Home loss: 0.00 + 0.00 - 0.03 = -0.03."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=False, margin=3, location="home")
        assert grade == -0.03

    def test_neutral_loss(self):
        """Neutral loss: 0.00 + 0.00 - 0.01 = -0.01."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=False, margin=7, location="neutral")
        assert grade == -0.01

    def test_home_blowout_win(self):
        """Home win by 28: 0.70 + 0.20 + 0.00 = 0.90."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=True, margin=28, location="home")
        assert abs(grade - 0.90) < 0.01

    def test_close_road_loss(self):
        """Close road loss: 0.00 (no penalty for road losses)."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=False, margin=1, location="away")
        assert grade == 0.00

    def test_blowout_home_loss(self):
        """Blowout home loss: still just -0.03."""
        from src.algorithm.game_grade import compute_game_grade

        grade = compute_game_grade(is_win=False, margin=35, location="home")
        assert grade == -0.03


class TestGameGradeWithConfig:
    """Tests for game grade with custom config."""

    def test_custom_win_base(self):
        """Custom win base value is used."""
        from src.algorithm.game_grade import compute_game_grade
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(win_base=0.80)
        grade = compute_game_grade(
            is_win=True, margin=0, location="home", config=config
        )
        # 0.80 + 0 + 0 = 0.80
        assert abs(grade - 0.80) < 0.01

    def test_custom_margin_weight(self):
        """Custom margin weight is used."""
        from src.algorithm.game_grade import compute_game_grade
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(margin_weight=0.30)  # Higher weight
        grade_default = compute_game_grade(is_win=True, margin=28, location="home")
        grade_custom = compute_game_grade(
            is_win=True, margin=28, location="home", config=config
        )

        # Custom should give more margin bonus
        assert grade_custom > grade_default

    def test_custom_venue_values(self):
        """Custom venue adjustments are used."""
        from src.algorithm.game_grade import compute_game_grade
        from src.data.models import AlgorithmConfig

        config = AlgorithmConfig(venue_road_win=0.15)  # Higher road bonus
        grade = compute_game_grade(
            is_win=True, margin=28, location="away", config=config
        )
        # 0.70 + 0.20 + 0.15 = 1.05
        assert abs(grade - 1.05) < 0.01


class TestGameGradeForGameResult:
    """Tests for computing game grade from GameResult objects."""

    def test_game_grade_from_result(self):
        """Compute grade from GameResult object."""
        from src.algorithm.game_grade import compute_game_grade_for_result
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
        grade = compute_game_grade_for_result(result)

        # Should be: 0.70 + margin_bonus(15) + 0.00
        expected_margin_bonus = 0.20 * math.log(16) / math.log(29)
        expected = 0.70 + expected_margin_bonus + 0.00
        assert abs(grade - expected) < 0.01

    def test_game_grade_from_result_loss(self):
        """Compute grade from losing GameResult."""
        from src.algorithm.game_grade import compute_game_grade_for_result
        from src.data.models import GameResult

        result = GameResult(
            game_id=1,
            team_id="michigan",
            opponent_id="ohio-state",
            team_score=27,
            opponent_score=42,
            location="away",
            is_win=False,
            margin=-15,
        )
        grade = compute_game_grade_for_result(result)

        # Road loss = 0.00
        assert grade == 0.00
