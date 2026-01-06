"""Unit tests for Vegas upset analyzer."""

import pytest
from unittest.mock import MagicMock

from src.data.models import Game, TeamRating
from src.validation.models import (
    AnomalyFactor,
    PatternReport,
    UpsetAnalysis,
    UpsetStats,
    VegasUpset,
    ValidatorRating,
)
from src.validation.upset_analyzer import (
    UpsetAnalyzer,
    get_conference_tier,
    get_week_range,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_conference_tier_p5(self):
        """Should classify P5 conferences correctly."""
        assert get_conference_tier("SEC") == "P5"
        assert get_conference_tier("Big Ten") == "P5"
        assert get_conference_tier("Big 12") == "P5"
        assert get_conference_tier("ACC") == "P5"
        assert get_conference_tier("Pac-12") == "P5"

    def test_get_conference_tier_g5(self):
        """Should classify G5 conferences correctly."""
        assert get_conference_tier("American Athletic") == "G5"
        assert get_conference_tier("Mountain West") == "G5"
        assert get_conference_tier("Sun Belt") == "G5"
        assert get_conference_tier("MAC") == "G5"
        assert get_conference_tier("Conference USA") == "G5"

    def test_get_conference_tier_independent(self):
        """Should classify FBS Independents correctly."""
        assert get_conference_tier("FBS Independents") == "IND"

    def test_get_conference_tier_fcs(self):
        """Should default to FCS for unknown or None."""
        assert get_conference_tier(None) == "FCS"
        assert get_conference_tier("Unknown Conference") == "FCS"

    def test_get_week_range_early(self):
        """Should classify weeks 1-4 as early."""
        assert get_week_range(1) == "early (1-4)"
        assert get_week_range(4) == "early (1-4)"

    def test_get_week_range_mid(self):
        """Should classify weeks 5-9 as mid."""
        assert get_week_range(5) == "mid (5-9)"
        assert get_week_range(9) == "mid (5-9)"

    def test_get_week_range_late(self):
        """Should classify weeks 10+ as late."""
        assert get_week_range(10) == "late (10+)"
        assert get_week_range(15) == "late (10+)"


class TestVegasUpsetModel:
    """Tests for VegasUpset dataclass."""

    def test_spread_bucket_tossup(self):
        """Spread < 3 should be toss-up."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-2.5, implied_prob=0.55,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=7, home_team="team-a",
        )
        assert upset.spread_bucket == "toss-up (<3)"

    def test_spread_bucket_slight(self):
        """Spread 3-7 should be slight."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-5.5, implied_prob=0.65,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=7, home_team="team-a",
        )
        assert upset.spread_bucket == "slight (3-7)"

    def test_spread_bucket_moderate(self):
        """Spread 7-14 should be moderate."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-10.5, implied_prob=0.78,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=7, home_team="team-a",
        )
        assert upset.spread_bucket == "moderate (7-14)"

    def test_spread_bucket_heavy(self):
        """Spread 14+ should be heavy."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-17.5, implied_prob=0.88,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=7, home_team="team-a",
        )
        assert upset.spread_bucket == "heavy (14+)"

    def test_favorite_was_home(self):
        """Should correctly identify if favorite was home."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=7, home_team="team-a",
        )
        assert upset.favorite_was_home is True

    def test_favorite_was_away(self):
        """Should correctly identify if favorite was away."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=7, home_team="team-b",
        )
        assert upset.favorite_was_home is False


class TestUpsetStatsModel:
    """Tests for UpsetStats dataclass."""

    def test_upset_rate_calculation(self):
        """Should calculate upset rate correctly."""
        stats = UpsetStats(total_games=100, upsets=15, we_predicted=5)
        assert stats.upset_rate == 0.15

    def test_upset_rate_zero_games(self):
        """Should handle zero games gracefully."""
        stats = UpsetStats(total_games=0, upsets=0, we_predicted=0)
        assert stats.upset_rate == 0.0

    def test_we_predicted_rate_calculation(self):
        """Should calculate prediction rate correctly."""
        stats = UpsetStats(total_games=100, upsets=20, we_predicted=8)
        assert stats.we_predicted_rate == 0.4

    def test_we_predicted_rate_zero_upsets(self):
        """Should handle zero upsets gracefully."""
        stats = UpsetStats(total_games=100, upsets=0, we_predicted=0)
        assert stats.we_predicted_rate == 0.0


class TestPatternReportModel:
    """Tests for PatternReport dataclass."""

    def test_vegas_accuracy_calculation(self):
        """Should calculate Vegas accuracy correctly."""
        report = PatternReport(season=2024, vegas_correct=85, vegas_wrong=15)
        assert report.vegas_accuracy == 0.85

    def test_vegas_accuracy_zero_games(self):
        """Should handle zero games gracefully."""
        report = PatternReport(season=2024, vegas_correct=0, vegas_wrong=0)
        assert report.vegas_accuracy == 0.0

    def test_our_upset_accuracy_calculation(self):
        """Should calculate our upset accuracy correctly."""
        report = PatternReport(
            season=2024,
            vegas_wrong=20,
            we_predicted_upset=6,
            we_also_wrong=14,
        )
        assert report.our_upset_accuracy == 0.3

    def test_our_upset_accuracy_zero_upsets(self):
        """Should handle zero upsets gracefully."""
        report = PatternReport(season=2024, vegas_wrong=0, we_predicted_upset=0)
        assert report.our_upset_accuracy == 0.0


class TestAnomalyFactorModel:
    """Tests for AnomalyFactor dataclass."""

    def test_we_predicted_rate(self):
        """Should calculate prediction rate correctly."""
        factor = AnomalyFactor(
            name="weak_sos",
            description="Favorite had weak SOS",
            occurrences=10,
            we_predicted_when_present=4,
        )
        assert factor.we_predicted_rate == 0.4

    def test_we_predicted_rate_zero_occurrences(self):
        """Should handle zero occurrences gracefully."""
        factor = AnomalyFactor(
            name="weak_sos",
            description="Favorite had weak SOS",
            occurrences=0,
            we_predicted_when_present=0,
        )
        assert factor.we_predicted_rate == 0.0


class TestUpsetAnalyzer:
    """Tests for UpsetAnalyzer class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock CFBDataClient."""
        return MagicMock()

    @pytest.fixture
    def sample_games(self):
        """Create sample games for testing."""
        from datetime import date
        return [
            Game(
                game_id=1, season=2024, week=5, game_date=date(2024, 10, 5),
                home_team_id="ohio-state", away_team_id="michigan",
                home_score=21, away_score=24,  # Michigan wins as underdog
            ),
            Game(
                game_id=2, season=2024, week=5, game_date=date(2024, 10, 5),
                home_team_id="alabama", away_team_id="georgia",
                home_score=28, away_score=21,  # Alabama wins as favorite
            ),
        ]

    @pytest.fixture
    def sample_betting_lines(self):
        """Create sample betting lines."""
        return [
            {"game_id": 1, "spread": -7.0},  # Ohio State favored by 7
            {"game_id": 2, "spread": -3.0},  # Alabama favored by 3 (favorite wins)
        ]

    @pytest.fixture
    def sample_ratings(self):
        """Create sample team ratings."""
        return {
            "ohio-state": TeamRating(
                team_id="ohio-state", season=2024, week=15,
                rating=0.85, rank=2, wins=11, losses=1, games_played=12,
                strength_of_schedule=0.65,
            ),
            "michigan": TeamRating(
                team_id="michigan", season=2024, week=15,
                rating=0.78, rank=8, wins=10, losses=2, games_played=12,
                strength_of_schedule=0.60,
            ),
        }

    @pytest.mark.asyncio
    async def test_find_upsets_identifies_underdog_wins(self, mock_client, sample_games, sample_betting_lines):
        """Should identify games where underdog won."""
        analyzer = UpsetAnalyzer(mock_client)
        upsets = await analyzer.find_upsets(sample_games, sample_betting_lines, min_spread=3.0)

        # Should find 1 upset: Michigan beat Ohio State (7-point favorite)
        assert len(upsets) == 1
        assert upsets[0].vegas_favorite == "ohio-state"
        assert upsets[0].vegas_underdog == "michigan"
        assert upsets[0].actual_winner == "michigan"

    @pytest.mark.asyncio
    async def test_find_upsets_respects_min_spread(self, mock_client, sample_games):
        """Should filter out games below min_spread."""
        betting_lines = [
            {"game_id": 1, "spread": -2.5},  # Below min_spread
        ]
        analyzer = UpsetAnalyzer(mock_client)
        upsets = await analyzer.find_upsets(sample_games, betting_lines, min_spread=3.0)

        assert len(upsets) == 0

    @pytest.mark.asyncio
    async def test_find_upsets_skips_games_without_lines(self, mock_client, sample_games):
        """Should skip games without betting lines."""
        betting_lines = []  # No lines
        analyzer = UpsetAnalyzer(mock_client)
        upsets = await analyzer.find_upsets(sample_games, betting_lines, min_spread=3.0)

        assert len(upsets) == 0

    def test_analyze_upset_calculates_prediction(self, mock_client, sample_ratings):
        """Should analyze upset and determine if we predicted it."""
        analyzer = UpsetAnalyzer(mock_client)

        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="ohio-state", vegas_underdog="michigan",
            spread=-7.0, implied_prob=0.70,
            actual_winner="michigan", actual_loser="ohio-state",
            final_margin=3, home_team="ohio-state",
        )

        teams = {
            "ohio-state": {"conference": "Big Ten"},
            "michigan": {"conference": "Big Ten"},
        }

        analysis = analyzer.analyze_upset(upset, sample_ratings, teams)

        assert analysis.upset == upset
        assert analysis.our_pick in ["ohio-state", "michigan"]
        assert 0 <= analysis.our_prob <= 1
        # Since OSU has higher rating, we likely picked them
        assert analysis.we_predicted_upset is False
        assert analysis.is_conference_game is True

    def test_identify_anomaly_factors_weak_sos(self, mock_client):
        """Should identify weak favorite SOS as anomaly factor."""
        analyzer = UpsetAnalyzer(mock_client)

        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=3, home_team="team-a",
        )

        factors = analyzer._identify_anomaly_factors(
            upset=upset,
            rating_gap=0.10,
            sos_gap=-0.05,
            fav_sos=0.40,  # Weak SOS
            we_predicted=False,
            conference_matchup="P5vP5",
        )

        assert "weak_favorite_sos" in factors

    def test_identify_anomaly_factors_small_rating_gap(self, mock_client):
        """Should identify small rating gap as anomaly factor."""
        analyzer = UpsetAnalyzer(mock_client)

        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=3, home_team="team-a",
        )

        factors = analyzer._identify_anomaly_factors(
            upset=upset,
            rating_gap=0.03,  # Small gap
            sos_gap=0.10,
            fav_sos=0.60,
            we_predicted=True,
            conference_matchup="P5vP5",
        )

        assert "small_rating_gap" in factors
        assert "we_predicted_upset" in factors

    def test_identify_anomaly_factors_late_season(self, mock_client):
        """Should identify late season as anomaly factor."""
        analyzer = UpsetAnalyzer(mock_client)

        upset = VegasUpset(
            game_id=1, season=2024, week=12,  # Late season
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=3, home_team="team-a",
        )

        factors = analyzer._identify_anomaly_factors(
            upset=upset,
            rating_gap=0.15,
            sos_gap=0.10,
            fav_sos=0.60,
            we_predicted=False,
            conference_matchup="P5vP5",
        )

        assert "late_season" in factors

    def test_generate_pattern_report(self, mock_client):
        """Should generate pattern report from upset analyses."""
        analyzer = UpsetAnalyzer(mock_client)

        # Create sample upsets
        upset1 = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=3, home_team="team-a",
        )

        analysis1 = UpsetAnalysis(
            upset=upset1,
            our_pick="team-b",
            our_prob=0.52,
            we_predicted_upset=True,
            conference_matchup="P5vP5",
            anomaly_factors=["weak_favorite_sos"],
        )

        upset2 = VegasUpset(
            game_id=2, season=2024, week=8,
            vegas_favorite="team-c", vegas_underdog="team-d",
            spread=-10.0, implied_prob=0.78,
            actual_winner="team-d", actual_loser="team-c",
            final_margin=7, home_team="team-c",
        )

        analysis2 = UpsetAnalysis(
            upset=upset2,
            our_pick="team-c",
            our_prob=0.65,
            we_predicted_upset=False,
            conference_matchup="P5vG5",
            anomaly_factors=["small_rating_gap", "road_favorite"],
        )

        report = analyzer.generate_pattern_report(
            upsets=[analysis1, analysis2],
            all_games_count=100,
            games_with_lines_count=80,
            season=2024,
        )

        assert report.season == 2024
        assert report.total_games == 100
        assert report.games_with_lines == 80
        assert report.vegas_wrong == 2
        assert report.vegas_correct == 78
        assert report.we_predicted_upset == 1
        assert report.we_also_wrong == 1
        assert len(report.by_spread_bucket) > 0
        assert len(report.by_conference_matchup) > 0


class TestUpsetAnalysisModel:
    """Tests for UpsetAnalysis dataclass."""

    def test_basic_creation(self):
        """Should create UpsetAnalysis with all fields."""
        upset = VegasUpset(
            game_id=1, season=2024, week=5,
            vegas_favorite="team-a", vegas_underdog="team-b",
            spread=-7.0, implied_prob=0.70,
            actual_winner="team-b", actual_loser="team-a",
            final_margin=3, home_team="team-a",
        )

        analysis = UpsetAnalysis(
            upset=upset,
            our_pick="team-a",
            our_prob=0.62,
            we_predicted_upset=False,
            favorite_rating=0.75,
            underdog_rating=0.68,
            rating_gap=0.07,
            favorite_sos=0.55,
            underdog_sos=0.48,
            sos_gap=0.07,
            conference_matchup="P5vP5",
            is_conference_game=True,
            anomaly_factors=["small_rating_gap"],
        )

        assert analysis.upset == upset
        assert analysis.our_pick == "team-a"
        assert analysis.we_predicted_upset is False
        assert analysis.rating_gap == 0.07
        assert "small_rating_gap" in analysis.anomaly_factors
