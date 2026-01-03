"""Tests for the convergence algorithm."""

import pytest
from datetime import date


class TestInitialization:
    """Tests for rating initialization."""

    def test_all_teams_start_at_0_5(self, sample_games):
        """All teams initialized to 0.500 rating."""
        from src.algorithm.convergence import initialize_ratings

        ratings = initialize_ratings(sample_games)
        assert all(r == 0.5 for r in ratings.values())

    def test_includes_all_teams(self, sample_games):
        """Initialization includes every team from games."""
        from src.algorithm.convergence import initialize_ratings

        ratings = initialize_ratings(sample_games)
        team_ids = set()
        for game in sample_games:
            team_ids.add(game.home_team_id)
            team_ids.add(game.away_team_id)
        assert set(ratings.keys()) == team_ids

    def test_empty_games_list(self):
        """Empty games list returns empty ratings."""
        from src.algorithm.convergence import initialize_ratings

        ratings = initialize_ratings([])
        assert ratings == {}


class TestBuildGameResults:
    """Tests for building game results lookup."""

    def test_build_game_results(self, sample_games):
        """Game results are correctly built for each team."""
        from src.algorithm.convergence import build_game_results

        results = build_game_results(sample_games)

        # Each team should have their games listed
        for game in sample_games:
            assert game.home_team_id in results
            assert game.away_team_id in results

    def test_results_perspective_correct(self, sample_games):
        """Each result shows correct perspective."""
        from src.algorithm.convergence import build_game_results

        results = build_game_results(sample_games)

        # Check first game: ohio-state vs akron, 52-6
        osu_results = results.get("ohio-state", [])
        assert len(osu_results) >= 1

        osu_game = osu_results[0]
        assert osu_game.team_id == "ohio-state"
        assert osu_game.opponent_id == "akron"
        assert osu_game.is_win is True


class TestSingleIteration:
    """Tests for single iteration of the algorithm."""

    def test_deterministic_ordering(self, four_team_round_robin, algorithm_config):
        """Teams processed in sorted order for determinism."""
        from src.algorithm.convergence import (
            initialize_ratings,
            build_game_results,
            iterate_once,
        )

        ratings = initialize_ratings(four_team_round_robin)
        game_results = build_game_results(four_team_round_robin)

        # Run twice, verify identical results
        result1 = iterate_once(ratings, game_results, algorithm_config)
        result2 = iterate_once(ratings, game_results, algorithm_config)

        assert result1 == result2

    def test_win_adds_opponent_rating(self, algorithm_config):
        """Winning contribution includes +opponent_rating."""
        from src.data.models import Game
        from src.algorithm.convergence import (
            initialize_ratings,
            build_game_results,
            iterate_once,
        )

        # Simple 2-team scenario: A beats B
        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="B",
                home_score=28,
                away_score=14,
            )
        ]

        ratings = initialize_ratings(games)
        game_results = build_game_results(games)
        new_ratings = iterate_once(ratings, game_results, algorithm_config)

        # Team A won, should be rated higher
        assert new_ratings["A"] > new_ratings["B"]

    def test_loss_subtracts_opponent_rating(self, algorithm_config):
        """Losing contribution includes -opponent_rating."""
        from src.data.models import Game
        from src.algorithm.convergence import (
            initialize_ratings,
            build_game_results,
            iterate_once,
        )

        # Simple 2-team scenario: A beats B
        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="B",
                home_score=28,
                away_score=14,
            )
        ]

        ratings = initialize_ratings(games)
        game_results = build_game_results(games)
        new_ratings = iterate_once(ratings, game_results, algorithm_config)

        # Team B lost, should have negative contribution from loss
        # This is the CRITICAL test: losses subtract opponent rating
        assert new_ratings["B"] < 0.5  # Initial rating was 0.5


class TestConvergence:
    """Tests for the full convergence algorithm."""

    def test_converges_within_50_iterations(self, four_team_round_robin, algorithm_config):
        """Algorithm converges within max iterations."""
        from src.algorithm.convergence import converge

        result = converge(four_team_round_robin, algorithm_config)
        assert result.iterations < 50

    def test_max_delta_below_threshold(self, four_team_round_robin, algorithm_config):
        """Final max delta below convergence threshold."""
        from src.algorithm.convergence import converge

        result = converge(four_team_round_robin, algorithm_config)
        assert result.final_max_delta < algorithm_config.convergence_threshold

    def test_four_team_round_robin_order(self, four_team_round_robin, algorithm_config):
        """Canonical test: A > B > C > D."""
        from src.algorithm.convergence import converge

        result = converge(four_team_round_robin, algorithm_config)

        # Sort by rating descending
        sorted_teams = sorted(
            result.ratings.items(), key=lambda x: x[1], reverse=True
        )
        team_order = [t[0] for t in sorted_teams]

        # A beats B, C, D; B beats C, D; C beats D; D loses all
        assert team_order == ["A", "B", "C", "D"]

    def test_rating_gaps_exist(self, four_team_round_robin, algorithm_config):
        """Rating gaps reflect transitive advantage."""
        from src.algorithm.convergence import converge

        result = converge(four_team_round_robin, algorithm_config)

        # Each team should have different rating
        ratings = sorted(result.ratings.values(), reverse=True)
        for i in range(len(ratings) - 1):
            assert ratings[i] > ratings[i + 1]

    def test_convergence_result_has_history(self, four_team_round_robin, algorithm_config):
        """ConvergenceResult includes delta history."""
        from src.algorithm.convergence import converge

        result = converge(four_team_round_robin, algorithm_config)
        assert len(result.delta_history) == result.iterations
        assert result.delta_history[-1] == result.final_max_delta

    def test_delta_decreases_over_iterations(self, four_team_round_robin, algorithm_config):
        """Delta generally decreases as algorithm converges."""
        from src.algorithm.convergence import converge

        result = converge(four_team_round_robin, algorithm_config)

        # First delta should be larger than last delta
        if len(result.delta_history) > 1:
            assert result.delta_history[0] > result.delta_history[-1]


class TestFCSHandling:
    """Tests for FCS team handling."""

    def test_fcs_fixed_rating(self, algorithm_config):
        """FCS teams use fixed rating when configured."""
        from src.data.models import Game
        from src.algorithm.convergence import converge

        # FBS team plays FCS team
        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="ohio-state",
                away_team_id="youngstown-state",
                home_score=56,
                away_score=0,
            )
        ]

        algorithm_config.fcs_fixed_rating = 0.20
        fcs_teams = {"youngstown-state"}

        result = converge(games, algorithm_config, fcs_teams=fcs_teams)

        assert result.ratings["youngstown-state"] == 0.20

    def test_fcs_no_fixed_rating(self, algorithm_config):
        """FCS teams iterate normally when fcs_fixed_rating is None."""
        from src.data.models import Game
        from src.algorithm.convergence import converge

        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="ohio-state",
                away_team_id="youngstown-state",
                home_score=56,
                away_score=0,
            )
        ]

        algorithm_config.fcs_fixed_rating = None
        fcs_teams = {"youngstown-state"}

        result = converge(games, algorithm_config, fcs_teams=fcs_teams)

        # FCS team should have iterated to some value (not exactly 0.20)
        assert result.ratings["youngstown-state"] != 0.20


class TestNormalization:
    """Tests for rating normalization."""

    def test_normalize_to_0_1_range(self):
        """Normalized ratings in [0, 1]."""
        from src.algorithm.convergence import normalize_ratings

        ratings = {"A": 1.5, "B": 0.5, "C": -0.5}
        normalized = normalize_ratings(ratings)

        assert min(normalized.values()) == 0.0
        assert max(normalized.values()) == 1.0

    def test_relative_ordering_preserved(self):
        """Normalization doesn't change ordering."""
        from src.algorithm.convergence import normalize_ratings

        ratings = {"A": 1.5, "B": 0.5, "C": -0.5}
        normalized = normalize_ratings(ratings)

        assert normalized["A"] > normalized["B"] > normalized["C"]

    def test_normalize_equal_ratings(self):
        """Equal ratings normalize to 0.5."""
        from src.algorithm.convergence import normalize_ratings

        ratings = {"A": 0.5, "B": 0.5, "C": 0.5}
        normalized = normalize_ratings(ratings)

        assert all(v == 0.5 for v in normalized.values())


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_game_winner_rated_higher_after_one_iteration(self, algorithm_config):
        """Single game: winner rated higher after first iteration."""
        from src.data.models import Game
        from src.algorithm.convergence import initialize_ratings, build_game_results, iterate_once

        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="B",
                home_score=28,
                away_score=14,
            )
        ]

        # Note: Single-game scenarios don't converge due to circular dependency
        # but after one iteration, winner should be rated higher
        ratings = initialize_ratings(games)
        game_results = build_game_results(games)
        new_ratings = iterate_once(ratings, game_results, algorithm_config)

        assert new_ratings["A"] > new_ratings["B"]

    def test_blowout_vs_close_game(self, algorithm_config):
        """Blowout win rated higher than close win (symmetric opponents)."""
        from src.algorithm.game_grade import compute_game_grade

        # Test margin bonus directly - larger margin = higher game grade
        blowout_grade = compute_game_grade(is_win=True, margin=28, location="home")
        close_grade = compute_game_grade(is_win=True, margin=1, location="home")

        # Blowout should have higher game grade
        assert blowout_grade > close_grade

        # The difference should be approximately the margin bonus difference
        # (0.20 for 28 pts vs ~0.04 for 1 pt)
        assert blowout_grade - close_grade > 0.15

    def test_home_vs_road_win(self, algorithm_config):
        """Road win valued more than home win (connected graph)."""
        from src.data.models import Game
        from src.algorithm.convergence import converge

        # Create connected graph where we can compare home vs road wins
        # A wins at home vs C, B wins on road vs D
        # C and D play each other to connect the graph
        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="C",  # A is home
                home_score=28,
                away_score=14,
            ),
            Game(
                game_id=2,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="D",
                away_team_id="B",  # B is away
                home_score=14,
                away_score=28,
            ),
            # Connect the graph: C beats D (equal opponents)
            Game(
                game_id=3,
                season=2024,
                week=2,
                game_date=date(2024, 9, 7),
                home_team_id="C",
                away_team_id="D",
                home_score=17,
                away_score=14,
            ),
        ]

        result = converge(games, algorithm_config)

        # B should be rated higher than A due to road win bonus
        assert result.ratings["B"] > result.ratings["A"]

    def test_max_iterations_reached(self, algorithm_config):
        """Algorithm stops at max iterations if needed."""
        from src.data.models import Game
        from src.algorithm.convergence import converge

        # Very low max iterations to force hitting limit
        algorithm_config.max_iterations = 2
        algorithm_config.convergence_threshold = 0.00000001  # Very tight

        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="B",
                home_score=28,
                away_score=14,
            )
        ]

        result = converge(games, algorithm_config)
        assert result.iterations == 2


class TestLossPenalty:
    """CRITICAL: Tests that verify losses subtract opponent rating."""

    def test_loss_contribution_formula(self, algorithm_config):
        """Verify loss contribution uses game_grade - opponent_rating."""
        from src.data.models import Game
        from src.algorithm.convergence import (
            initialize_ratings,
            build_game_results,
            iterate_once,
        )

        # Simple 2-team game where B loses to A
        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="B",
                home_score=28,
                away_score=14,
            ),
        ]

        ratings = initialize_ratings(games)
        game_results = build_game_results(games)
        new_ratings = iterate_once(ratings, game_results, algorithm_config)

        # B lost, so B's rating should be:
        # game_grade(loss, away) - A_rating = 0 - 0.5 = -0.5
        assert new_ratings["B"] < 0  # Negative rating due to subtraction
        assert new_ratings["B"] == -0.5  # Exactly -0.5 (game_grade 0 minus initial 0.5)

    def test_losses_always_hurt(self, algorithm_config):
        """Verify that losses always reduce rating (no 'quality loss' bonus)."""
        from src.data.models import Game
        from src.algorithm.convergence import converge

        # Connected scenario with clear hierarchy:
        # C beats A beats B, ensuring connectivity
        games = [
            # A beats B
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="A",
                away_team_id="B",
                home_score=28,
                away_score=14,
            ),
            # C beats A
            Game(
                game_id=2,
                season=2024,
                week=2,
                game_date=date(2024, 9, 7),
                home_team_id="C",
                away_team_id="A",
                home_score=21,
                away_score=17,
            ),
            # C also beats B to fully connect the graph
            Game(
                game_id=3,
                season=2024,
                week=3,
                game_date=date(2024, 9, 14),
                home_team_id="C",
                away_team_id="B",
                home_score=35,
                away_score=7,
            ),
        ]

        result = converge(games, algorithm_config)

        # C should be rated highest (undefeated, beat A and B)
        # A should be higher than B (A beat B, only lost to C)
        assert result.ratings["C"] > result.ratings["A"]
        assert result.ratings["A"] > result.ratings["B"]
