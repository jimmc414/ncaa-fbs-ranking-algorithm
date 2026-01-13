"""Pydantic data models for NCAA ranking system."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Team(BaseModel):
    """Represents a college football team."""

    team_id: str
    name: str
    conference: str | None = None
    division: Literal["fbs", "fcs"]

    model_config = {"frozen": True}


class Game(BaseModel):
    """Represents a single game between two teams."""

    game_id: int
    season: int
    week: int
    game_date: date
    home_team_id: str
    away_team_id: str
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    neutral_site: bool = False
    postseason: bool = False

    @model_validator(mode="after")
    def teams_must_differ(self) -> "Game":
        """Ensure home and away teams are different."""
        if self.home_team_id == self.away_team_id:
            raise ValueError("Home and away teams must be different")
        return self

    def to_results(self) -> tuple["GameResult", "GameResult"]:
        """Generate GameResult for both teams."""
        home_wins = self.home_score > self.away_score
        margin = self.home_score - self.away_score

        # Determine location for each team
        if self.neutral_site:
            home_location: Literal["home", "away", "neutral"] = "neutral"
            away_location: Literal["home", "away", "neutral"] = "neutral"
        else:
            home_location = "home"
            away_location = "away"

        home_result = GameResult(
            game_id=self.game_id,
            team_id=self.home_team_id,
            opponent_id=self.away_team_id,
            team_score=self.home_score,
            opponent_score=self.away_score,
            location=home_location,
            is_win=home_wins,
            margin=margin,
        )

        away_result = GameResult(
            game_id=self.game_id,
            team_id=self.away_team_id,
            opponent_id=self.home_team_id,
            team_score=self.away_score,
            opponent_score=self.home_score,
            location=away_location,
            is_win=not home_wins and margin != 0,  # Tie is not a win
            margin=-margin,
        )

        return home_result, away_result


class GameResult(BaseModel):
    """View of a game from one team's perspective."""

    game_id: int
    team_id: str
    opponent_id: str
    team_score: int
    opponent_score: int
    location: Literal["home", "away", "neutral"]
    is_win: bool
    margin: int  # Positive for wins, negative for losses, 0 for ties

    model_config = {"frozen": True}


class TeamRating(BaseModel):
    """Computed rating for a team in a specific season/week."""

    team_id: str
    season: int
    week: int
    rating: float
    rank: int
    games_played: int
    wins: int
    losses: int
    strength_of_schedule: float | None = None
    strength_of_victory: float | None = None
    average_game_grade: float | None = None


class AlgorithmConfig(BaseModel):
    """Configuration parameters for the ranking algorithm.

    Contains 45 configurable levers organized into categories:
    - Core (5): Convergence and game grade settings
    - Venue (4): Home/away adjustments
    - Initialization (2): Starting ratings
    - Opponent Influence (3): How opponent quality affects ratings
    - Conference (5): P5/G5/FCS adjustments
    - Schedule Strength (5): SOS calculations and thresholds
    - Quality Tiers (6): Bonuses/penalties for elite/bad performance
    - Recency (3): Weight recent games more
    - Prior (3): Use historical data as starting point
    - Margin Curve (1): How margin of victory scales
    - Tiebreakers (2): How to break rating ties
    - Filters (4): Which games to include
    """

    # ========== CORE (5) ==========
    # Convergence settings
    max_iterations: int = Field(default=100, gt=0)
    convergence_threshold: float = Field(default=0.0001, gt=0)

    # Game grade weights
    win_base: float = 0.70
    margin_weight: float = 0.20
    margin_cap: int = Field(default=28, gt=0)

    # ========== VENUE (4) ==========
    venue_road_win: float = 0.10
    venue_neutral_win: float = 0.05
    venue_home_loss: float = -0.03
    venue_neutral_loss: float = -0.01

    # ========== INITIALIZATION (2) ==========
    initial_rating: float = 0.500
    fcs_fixed_rating: float | None = 0.20

    # ========== OPPONENT INFLUENCE (3) ==========
    opponent_weight: float = 1.0  # Multiplier on opponent rating
    loss_opponent_factor: float = 1.0  # Scale for loss penalty (penalty = (1-opp_rating) * factor)
    second_order_weight: float = 0.0  # Weight of opponent's SOS

    # ========== CONFERENCE (5) ==========
    enable_conference_adj: bool = False
    conference_method: Literal["empirical", "manual"] = "empirical"
    p5_multiplier: float = 1.0
    g5_multiplier: float = 1.0
    fcs_multiplier: float = 0.5

    # ========== SCHEDULE STRENGTH (5) ==========
    sos_adjustment_weight: float = 0.0  # Post-hoc SOS adjustment
    sos_method: Literal["mean", "median"] = "mean"
    min_sos_top_10: float = 0.0  # Min SOS to be ranked top 10
    min_sos_top_25: float = 0.0  # Min SOS to be ranked top 25
    min_p5_games_top_10: int = 0  # Min P5 games for top 10

    # ========== QUALITY TIERS (6) ==========
    enable_quality_tiers: bool = False
    elite_threshold: float = 0.80  # Rating to be considered elite
    good_threshold: float = 0.55  # Rating to be considered good
    bad_threshold: float = 0.35  # Rating to be considered bad
    elite_win_bonus: float = 0.0  # Bonus for beating elite teams
    bad_loss_penalty: float = 0.0  # Penalty for losing to bad teams

    # ========== RECENCY (3) ==========
    enable_recency: bool = False
    recency_half_life: int = Field(default=8, gt=0)  # Weeks for weight to halve
    recency_min_weight: float = 0.5  # Floor for oldest games

    # ========== PRIOR (3) ==========
    enable_prior: bool = False
    prior_weight: float = 0.0  # How much to weight prior vs games
    prior_decay_weeks: int = Field(default=8, gt=0)  # Weeks for prior to fade

    # ========== MARGIN CURVE (1) ==========
    margin_curve: Literal["log", "linear", "sqrt"] = "log"

    # ========== TIEBREAKERS (2) ==========
    tie_threshold: float = 0.001  # Rating difference to consider a tie
    tiebreaker_order: list[str] = Field(
        default=["h2h", "sov", "common", "away"]
    )

    # ========== FILTERS (5) ==========
    include_postseason: bool = True
    postseason_weight: float = 1.0  # Weight for bowl/playoff games
    include_fcs_games: bool = True
    exclude_fcs_from_rankings: bool = False  # Hide FCS teams from output
    min_games_to_rank: int = 1  # Min games to appear in rankings


class RatingExplanation(BaseModel):
    """Detailed breakdown of how a team's rating was computed."""

    team_id: str
    season: int
    week: int
    final_rating: float
    normalized_rating: float
    rank: int
    games: list[dict]  # List of game contributions
    total_contribution: float
    iterations_to_converge: int


class HistoricalRanking(BaseModel):
    """Historical poll ranking (AP, CFP, etc.)."""

    poll_type: Literal["ap", "cfp", "coaches"]
    season: int
    week: int
    team_id: str
    rank: int
    points: int | None = None
    first_place_votes: int | None = None


class ComparisonResult(BaseModel):
    """Result of comparing algorithm rankings to a poll."""

    season: int
    week: int
    poll_type: str
    spearman_correlation: float
    kendall_tau: float | None = None
    teams_compared: int
    overranked: list[dict]  # Teams algorithm ranks higher than poll
    underranked: list[dict]  # Teams algorithm ranks lower than poll
