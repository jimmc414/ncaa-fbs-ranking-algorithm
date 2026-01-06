"""Data models for validation and prediction."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ValidatorRating:
    """Single team rating from an external validator."""

    team_id: str
    source: Literal["sp", "srs", "elo"]
    rating: float
    rank: int | None = None

    # Source-specific fields
    offense: float | None = None  # SP+ offense rating
    defense: float | None = None  # SP+ defense rating


@dataclass
class TeamValidation:
    """How one team compares across all validators."""

    team_id: str
    our_rank: int
    our_rating: float
    validator_ranks: dict[str, int] = field(default_factory=dict)  # source -> rank
    validator_ratings: dict[str, float] = field(default_factory=dict)  # source -> rating
    max_gap: int = 0  # Largest disagreement with any validator
    flagged: bool = False  # Whether this team is flagged for investigation
    pattern: str | None = None  # Detected pattern (e.g., "G5", "weak SOS")

    def get_gap(self, source: str) -> int | None:
        """Get the rank gap with a specific validator."""
        if source not in self.validator_ranks:
            return None
        return self.our_rank - self.validator_ranks[source]


@dataclass
class ValidationReport:
    """Full validation report comparing rankings to external validators."""

    season: int
    week: int | None
    correlations: dict[str, float] = field(default_factory=dict)  # source -> Spearman correlation
    flagged_teams: list[TeamValidation] = field(default_factory=list)
    flagged_count_by_source: dict[str, int] = field(default_factory=dict)  # source -> count
    patterns: list[str] = field(default_factory=list)  # Detected patterns
    total_teams_compared: int = 0


@dataclass
class SourcePrediction:
    """Prediction from a single source for a game."""

    source: str  # "our_algorithm", "vegas", "pregame_wp", "sp_implied", "elo_implied"
    predicted_winner: str | None  # team_id of predicted winner
    win_probability: float | None  # Probability of predicted winner winning
    spread: float | None = None  # Point spread (if applicable)
    weight: float = 0.0  # Weight in consensus calculation
    available: bool = True  # Whether this source has data


@dataclass
class GamePrediction:
    """Comprehensive prediction for a single game with all sources."""

    game_id: int | None
    season: int
    week: int | None
    home_team_id: str
    away_team_id: str

    # Source predictions
    source_predictions: list[SourcePrediction] = field(default_factory=list)

    # Consensus
    consensus_winner: str | None = None
    consensus_prob: float | None = None
    confidence: Literal["HIGH", "MODERATE", "LOW", "SPLIT", "UNKNOWN"] = "UNKNOWN"
    sources_agreeing: int = 0
    sources_available: int = 0

    # Rating context
    home_rating: float | None = None
    away_rating: float | None = None
    rating_gap: float | None = None

    # Actual result (if game has been played)
    actual_winner: str | None = None
    was_correct: bool | None = None  # Did consensus pick the winner?

    def get_source(self, source_name: str) -> SourcePrediction | None:
        """Get prediction from a specific source."""
        for pred in self.source_predictions:
            if pred.source == source_name:
                return pred
        return None

    @property
    def matchup_str(self) -> str:
        """Get formatted matchup string."""
        return f"{self.away_team_id} @ {self.home_team_id}"


@dataclass
class ConfidenceBreakdown:
    """Breakdown of how confidence level was determined."""

    level: Literal["HIGH", "MODERATE", "LOW", "SPLIT", "UNKNOWN"]
    sources_agreeing: int
    sources_available: int
    probability_spread: float  # Range of win probabilities across sources
    reason: str  # Human-readable explanation


@dataclass
class WeekPredictions:
    """All predictions for a week, organized by confidence."""

    season: int
    week: int
    high_confidence: list[GamePrediction] = field(default_factory=list)
    moderate_confidence: list[GamePrediction] = field(default_factory=list)
    low_confidence: list[GamePrediction] = field(default_factory=list)
    split: list[GamePrediction] = field(default_factory=list)

    # Historical accuracy (if games have been played)
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float | None = None

    def all_predictions(self) -> list[GamePrediction]:
        """Get all predictions in a flat list."""
        return (
            self.high_confidence
            + self.moderate_confidence
            + self.low_confidence
            + self.split
        )


# ============================================================================
# Vegas Upset Analysis Models
# ============================================================================


@dataclass
class VegasUpset:
    """A game where the Vegas favorite lost outright."""

    game_id: int
    season: int
    week: int

    # Vegas prediction
    vegas_favorite: str  # team_id of Vegas favorite
    vegas_underdog: str  # team_id of Vegas underdog
    spread: float  # Spread from favorite's perspective (negative)
    implied_prob: float  # Win probability from spread

    # Actual result
    actual_winner: str  # team_id of actual winner
    actual_loser: str  # team_id of actual loser
    final_margin: int  # Winner's margin of victory

    # Game context
    home_team: str  # team_id of home team
    neutral_site: bool = False

    @property
    def spread_bucket(self) -> str:
        """Categorize the spread into buckets."""
        abs_spread = abs(self.spread)
        if abs_spread < 3:
            return "toss-up (<3)"
        elif abs_spread < 7:
            return "slight (3-7)"
        elif abs_spread < 14:
            return "moderate (7-14)"
        else:
            return "heavy (14+)"

    @property
    def favorite_was_home(self) -> bool:
        """Whether the favorite was the home team."""
        return self.vegas_favorite == self.home_team


@dataclass
class UpsetStats:
    """Statistics for a category of upsets."""

    total_games: int = 0  # Total games in this category
    upsets: int = 0  # Number of upsets
    we_predicted: int = 0  # Upsets we correctly predicted

    @property
    def upset_rate(self) -> float:
        """Percentage of games that were upsets."""
        return self.upsets / self.total_games if self.total_games > 0 else 0.0

    @property
    def we_predicted_rate(self) -> float:
        """Percentage of upsets we correctly predicted."""
        return self.we_predicted / self.upsets if self.upsets > 0 else 0.0


@dataclass
class UpsetAnalysis:
    """Deep analysis of a single upset."""

    upset: VegasUpset

    # Our prediction
    our_pick: str  # team_id we predicted to win
    our_prob: float  # Our win probability for our pick
    we_predicted_upset: bool  # Did we predict the underdog to win?

    # External validators
    sp_pick: str | None = None  # SP+ implied winner
    srs_pick: str | None = None  # SRS implied winner
    elo_pick: str | None = None  # Elo implied winner
    validators_predicting_upset: int = 0  # How many validators picked the underdog?

    # Team ratings at game time
    favorite_rating: float = 0.0
    underdog_rating: float = 0.0
    rating_gap: float = 0.0  # favorite - underdog (positive = favorite better)

    # Schedule strength
    favorite_sos: float = 0.0
    underdog_sos: float = 0.0
    sos_gap: float = 0.0  # Positive = favorite had stronger schedule

    # Context
    conference_matchup: str = ""  # "P5vP5", "P5vG5", "G5vG5", etc.
    is_conference_game: bool = False
    favorite_recent_record: str = ""  # e.g., "3-2 last 5"
    underdog_recent_record: str = ""

    # Detected anomaly factors
    anomaly_factors: list[str] = field(default_factory=list)


@dataclass
class AnomalyFactor:
    """A factor that correlates with Vegas misses."""

    name: str
    description: str
    occurrences: int  # How many upsets had this factor
    we_predicted_when_present: int  # How many we got right when factor present

    @property
    def we_predicted_rate(self) -> float:
        """Rate at which we predicted upsets with this factor."""
        return self.we_predicted_when_present / self.occurrences if self.occurrences > 0 else 0.0


@dataclass
class PatternReport:
    """Aggregate analysis of Vegas upsets for a season."""

    season: int
    week: int | None = None  # None = full season

    # Overall counts
    total_games: int = 0
    games_with_lines: int = 0  # Games where we had Vegas data
    vegas_correct: int = 0
    vegas_wrong: int = 0  # Upsets

    # Our performance
    we_predicted_upset: int = 0  # Upsets we correctly picked
    we_also_wrong: int = 0  # Upsets we also missed

    # Pattern breakdowns
    by_spread_bucket: dict[str, UpsetStats] = field(default_factory=dict)
    by_conference_matchup: dict[str, UpsetStats] = field(default_factory=dict)
    by_week_range: dict[str, UpsetStats] = field(default_factory=dict)

    # Anomaly factors
    anomaly_factors: list[AnomalyFactor] = field(default_factory=list)

    # Highlighted upsets
    upsets_we_called: list[UpsetAnalysis] = field(default_factory=list)
    biggest_upsets: list[UpsetAnalysis] = field(default_factory=list)

    @property
    def vegas_accuracy(self) -> float:
        """Vegas straight-up accuracy."""
        total = self.vegas_correct + self.vegas_wrong
        return self.vegas_correct / total if total > 0 else 0.0

    @property
    def our_upset_accuracy(self) -> float:
        """Our accuracy on games Vegas got wrong."""
        return self.we_predicted_upset / self.vegas_wrong if self.vegas_wrong > 0 else 0.0

    @property
    def edge_vs_vegas(self) -> float:
        """Our edge: accuracy on upsets vs Vegas baseline."""
        # If Vegas is 87% accurate, they miss 13%
        # If we predict 30% of those misses, we have edge
        return self.our_upset_accuracy
