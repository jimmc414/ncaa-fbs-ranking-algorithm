# PROJECT_SPEC.md

**NCAA FBS Ranking Algorithm v2.0**

A bias-free ranking system for NCAA FBS football using iterative convergence, multi-source prediction consensus, and diagnostic feedback loops.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Algorithm](#2-core-algorithm)
3. [Prediction System](#3-prediction-system)
4. [Validation System](#4-validation-system)
5. [Configuration](#5-configuration)
6. [CLI Commands](#6-cli-commands)
7. [Data Models](#7-data-models)
8. [Project Structure](#8-project-structure)
9. [Diagnostic Feedback Loop](#9-diagnostic-feedback-loop)
10. [API Integration](#10-api-integration)
11. [Performance Metrics](#11-performance-metrics)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Overview

### What This System Does

1. **Generates team rankings** using iterative convergence (PageRank-style)
2. **Predicts game outcomes** via 5-source weighted consensus
3. **Validates rankings** against external systems (SP+, SRS, Elo)
4. **Analyzes Vegas upsets** to identify patterns and edge cases
5. **Provides diagnostics** for parameter tuning via Brier scores and calibration curves

### Design Principles

- **Determinism**: Same inputs always produce identical outputs
- **Transparency**: Every ranking can be decomposed into constituent factors
- **Mathematical soundness**: Quality losses hurt less, not more (see [Loss Formula](#22-loss-formula))
- **Validation-driven**: Rankings are compared against external validators to detect anomalies

### Key Insight

You cannot know how good a team is until you know how good their opponents are.
But opponent quality depends on *their* opponents, recursively.
The algorithm resolves this through iterative convergence until ratings stabilize.

---

## 2. Core Algorithm

### 2.1 Rating Formula

For each team, the rating is the mean contribution across all games:

```
R_t = (1/N) * SUM(contribution_i)
```

Where contribution depends on game outcome:

```python
if is_win:
    contribution = game_grade + (opponent_rating * opponent_weight)
else:
    # Quality losses hurt less than bad losses
    opponent_weakness = 1.0 - (opponent_rating * opponent_weight)
    contribution = game_grade - (opponent_weakness * loss_opponent_factor)
```

### 2.2 Loss Formula

**Critical insight**: Losing to a good team should hurt less than losing to a bad team.

With default `loss_opponent_factor=1.0`:

| Loss to team rated | Penalty calculation | Result |
|--------------------|---------------------|--------|
| 0.90 (elite)       | (1 - 0.90) * 1.0    | -0.10 (small) |
| 0.50 (average)     | (1 - 0.50) * 1.0    | -0.50 (moderate) |
| 0.10 (bad)         | (1 - 0.10) * 1.0    | -0.90 (large) |

Losses always hurt, but quality losses hurt less.

**Why not `+opponent_rating` for losses?** That would reward losing to good teams (they add rating). The inverted formula correctly penalizes losses while scaling the penalty by opponent weakness.

### 2.3 Game Grade Calculation

```
GameGrade = BaseResult + MarginBonus + VenueAdjustment
```

**Base Result:**
- Win: `0.70`
- Loss: `0.00`

**Margin Bonus (wins only):**

```python
import math

def margin_bonus(margin: int, margin_weight: float = 0.20, cap: int = 28) -> float:
    if margin <= 0:
        return 0.0
    capped_margin = min(margin, cap)
    return margin_weight * math.log(1 + capped_margin) / math.log(1 + cap)
```

The logarithmic curve rewards the first touchdown of margin more than the third:

| Margin | Bonus (default settings) |
|--------|--------------------------|
| 1 pt   | 0.021 |
| 7 pts  | 0.112 |
| 14 pts | 0.156 |
| 21 pts | 0.183 |
| 28 pts | 0.200 (cap) |
| 42 pts | 0.200 (no extra) |

**Venue Adjustment:**

| Scenario | Adjustment |
|----------|------------|
| Road win | +0.10 |
| Neutral site win | +0.05 |
| Home win | +0.00 |
| Home loss | -0.03 |
| Neutral site loss | -0.01 |
| Road loss | +0.00 |

### 2.4 Example Game Grade Calculations

| Scenario | Base | Margin | Venue | Total |
|----------|------|--------|-------|-------|
| Road blowout win (W by 28) | 0.70 | 0.20 | +0.10 | **1.00** |
| Home squeaker win (W by 1) | 0.70 | 0.02 | +0.00 | **0.72** |
| Neutral close win (W by 7) | 0.70 | 0.11 | +0.05 | **0.86** |
| Home loss (L by 3) | 0.00 | 0.00 | -0.03 | **-0.03** |
| Road loss (L by 21) | 0.00 | 0.00 | +0.00 | **0.00** |

### 2.5 Iterative Convergence

```python
def converge(games, config):
    # 1. Initialize all teams to 0.500
    ratings = {team: 0.5 for team in all_teams}

    # 2. Iterate until stable
    for iteration in range(config.max_iterations):
        new_ratings = {}

        # CRITICAL: Process in sorted order for determinism
        for team_id in sorted(ratings.keys()):
            new_ratings[team_id] = compute_rating(team_id, ratings, games)

        # Check convergence
        max_delta = max(abs(new_ratings[t] - ratings[t]) for t in ratings)
        if max_delta < config.convergence_threshold:
            break

        ratings = new_ratings

    # 3. Normalize to [0, 1] range AFTER convergence
    return normalize(ratings)
```

**Typical convergence:** 20-40 iterations with default threshold (0.0001).

**Key requirements:**
- Process teams in sorted order (determinism)
- Normalize AFTER convergence, not during
- FCS teams can use fixed rating (0.20) or iterate with FBS

### 2.6 Optional Modifiers

The core algorithm can be augmented with:

| Modifier | Purpose | Config Params |
|----------|---------|---------------|
| Conference multipliers | Adjust opponent value by P5/G5/FCS tier | `p5_multiplier`, `g5_multiplier`, `fcs_multiplier` |
| Quality tier bonuses | Extra credit for elite wins, penalty for bad losses | `elite_win_bonus`, `bad_loss_penalty` |
| Recency weighting | Recent games count more | `recency_half_life`, `recency_min_weight` |
| SOS adjustment | Post-hoc adjustment for schedule strength | `sos_adjustment_weight` |
| Second-order SOS | Factor in opponent's opponents | `second_order_weight` |

---

## 3. Prediction System

### 3.1 Multi-Source Consensus

Predictions blend 5 sources with empirically-tuned weights:

```python
CONSENSUS_WEIGHTS = {
    "vegas": 0.35,          # Vegas betting lines
    "our_algorithm": 0.25,  # Our iterative ratings
    "pregame_wp": 0.20,     # Market-derived win probability
    "sp_implied": 0.10,     # SP+ ratings implied probability
    "elo_implied": 0.10,    # Elo ratings implied probability
}
```

**Why Vegas weighted highest?** Historically most accurate single predictor. Markets aggregate information efficiently.

**Missing sources:** Weights are rebalanced across available sources:

```python
available_weight = sum(weights[s] for s in available_sources)
rebalanced = {s: weights[s] / available_weight for s in available_sources}
```

### 3.2 Probability Conversions

**Vegas Spread to Probability:**

```python
def spread_to_probability(spread: float) -> float:
    """Convert point spread to win probability.

    Calibrated so:
      -3 spread = ~58% win probability
      -7 spread = ~70% win probability
      -14 spread = ~85% win probability
    """
    k = 0.148  # Steepness factor
    return 1.0 / (1.0 + math.exp(k * spread))
```

**Rating Gap to Probability:**

```python
def rating_gap_to_probability(gap: float, home_advantage: float = 0.03) -> float:
    """Convert rating difference to win probability."""
    k = 10.0  # Steepness factor
    adjusted_gap = gap + home_advantage
    return 1.0 / (1.0 + math.exp(-k * adjusted_gap))
```

**Elo to Probability:**

```python
def elo_to_probability(home_elo: float, away_elo: float, home_adv: float = 55.0) -> float:
    """Standard Elo probability formula."""
    diff = home_elo + home_adv - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / 400))
```

### 3.3 Confidence Levels

Predictions are categorized by source agreement:

| Level | Criteria | Interpretation |
|-------|----------|----------------|
| HIGH | All sources agree, probability spread < 10% | Strong consensus |
| MODERATE | All sources agree, spread 10-20% | Reasonable confidence |
| LOW | All sources agree, spread > 20% | Weak confidence |
| SPLIT | Sources disagree on winner | Toss-up or contrarian picks |
| UNKNOWN | Insufficient data | Cannot predict |

---

## 4. Validation System

### 4.1 External Validators

Rankings are compared against three external systems:

| Validator | Description | Data Source |
|-----------|-------------|-------------|
| **SP+** | Efficiency-based, offense/defense splits | CollegeFootballData.com |
| **SRS** | Simple Rating System (margin + SOS) | CollegeFootballData.com |
| **Elo** | Historical rating with game-by-game updates | CollegeFootballData.com |

### 4.2 Correlation Metrics

**Spearman Rank Correlation:** Measures ranking agreement regardless of rating scale.

```python
from scipy.stats import spearmanr

def calculate_correlation(our_ranks: list[int], their_ranks: list[int]) -> float:
    correlation, _ = spearmanr(our_ranks, their_ranks)
    return correlation
```

Interpretation:
- `> 0.85`: Strong agreement
- `0.70-0.85`: Moderate agreement
- `< 0.70`: Investigate discrepancies

### 4.3 Flagging Threshold

Teams are flagged when rank gap exceeds threshold (default: 20 positions):

```
Our #5 vs SP+ #28 = Gap of 23 → FLAGGED
Our #12 vs SP+ #18 = Gap of 6 → OK
```

**Common causes of flagging:**
- Weak schedule inflating rating
- High margin bonus from blowouts vs weak teams
- Efficiency (underlying play quality) masked by score

### 4.4 Pattern Detection

The validation report identifies systematic patterns:

```python
patterns = []
if overrated_count > len(flagged) * 0.6:
    patterns.append(f"{overrated_count}/{len(flagged)} flagged teams appear overrated")
if weak_sos_count >= len(flagged) * 0.6:
    patterns.append(f"{weak_sos_count}/{len(flagged)} flagged teams have weak SOS (<0.45)")
```

### 4.5 Vegas Upset Analysis

Analyzes games where Vegas favorites lost outright:

**Anomaly factors detected:**
- Favorite had weak SOS
- Small rating gap between teams
- Road favorite (travel disadvantage)
- Conference mismatch (P5 vs G5)
- Late-season letdown spots

**Output metrics:**
- Total upsets (Vegas wrong)
- Upsets we predicted correctly
- Our edge: % of upsets we called

---

## 5. Configuration

### 5.1 Parameter Categories

44 configurable parameters organized into 12 categories:

| Category | Count | Key Parameters |
|----------|-------|----------------|
| **Core** | 5 | `max_iterations`, `convergence_threshold`, `win_base`, `margin_weight`, `margin_cap` |
| **Venue** | 4 | `venue_road_win`, `venue_neutral_win`, `venue_home_loss`, `venue_neutral_loss` |
| **Initialization** | 2 | `initial_rating`, `fcs_fixed_rating` |
| **Opponent Influence** | 3 | `opponent_weight`, `loss_opponent_factor`, `second_order_weight` |
| **Conference** | 5 | `enable_conference_adj`, `conference_method`, `p5_multiplier`, `g5_multiplier`, `fcs_multiplier` |
| **Schedule Strength** | 5 | `sos_adjustment_weight`, `sos_method`, `min_sos_top_10`, `min_sos_top_25`, `min_p5_games_top_10` |
| **Quality Tiers** | 6 | `enable_quality_tiers`, `elite_threshold`, `good_threshold`, `bad_threshold`, `elite_win_bonus`, `bad_loss_penalty` |
| **Recency** | 3 | `enable_recency`, `recency_half_life`, `recency_min_weight` |
| **Prior** | 3 | `enable_prior`, `prior_weight`, `prior_decay_weeks` |
| **Margin Curve** | 1 | `margin_curve` (log/linear/sqrt) |
| **Tiebreakers** | 2 | `tie_threshold`, `tiebreaker_order` |
| **Filters** | 5 | `include_postseason`, `postseason_weight`, `include_fcs_games`, `exclude_fcs_from_rankings`, `min_games_to_rank` |

See `CONFIG_REFERENCE.md` for complete documentation of each parameter.

### 5.2 Profiles

5 pre-defined profiles for common use cases:

| Profile | Philosophy | Key Settings |
|---------|------------|--------------|
| `pure_results` | Raw algorithm, no adjustments | All defaults |
| `balanced` | Moderate real-world factors | Conference adj (P5: 1.02, G5: 0.95), quality tiers |
| `predictive` | Optimized for predicting | Strong conference (P5: 1.05, G5: 0.85), recency (half-life: 6) |
| `tuned_predictive` | Calibrated from 2025 data | Reduced margin_weight (0.15), opponent_weight (0.9) |
| `conservative` | Early-season stability | Historical priors (weight: 0.20), long recency (half-life: 10) |

### 5.3 Configuration Loading

```bash
# Use a profile
ncaa-rank rank 2025 --profile predictive

# Use custom JSON config
ncaa-rank rank 2025 --config my_config.json

# Export profile for customization
ncaa-rank config export my_config.json --profile predictive
```

---

## 6. CLI Commands

### 6.1 Ranking Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `rank` | Generate team rankings | `--week`, `--top`, `--profile`, `--config`, `--exclude-fcs` |
| `explain` | Rating breakdown for a team | `--week` |
| `compare` | Compare to AP/CFP poll | `--poll` (ap/cfp/coaches), `--week` |
| `export` | Export to CSV/JSON | `--format`, `--output`, `--top` |

**Examples:**

```bash
# Top 25 for 2025 season
ncaa-rank rank 2025 --top 25

# Week 10 rankings using predictive profile
ncaa-rank rank 2025 --week 10 --profile predictive

# Why is Ohio State ranked #3?
ncaa-rank explain 2025 ohio-state

# Compare to CFP rankings
ncaa-rank compare 2025 --poll cfp
```

### 6.2 Prediction Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `predict` | Week predictions | `--consensus`, `--high-confidence`, `--show-splits` |
| `predict-game` | Single game analysis | `--season`, `--week`, `--neutral` |

**Examples:**

```bash
# All predictions for Week 12
ncaa-rank predict 2025 12

# Consensus view with all sources
ncaa-rank predict 2025 12 --consensus

# Only high-confidence picks
ncaa-rank predict 2025 12 --consensus --high-confidence

# Single matchup analysis
ncaa-rank predict-game "Ohio State" "Michigan" --season 2025 --week 13
```

### 6.3 Diagnostic Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `diagnose` | Accuracy analysis | `--upsets`, `--attributions` |
| `decompose` | Game contribution breakdown | `--profile` |
| `validate` | Compare to SP+/SRS/Elo | `--threshold`, `--team`, `--source` |
| `vegas-analysis` | Upset pattern analysis | `--min-spread`, `--we-got-right`, `--export` |

**Examples:**

```bash
# Full diagnostic report
ncaa-rank diagnose 2025

# How did each game affect Texas's rating?
ncaa-rank decompose 2025 texas

# Validate against SP+
ncaa-rank validate 2025 --source sp

# Analyze upsets where we beat Vegas
ncaa-rank vegas-analysis 2025 --we-got-right
```

### 6.4 Config Commands

| Command | Description |
|---------|-------------|
| `config list` | List available profiles |
| `config show [profile]` | Display profile settings |
| `config export <file>` | Export config to JSON |
| `config create <file>` | Create documented config template |

---

## 7. Data Models

### 7.1 Core Entities

**Team:**
```python
class Team(BaseModel):
    team_id: str           # e.g., "ohio-state"
    name: str              # e.g., "Ohio State Buckeyes"
    conference: str | None # e.g., "Big Ten"
    division: Literal["fbs", "fcs"]
```

**Game:**
```python
class Game(BaseModel):
    game_id: int
    season: int
    week: int
    game_date: date
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int
    neutral_site: bool = False
    postseason: bool = False
```

**GameResult (derived view):**
```python
class GameResult(BaseModel):
    game_id: int
    team_id: str
    opponent_id: str
    team_score: int
    opponent_score: int
    location: Literal["home", "away", "neutral"]
    is_win: bool
    margin: int  # Positive=win, negative=loss
```

**TeamRating:**
```python
class TeamRating(BaseModel):
    team_id: str
    season: int
    week: int
    rating: float          # Normalized [0, 1]
    rank: int
    games_played: int
    wins: int
    losses: int
    strength_of_schedule: float | None
    strength_of_victory: float | None
    average_game_grade: float | None
```

### 7.2 Prediction Entities

**SourcePrediction:**
```python
class SourcePrediction:
    source: str                    # "vegas", "our_algorithm", etc.
    predicted_winner: str | None
    win_probability: float | None
    spread: float | None           # Vegas only
    weight: float                  # Weight in consensus
    available: bool
```

**GamePrediction:**
```python
class GamePrediction:
    game_id: int | None
    season: int
    week: int | None
    home_team_id: str
    away_team_id: str
    source_predictions: list[SourcePrediction]
    consensus_winner: str | None
    consensus_prob: float | None
    confidence: Literal["HIGH", "MODERATE", "LOW", "SPLIT", "UNKNOWN"]
    sources_agreeing: int
    sources_available: int
```

**ConfidenceBreakdown:**
```python
class ConfidenceBreakdown:
    level: Literal["HIGH", "MODERATE", "LOW", "SPLIT", "UNKNOWN"]
    sources_agreeing: int
    sources_available: int
    probability_spread: float
    reason: str
```

### 7.3 Validation Entities

**ValidatorRating:**
```python
class ValidatorRating:
    team_id: str
    source: Literal["sp", "srs", "elo"]
    rating: float
    rank: int | None
    offense: float | None  # SP+ only
    defense: float | None  # SP+ only
```

**ValidationReport:**
```python
class ValidationReport:
    season: int
    week: int | None
    correlations: dict[str, float]          # source -> Spearman correlation
    flagged_teams: list[TeamValidation]
    flagged_count_by_source: dict[str, int]
    patterns: list[str]
    total_teams_compared: int
```

### 7.4 Vegas Upset Entities

**VegasUpset:**
```python
class VegasUpset:
    game_id: int
    season: int
    week: int
    vegas_favorite: str
    vegas_underdog: str
    spread: float
    implied_prob: float
    actual_winner: str
    actual_loser: str
    final_margin: int
```

**PatternReport:**
```python
class PatternReport:
    season: int
    week: int | None
    total_games: int
    games_with_lines: int
    vegas_correct: int
    vegas_wrong: int
    we_predicted_upset: int
    we_also_wrong: int
    by_spread_bucket: dict[str, UpsetStats]
    by_conference_matchup: dict[str, UpsetStats]
    anomaly_factors: list[AnomalyFactor]
```

---

## 8. Project Structure

```
ncaa-fbs-ranking-algorithm/
├── PROJECT_SPEC.md          # This file
├── README.md                # User-facing quick start
├── CLAUDE.md                # AI assistant context
├── CONFIG_REFERENCE.md      # All 44 parameters documented
├── pyproject.toml           # Dependencies and metadata
├── .env.example             # Environment template
│
├── src/
│   ├── algorithm/
│   │   ├── convergence.py   # Core iterative algorithm
│   │   ├── game_grade.py    # Game grade calculation
│   │   ├── diagnostics.py   # Prediction analysis and attribution
│   │   └── tiebreaker.py    # H2H, SOV, common opponents
│   │
│   ├── cli/
│   │   └── main.py          # All 11 commands (Typer)
│   │
│   ├── data/
│   │   ├── models.py        # Pydantic entities (44 config params)
│   │   ├── profiles.py      # 5 pre-defined profiles
│   │   ├── client.py        # CollegeFootballData API client
│   │   └── storage.py       # Caching layer
│   │
│   ├── ranking/
│   │   ├── engine.py        # Orchestration
│   │   └── comparison.py    # AP/CFP poll comparison
│   │
│   ├── validation/
│   │   ├── consensus.py     # 5-source prediction blending
│   │   ├── validators.py    # SP+/SRS/Elo integration
│   │   ├── upset_analyzer.py # Vegas upset patterns
│   │   └── models.py        # Validation entities
│   │
│   └── web/
│       ├── app.py           # FastAPI dashboard
│       ├── templates/       # Jinja2 templates
│       └── static/          # CSS/JS assets
│
├── tests/
│   ├── unit/
│   │   ├── test_convergence.py
│   │   ├── test_game_grade.py
│   │   ├── test_consensus.py
│   │   └── test_config_expanded.py
│   ├── integration/
│   │   └── test_ranking_engine.py
│   └── fixtures/
│       └── sample_games.json
│
├── data/
│   └── cache/               # Cached API responses
│
├── archive/
│   └── PROJECT_SPEC_v0.1.md # Original spec (historical reference)
│
└── docs/
    └── diagrams/            # Architecture diagrams
```

---

## 9. Diagnostic Feedback Loop

### 9.1 Tuning Workflow

```
┌────────────────┐
│   Rankings     │ ← Initial algorithm output
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Validators    │ ← Compare to SP+/SRS/Elo
└───────┬────────┘
        │ Flag discrepancies
        ▼
┌────────────────┐
│  Diagnostics   │ ← Accuracy, Brier, calibration
└───────┬────────┘
        │ Identify error patterns
        ▼
┌────────────────┐
│ Parameter Tune │ ← Adjust levers
└───────┬────────┘
        │
        └──────────────────────────────► Repeat
```

### 9.2 Key Diagnostic Metrics

**Prediction Accuracy:**
```
Accuracy = Correct Predictions / Total Games
```

Target: > 80% (2025 achieved: 84.7%)

**Brier Score:**
```
Brier = (1/N) * SUM((probability - outcome)^2)
```

Scale:
- 0.00 = Perfect
- 0.25 = Random guessing (50% always)
- 0.106 = Current (2025 tuned)

Lower is better. Measures probability calibration, not just correct/wrong.

**Calibration Error:**
```
For each probability bucket (e.g., 70-80%):
  expected = midpoint (75%)
  actual = % of games in bucket that we got right
  error = |expected - actual|

Calibration Error = mean(errors across buckets)
```

Target: < 0.08

### 9.3 Parameter Attribution

Diagnostics identify which parameters contributed to errors:

```
Parameter Attribution (which levers caused errors)
┌──────────────────────┬────────┬──────────────────────────┬─────────────────────────┐
│ Parameter            │ Errors │ Pattern                  │ Suggestion              │
├──────────────────────┼────────┼──────────────────────────┼─────────────────────────┤
│ venue_road_win       │ 12     │ Road favorites losing    │ Reduce from 0.10→0.08   │
│ margin_weight        │ 8      │ Blowout teams regressing │ Reduce from 0.20→0.15   │
│ opponent_weight      │ 6      │ Rating spread too wide   │ Reduce from 1.0→0.9     │
└──────────────────────┴────────┴──────────────────────────┴─────────────────────────┘
```

### 9.4 Tuned Profile Creation

The `tuned_predictive` profile was created by:

1. Running `diagnose` on 2025 season
2. Identifying top error-contributing parameters
3. Adjusting in small increments
4. Re-running validation until Brier score improved

Changes from `predictive`:
- `margin_weight`: 0.20 → 0.15
- `venue_road_win`: 0.10 → 0.08
- `opponent_weight`: 1.0 → 0.9

---

## 10. API Integration

### 10.1 Data Source

**CollegeFootballData.com**
- Base URL: `https://api.collegefootballdata.com`
- Authentication: Bearer token (`CFBD_API_KEY`)
- Rate limit: 1000 requests/hour

### 10.2 Endpoints Used

| Endpoint | Purpose | Response Size |
|----------|---------|---------------|
| `/games` | Game results | ~900 games/season |
| `/teams` | Team metadata | ~130 FBS teams |
| `/teams/fbs` | FBS team list | ~130 teams |
| `/lines` | Vegas betting lines | ~800 lines/season |
| `/metrics/wp/pregame` | Pre-game win probabilities | ~800 games/season |
| `/ratings/sp` | SP+ ratings | ~130 teams |
| `/ratings/srs` | SRS ratings | ~130 teams |
| `/ratings/elo` | Elo ratings | ~130 teams |
| `/rankings` | AP/CFP poll rankings | ~25 teams/week |

### 10.3 Caching Strategy

```python
CACHE_DIR = Path("data/cache")

def get_cached(key: str) -> dict | None:
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists() and not stale(cache_file):
        return json.loads(cache_file.read_text())
    return None

def stale(file: Path, max_age_hours: int = 24) -> bool:
    age = datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)
    return age.total_seconds() > max_age_hours * 3600
```

API responses are cached for 24 hours to avoid redundant calls and rate limit issues.

### 10.4 Rate Limiting

```python
class CFBDataClient:
    def __init__(self, api_key: str):
        self._semaphore = asyncio.Semaphore(10)  # Max concurrent
        self._request_times: list[float] = []

    async def _rate_limit(self):
        # Enforce 1000/hour limit
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 3600]
        if len(self._request_times) >= 1000:
            wait = 3600 - (now - self._request_times[0])
            await asyncio.sleep(wait)
        self._request_times.append(now)
```

---

## 11. Performance Metrics

### 11.1 2025 Season Results (tuned_predictive profile)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Prediction Accuracy** | 84.7% | Correct winner predictions |
| **Brier Score** | 0.106 | Probability calibration (0.25 = random) |
| **Calibration Error** | 0.062 | Bucket accuracy deviation |
| **CFP Correlation** | 79.6% | Spearman rank correlation |
| **SP+ Correlation** | 0.89 | Agreement with efficiency metric |
| **Convergence** | ~35 iterations | Typical stabilization |

### 11.2 Performance by Confidence Level

| Confidence | Games | Correct | Accuracy |
|------------|-------|---------|----------|
| HIGH | 412 | 378 | 91.7% |
| MODERATE | 198 | 163 | 82.3% |
| LOW | 89 | 64 | 71.9% |
| SPLIT | 52 | 28 | 53.8% |

### 11.3 Vegas Comparison

| Metric | Vegas | Our Algorithm | Our Edge |
|--------|-------|---------------|----------|
| Overall accuracy | 87.2% | 84.7% | -2.5% |
| On games Vegas missed | 0% | 32.1% | **+32.1%** |

When Vegas is wrong (upsets), we correctly predicted 32.1% of those outcomes.

### 11.4 Runtime Performance

| Operation | Time |
|-----------|------|
| Full season ranking | < 5 seconds |
| Single week prediction | < 2 seconds |
| Validation report | < 3 seconds |
| Vegas analysis | < 4 seconds |

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

**FCS Data Gaps:**
- FCS-vs-FCS games not fully tracked
- FCS teams use fixed rating (0.20) by default
- Can cause inaccuracies for teams with heavy FCS schedules

**No Player-Level Data:**
- Algorithm uses team-level outcomes only
- Cannot factor in injuries, transfers, suspensions
- Player EPA integration was planned but not implemented

**Early Season Instability:**
- Requires ~4 weeks of data for meaningful ratings
- Preseason rankings not supported (no prior year carryover by default)
- `conservative` profile with priors helps but isn't perfect

**COVID 2020 Season:**
- Shortened season with conference-only games
- Data is included but results may be anomalous

### 12.2 Known Gaps vs Original Spec

The original PROJECT_SPEC v0.1 described features that were never implemented:

| Planned Feature | Status | Notes |
|-----------------|--------|-------|
| Player rankings (EPA-adjusted) | Not implemented | Would require PPA endpoint integration |
| Conference strength aggregation | Not implemented | Low priority |
| Real-time webhook updates | Not implemented | Would require infrastructure |

### 12.3 Future Possibilities

**Short-term improvements:**
- Historical backtest automation (validate across multiple seasons)
- Cross-validation for parameter tuning
- Confidence interval estimation for ratings

**Long-term extensions:**
- Player EPA integration (adjusts team rating by personnel)
- Live game updates via webhooks
- Mobile-friendly web dashboard
- Bowl game prediction markets integration

---

## Appendix A: Algorithm Pseudocode

```python
def rank_season(games: list[Game], config: AlgorithmConfig) -> list[TeamRating]:
    """Complete ranking pipeline."""

    # 1. Build game results lookup
    results_by_team = build_game_results(games)

    # 2. Initialize all teams to 0.500
    ratings = {team: 0.5 for team in all_teams(games)}

    # 3. Iterative convergence
    for iteration in range(config.max_iterations):
        new_ratings = {}

        for team_id in sorted(ratings.keys()):
            if team_id in fcs_teams and config.fcs_fixed_rating:
                new_ratings[team_id] = config.fcs_fixed_rating
                continue

            total = 0.0
            for game in results_by_team[team_id]:
                grade = compute_game_grade(game, config)
                opp_rating = ratings[game.opponent_id] * config.opponent_weight

                if game.is_win:
                    contribution = grade + opp_rating
                else:
                    weakness = 1.0 - opp_rating
                    contribution = grade - weakness * config.loss_opponent_factor

                total += contribution

            new_ratings[team_id] = total / len(results_by_team[team_id])

        # Check convergence
        max_delta = max(abs(new_ratings[t] - ratings[t]) for t in ratings)
        ratings = new_ratings

        if max_delta < config.convergence_threshold:
            break

    # 4. Normalize to [0, 1]
    normalized = normalize_ratings(ratings)

    # 5. Build output with rankings
    return sorted([
        TeamRating(team_id=t, rating=r, rank=i+1, ...)
        for i, (t, r) in enumerate(sorted(normalized.items(), key=lambda x: -x[1]))
    ], key=lambda x: x.rank)
```

---

## Appendix B: Consensus Pseudocode

```python
def predict_game(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    vegas_spread: float | None,
    sp_ratings: dict[str, float],
    elo_ratings: dict[str, float],
) -> GamePrediction:
    """Build consensus prediction from all sources."""

    sources = {}

    # Our algorithm
    if home_team in ratings and away_team in ratings:
        gap = ratings[home_team] - ratings[away_team]
        sources["our_algorithm"] = rating_gap_to_probability(gap)

    # Vegas
    if vegas_spread is not None:
        sources["vegas"] = spread_to_probability(vegas_spread)

    # SP+
    if home_team in sp_ratings and away_team in sp_ratings:
        sp_gap = sp_ratings[home_team] - sp_ratings[away_team]
        sources["sp_implied"] = sp_to_probability(sp_gap)

    # Elo
    if home_team in elo_ratings and away_team in elo_ratings:
        sources["elo_implied"] = elo_to_probability(
            elo_ratings[home_team], elo_ratings[away_team]
        )

    # Calculate weighted consensus
    weights = {
        "vegas": 0.35, "our_algorithm": 0.25,
        "pregame_wp": 0.20, "sp_implied": 0.10, "elo_implied": 0.10
    }

    available_weight = sum(weights[s] for s in sources)
    consensus_prob = sum(
        sources[s] * weights[s] / available_weight for s in sources
    )

    # Assess confidence
    confidence = assess_confidence(sources)

    return GamePrediction(
        consensus_winner=home_team if consensus_prob > 0.5 else away_team,
        consensus_prob=max(consensus_prob, 1 - consensus_prob),
        confidence=confidence,
        ...
    )
```

---

*Last updated: 2026-01-12*
*Version: 2.0*
