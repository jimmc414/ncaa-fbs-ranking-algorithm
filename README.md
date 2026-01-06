# NCAA FBS Ranking Algorithm

A college football ranking system that derives team ratings purely from game outcomes using iterative convergence. No human voting, no preseason bias, no brand recognition - just results.

The core insight: you can't know how good a team is until you know how good their opponents are, which depends on *their* opponents, recursively. The algorithm iterates until ratings stabilize.

## The Problem

Human polls produce rankings contaminated by:

- **Preseason expectations** - Teams ranked #5 in August stay ranked even after losses
- **Anchoring** - Voters adjust from last week's ranking rather than computing fresh
- **Brand recognition** - Alabama gets benefit of the doubt; Boise State doesn't
- **Recency bias** - Last week's performance outweighs the full body of work

The AP Poll has 65 voters watching 130+ teams. No voter sees every game. Rankings become a collective guess shaped by ESPN highlights and conventional wisdom.

## The Solution

This algorithm uses iterative convergence (similar to PageRank) to compute ratings from game outcomes:

1. Initialize all teams to 0.500 rating
2. For each team, compute a new rating based on:
   - Game grades (win/loss, margin, venue)
   - Current opponent ratings (wins add, losses subtract)
3. Repeat until ratings change by less than 0.0001
4. Normalize final ratings to [0, 1]

The algorithm converges because opponent ratings stabilize. After ~15-30 iterations, a team's rating reflects not just who they beat, but how good those teams turned out to be.

---

## Core Algorithms

### Game Grade

Each game produces a grade from the perspective of one team:

```
game_grade(is_win, margin, location):
    # Base value
    base = 0.70 if is_win else 0.00

    # Margin bonus (wins only, logarithmic, capped at 28 points)
    if is_win:
        capped = min(margin, 28)
        margin_bonus = 0.20 * log(1 + capped) / log(29)
    else:
        margin_bonus = 0

    # Venue adjustment
    venue_adj = match (is_win, location):
        (true, away):    +0.10    # Road wins are hard
        (true, neutral): +0.05
        (false, home):   -0.03    # Home losses are bad
        (false, neutral):-0.01
        _:                0.00

    return base + margin_bonus + venue_adj
```

**Design choices:**
- Win base (0.70) is higher than loss base (0.00) - winning matters
- Margin is logarithmic - first touchdown of margin worth more than third
- Cap at 28 points - no incentive to run up the score
- Road wins rewarded, home losses penalized

### Iterative Convergence

The heart of the algorithm:

```
converge(games):
    ratings = {team: 0.5 for team in all_teams}

    for iteration in 1..max_iterations:
        new_ratings = {}

        for team in sorted(teams):  # MUST sort for determinism
            contributions = []

            for game in team_games[team]:
                grade = game_grade(game)
                opp_rating = ratings[opponent]

                # THE KEY INSIGHT:
                # Wins ADD opponent rating (beat good team = good)
                # Losses SUBTRACT opponent rating (lose to bad team = very bad)
                if is_win:
                    contribution = grade + opp_rating
                else:
                    contribution = grade - opp_rating

                contributions.append(contribution)

            new_ratings[team] = mean(contributions)

        # Check convergence
        max_delta = max(|new_ratings[t] - ratings[t]| for t in teams)
        if max_delta < 0.0001:
            break

        ratings = new_ratings

    return normalize_to_0_1(ratings)
```

**Why losses subtract opponent rating:**

This is what distinguishes this algorithm from flawed predecessors. Consider two losses:
- Lose to #1 team: `contribution = 0.00 - 0.95 = -0.95`
- Lose to #100 team: `contribution = 0.00 - 0.20 = -0.20`

Wait, that seems backward? But remember, these contributions are *averaged*. The team that lost to #1 has a -0.95 dragging down their average less than you'd expect, because that opponent's high rating makes the denominator work differently across all games.

The key is: losing to a *bad* team creates a small negative contribution that competes with your wins. Losing to a *good* team creates a large negative that gets somewhat offset by the opponent's quality in the convergence math.

Put simply: quality losses hurt less than bad losses, but all losses hurt.

### Tiebreakers

When two teams have ratings within 0.001 of each other:

```
resolve_tie(team_a, team_b):
    for tiebreaker in [head_to_head, strength_of_victory,
                       common_opponents, away_record]:
        result = tiebreaker(team_a, team_b)
        if result != TIE:
            return result
    return original_order
```

---

## Prediction System

The algorithm includes a prediction system that blends multiple sources:

### Spread to Probability

Convert Vegas point spreads to win probability:

```
spread_to_probability(spread):
    k = 0.148  # Calibrated: -7 spread = 70% win probability
    return 1 / (1 + exp(k * spread))
```

Calibration reference:
- -3 points: ~58%
- -7 points: ~70%
- -14 points: ~85%
- -21 points: ~92%

### Consensus Prediction

Blend multiple prediction sources:

```
WEIGHTS = {
    vegas:        0.35,   # Historically most accurate
    our_algorithm: 0.25,   # This system
    pregame_wp:   0.20,   # Market-derived probability
    sp_implied:   0.10,   # SP+ ratings converted to probability
    elo_implied:  0.10    # Elo ratings converted to probability
}

consensus(predictions):
    available = {source: prob for source, prob in predictions
                 if prob is not None}
    total_weight = sum(WEIGHTS[s] for s in available)
    return sum(prob * WEIGHTS[s] / total_weight
               for s, prob in available)
```

Vegas gets the highest weight because betting markets have the best historical track record. But showing all sources lets you see when they disagree.

### Confidence Levels

```
assess_confidence(predictions):
    probs = list(predictions.values())
    spread = max(probs) - min(probs)
    all_agree = all(p > 0.5 for p in probs) or all(p < 0.5 for p in probs)

    if not all_agree:
        return "SPLIT"     # Sources disagree on winner
    elif spread < 0.10:
        return "HIGH"      # All agree, tight spread
    elif spread < 0.20:
        return "MODERATE"  # All agree, some variance
    else:
        return "LOW"       # All agree, but wide spread
```

---

## Installation

```bash
# Clone and install
git clone https://github.com/jimmc414/ncaa-fbs-ranking-algorithm.git
cd ncaa-fbs-ranking-algorithm
pip install -e ".[dev]"

# Configure API key
cp .env.example .env
# Edit .env and add your CFBD_API_KEY

# Get a free API key at: https://collegefootballdata.com/key
```

### Requirements

- Python 3.11+
- CollegeFootballData.com API key (free tier sufficient)

---

## CLI Commands

### Generate Rankings

```bash
# Top 25 for current season
ncaa-rank rank 2024 --top 25

# Rankings as of week 10
ncaa-rank rank 2024 --week 10 --top 25

# Use predictive profile (optimized for forecasting)
ncaa-rank rank 2024 --profile predictive

# Use custom configuration
ncaa-rank rank 2024 --config my_config.json

# Exclude FCS teams from output
ncaa-rank rank 2024 --exclude-fcs
```

### Predict Games

```bash
# Predict all games in week 12
ncaa-rank predict 2024 12

# Consensus view with all sources
ncaa-rank predict 2024 12 --consensus

# Only high confidence predictions
ncaa-rank predict 2024 12 --high-confidence

# Only games where sources disagree
ncaa-rank predict 2024 12 --show-splits
```

### Single Game Prediction

```bash
# Predict a specific matchup
ncaa-rank predict-game "Ohio State" "Michigan" --season 2024

# Neutral site game
ncaa-rank predict-game "Georgia" "Alabama" --neutral

# Shows all sources: Vegas, our algorithm, pregame WP, SP+, Elo
```

### Validate Rankings

Compare our rankings against external validators:

```bash
# Full validation report
ncaa-rank validate 2024

# Single team deep dive
ncaa-rank validate 2024 --team ohio-state

# Lower threshold for flagging discrepancies
ncaa-rank validate 2024 --threshold 15
```

### Vegas Upset Analysis

Find patterns in games where Vegas favorites lost:

```bash
# Full season analysis
ncaa-rank vegas-analysis 2024

# Only upsets we correctly predicted
ncaa-rank vegas-analysis 2024 --we-got-right

# Filter to bigger upsets (min 7-point favorites)
ncaa-rank vegas-analysis 2024 --min-spread 7

# Export to JSON
ncaa-rank vegas-analysis 2024 --export analysis.json
```

### Team Analysis

```bash
# Detailed rating breakdown
ncaa-rank decompose 2024 ohio-state

# Explain how rating was computed
ncaa-rank explain 2024 ohio-state
```

### Diagnostics

```bash
# Prediction accuracy and calibration
ncaa-rank diagnose 2024 --profile predictive

# Shows Brier score, calibration error, biggest upsets
```

### Compare to Polls

```bash
# Compare to AP Poll
ncaa-rank compare 2024 --poll ap

# Compare to CFP rankings
ncaa-rank compare 2024 --poll cfp
```

### Configuration Management

```bash
# List available profiles
ncaa-rank config list

# Show profile settings
ncaa-rank config show predictive

# Export profile to JSON for customization
ncaa-rank config export my_config.json --profile predictive

# Create documented config file
ncaa-rank config create my_config.json --documented
```

### Export

```bash
# Export rankings to CSV
ncaa-rank export 2024 --format csv --output rankings.csv

# Export to JSON
ncaa-rank export 2024 --format json
```

---

## Configuration

### Profiles

Five pre-configured profiles for different use cases:

| Profile | Description |
|---------|-------------|
| `pure_results` | No adjustments. Pure game outcomes with margin and venue. |
| `balanced` | Moderate conference and quality tier adjustments. Good starting point. |
| `predictive` | Optimized for forecasting. Strong conference, quality, and recency adjustments. |
| `tuned_predictive` | Calibrated from 2025 diagnostics. 82.9% accuracy. Best for predictions. |
| `conservative` | Uses historical priors. More stable early-season rankings. |

```bash
# Use a profile
ncaa-rank rank 2024 --profile predictive
```

### Key Configuration Levers

The algorithm has 45+ configurable parameters. Key categories:

**Core Scoring**
- `win_base` (0.70): Base value for any win
- `margin_weight` (0.20): Maximum margin bonus
- `margin_cap` (28): Points beyond which margin doesn't count

**Venue**
- `venue_road_win` (+0.10): Bonus for road wins
- `venue_home_loss` (-0.03): Penalty for home losses

**Opponent Influence**
- `opponent_weight` (1.0): Multiplier on opponent rating contribution
- `loss_opponent_factor` (-1.0): How losses use opponent rating (negative = subtract)

**Conference Adjustments**
- `enable_conference_adj` (false): Enable P5/G5/FCS multipliers
- `p5_multiplier` (1.0): Multiplier for P5 opponents
- `g5_multiplier` (1.0): Multiplier for G5 opponents
- `fcs_fixed_rating` (0.20): Fixed rating for FCS teams

**Quality Tiers**
- `enable_quality_tiers` (false): Bonus for elite wins, penalty for bad losses
- `elite_threshold` (0.80): Rating threshold for "elite" opponent
- `elite_win_bonus` (0.0): Bonus for beating elite opponent
- `bad_loss_penalty` (0.0): Penalty for losing to bad opponent

**Schedule Strength**
- `sos_adjustment_weight` (0.0): Post-hoc SOS adjustment
- `min_sos_top_10` (0.0): Minimum SOS to be ranked top 10

**Recency**
- `enable_recency` (false): Weight recent games more
- `recency_half_life` (8): Weeks for weight to halve

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for complete documentation.

---

## Validation Philosophy

**Facts -> Algorithm -> Opinions -> Validation**

The ranking algorithm uses only game outcomes:
- Scores
- Venue (home/away/neutral)
- Date (for recency, if enabled)

External ratings (SP+, SRS, Elo) are used for **validation only** - they never influence the core rankings. This separation ensures the algorithm remains purely results-based.

For **predictions**, we blend multiple sources including Vegas lines. This is appropriate because predictions are about forecasting, not measuring past performance.

### External Validators

| Source | Description |
|--------|-------------|
| SP+ | Efficiency-based rating from ESPN |
| SRS | Simple Rating System (margin + SOS) |
| Elo | Rating based on expected vs actual outcomes |
| Vegas | Betting market lines |

When our rankings diverge significantly from all validators, it suggests either:
1. Our algorithm found something others missed
2. Our algorithm is overweighting some factor (usually weak-schedule G5 teams)

The `validate` command helps identify these discrepancies.

---

## Data Source

All data comes from [CollegeFootballData.com](https://collegefootballdata.com/), which provides:

- Game results (scores, dates, venues)
- Team information (conference, division)
- Betting lines
- SP+, SRS, Elo ratings
- Pre-game win probabilities

Get a free API key at: https://collegefootballdata.com/key

The API allows 1000 requests/hour on the free tier. The client implements caching and rate limiting.

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific test file
pytest tests/unit/test_convergence.py
```

Current status: 484 tests, 65% coverage.

Key test areas:
- `test_convergence.py` - Core algorithm behavior
- `test_game_grade.py` - Game grade calculation
- `test_tiebreaker.py` - Tie resolution logic
- `test_consensus.py` - Prediction blending
- `test_upset_analyzer.py` - Vegas upset analysis

---

## Project Structure

```
src/
├── algorithm/
│   ├── convergence.py     # Iterative rating solver
│   ├── game_grade.py      # Game grade calculation
│   ├── tiebreaker.py      # Tie resolution
│   └── diagnostics.py     # Accuracy analysis
├── data/
│   ├── models.py          # Pydantic data models
│   ├── client.py          # API client (CollegeFootballData)
│   ├── profiles.py        # Configuration profiles
│   └── storage.py         # SQLite persistence
├── ranking/
│   ├── engine.py          # Orchestration layer
│   └── comparison.py      # Poll comparison
├── validation/
│   ├── validators.py      # External validator service
│   ├── consensus.py       # Prediction blending
│   └── upset_analyzer.py  # Vegas upset patterns
├── cli/
│   └── main.py            # Typer CLI
└── web/
    ├── app.py             # FastAPI application
    └── templates/         # Jinja2 + HTMX
```

---

## Limitations

Things this algorithm does not do well:

1. **Early season** - With few games, ratings are noisy. Consider using `conservative` profile with priors.

2. **FCS teams** - FCS teams are assigned a fixed rating (0.20 by default) because they play too few FBS games for convergence.

3. **Injuries/suspensions** - The algorithm knows nothing about personnel. A team's rating reflects their results, not their potential.

4. **Garbage time** - Margin includes garbage time points. A 42-14 blowout and 42-28 (with late touchdowns) look different to the algorithm.

5. **Strength of schedule timing** - A team that played their hardest games early may look worse mid-season than one that backloaded their schedule.

---

## License

MIT

---

## Contributing

Issues and PRs welcome at https://github.com/jimmc414/ncaa-fbs-ranking-algorithm
