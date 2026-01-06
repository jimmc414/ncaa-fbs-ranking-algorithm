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

## What Goes Into the Rating

The algorithm considers these factors when computing a team's rating:

### Factors Used

**Win/Loss Outcome**
The most fundamental input. A win contributes positively to a team's rating; a loss contributes negatively. This seems obvious, but the key is *how* wins and losses are valued relative to opponent quality (see Opponent Rating below).

**Margin of Victory**
Winning by more indicates dominance, but with diminishing returns. The margin bonus uses a logarithmic curve: beating a team by 7 is worth more than beating them by 1, but beating them by 28 isn't worth much more than beating them by 21. The cap at 28 points means running up the score beyond four touchdowns provides no benefit.

Why logarithmic? A team that consistently wins by 14 is probably better than one that wins by 3, but a team that wins 56-0 isn't necessarily twice as good as one that wins 28-0. The first might have played their starters into the fourth quarter; the second might have rested them.

**Game Location (Venue)**
Where the game was played matters. Home teams win roughly 57% of games historically. The algorithm adjusts:
- Road wins get a bonus (+0.10) because winning away is harder
- Neutral site wins get a smaller bonus (+0.05)
- Home losses get a penalty (-0.03) because losing at home is unexpected
- Neutral site losses get a smaller penalty (-0.01)

This ensures a team that wins all their road games is rated higher than an equivalent team that won all at home.

**Opponent Rating**
The crucial factor. Beating a 0.80-rated team is worth more than beating a 0.40-rated team. But since opponent ratings depend on their opponents' ratings, this creates a circular dependency. The iterative convergence process resolves this: ratings are computed, then recomputed using updated opponent ratings, until they stabilize.

The key insight is how losses use opponent rating: they *subtract* it rather than add it. This means:
- Losing to a good team: large negative contribution (you lost, minus their high rating)
- Losing to a bad team: small negative contribution (you lost, minus their low rating)

In the averaging, the loss to a good team hurts your average less because it's a larger absolute number being divided across all games. This is the mechanism that makes "quality losses" less damaging than "bad losses" without requiring any explicit quality-loss logic.

### Factors NOT Used

The algorithm deliberately excludes:

**Preseason Rankings**: No prior assumptions about team quality. Every team starts at 0.500.

**Conference Reputation**: In `pure_results` mode, an SEC team isn't valued differently than a MAC team. (Conference adjustments are available in other profiles but are post-hoc, not built into the core algorithm.)

**Recruiting Rankings**: Five-star players don't directly affect ratings. They only matter if they help win games.

**Returning Production**: Doesn't matter how many starters returned. Only game results count.

**Injuries/Suspensions**: The algorithm has no knowledge of roster changes. A team's rating reflects what happened, not what might have happened.

**Point Spreads or Expectations**: Vegas lines are used for validation and prediction blending, but never influence the core ranking.

**Media Coverage or Brand Value**: Notre Dame and Alabama are treated identically to Western Michigan. The only thing that matters is who you beat and who beat you.

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

### rank

Generate rankings for a season using the iterative convergence algorithm.

```bash
ncaa-rank rank <season> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--week` | `-w` | Calculate rankings as of this week (default: full season) |
| `--top` | `-t` | Number of teams to display (default: 25) |
| `--profile` | `-p` | Configuration profile: `pure_results`, `balanced`, `predictive`, `tuned_predictive`, `conservative` |
| `--config` | `-c` | Path to custom JSON config file |
| `--exclude-fcs` | | Hide FCS teams from output (they're still used in calculations) |

```bash
# Examples
ncaa-rank rank 2024 --top 25
ncaa-rank rank 2024 --week 10 --profile predictive
ncaa-rank rank 2024 --config my_config.json --exclude-fcs
```

### predict

Predict outcomes for all games in a specific week. Uses our algorithm's ratings to calculate win probabilities.

```bash
ncaa-rank predict <season> <week> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--profile` | `-p` | Configuration profile to use |
| `--config` | `-c` | Path to custom JSON config file |
| `--min-confidence` | | Only show predictions above this probability (0.0-1.0) |
| `--consensus` | | Show consensus view blending Vegas, SP+, Elo, pregame WP |
| `--high-confidence` | | Only show HIGH confidence predictions (all sources agree, <10% spread) |
| `--show-splits` | | Only show games where sources disagree on the winner |

```bash
# Examples
ncaa-rank predict 2024 12
ncaa-rank predict 2024 12 --consensus --high-confidence
ncaa-rank predict 2024 12 --show-splits
ncaa-rank predict 2024 12 --min-confidence 0.65
```

### predict-game

Predict a single game showing all available sources side-by-side. Displays our algorithm, Vegas spread, pregame win probability, SP+ implied, and Elo implied probabilities.

```bash
ncaa-rank predict-game <team1> <team2> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--season` | `-s` | Season year (default: current) |
| `--week` | `-w` | Week number (for fetching current lines) |
| `--neutral` | `-n` | Treat as neutral site game (no home advantage) |
| `--profile` | `-p` | Configuration profile |
| `--config` | `-c` | Path to custom JSON config file |

Team names can be quoted full names ("Ohio State") or hyphenated IDs (ohio-state).

```bash
# Examples
ncaa-rank predict-game "Ohio State" "Michigan" --season 2024
ncaa-rank predict-game georgia alabama --neutral
ncaa-rank predict-game "Texas" "Oklahoma" -s 2024 -w 12
```

### validate

Compare our rankings against external validators (SP+, SRS, Elo). Calculates Spearman correlation and flags teams where rankings diverge significantly.

```bash
ncaa-rank validate <season> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--week` | `-w` | Validate rankings as of this week |
| `--profile` | `-p` | Configuration profile |
| `--config` | `-c` | Path to custom JSON config file |
| `--threshold` | `-t` | Rank gap to flag as discrepancy (default: 20) |
| `--team` | | Show detailed validation for a specific team |
| `--source` | `-s` | Compare against specific source only: `sp`, `srs`, or `elo` |

```bash
# Examples
ncaa-rank validate 2024
ncaa-rank validate 2024 --threshold 15 --source sp
ncaa-rank validate 2024 --team ohio-state
ncaa-rank validate 2024 --week 10 --profile predictive
```

### vegas-analysis

Analyze games where Vegas favorites lost outright. Identifies patterns and anomalies, tracks which upsets our algorithm correctly predicted.

```bash
ncaa-rank vegas-analysis <season> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--week` | `-w` | Analyze specific week only |
| `--min-spread` | `-m` | Minimum point spread to consider (default: 3.0, filters toss-ups) |
| `--we-got-right` | | Only show upsets our algorithm correctly predicted |
| `--profile` | `-p` | Configuration profile |
| `--config` | `-c` | Path to custom JSON config file |
| `--export` | `-e` | Export results to JSON file |

```bash
# Examples
ncaa-rank vegas-analysis 2024
ncaa-rank vegas-analysis 2024 --min-spread 7 --we-got-right
ncaa-rank vegas-analysis 2024 --week 12 --export upsets.json
```

### decompose

Show detailed contribution breakdown for each game in a team's season. Diagnostic view showing how each game affects the final rating.

```bash
ncaa-rank decompose <season> <team> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--profile` | `-p` | Configuration profile |
| `--config` | `-c` | Path to custom JSON config file |

```bash
# Examples
ncaa-rank decompose 2024 ohio-state
ncaa-rank decompose 2024 "Notre Dame" --profile predictive
```

### explain

Show detailed rating breakdown for a team including record, strength of schedule, and key wins/losses.

```bash
ncaa-rank explain <season> <team> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--week` | `-w` | Explain rating as of this week |

```bash
# Examples
ncaa-rank explain 2024 ohio-state
ncaa-rank explain 2024 georgia --week 10
```

### diagnose

Analyze prediction accuracy and calibration. Shows Brier score, calibration error by probability bucket, and biggest upsets.

```bash
ncaa-rank diagnose <season> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--week` | `-w` | Analyze through this week only |
| `--profile` | `-p` | Configuration profile |
| `--config` | `-c` | Path to custom JSON config file |
| `--upsets` | `-u` | Number of biggest upsets to show (default: 5) |
| `--attributions` | `-a` | Number of parameter attributions to show (default: 5) |

```bash
# Examples
ncaa-rank diagnose 2024 --profile predictive
ncaa-rank diagnose 2024 --upsets 10 --attributions 10
```

### compare

Compare algorithm rankings to human polls (AP, CFP, Coaches).

```bash
ncaa-rank compare <season> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--poll` | `-p` | Poll to compare: `ap` (default), `cfp`, or `coaches` |
| `--week` | `-w` | Compare for specific week |

```bash
# Examples
ncaa-rank compare 2024 --poll ap
ncaa-rank compare 2024 --poll cfp --week 15
```

### export

Export rankings to CSV or JSON file.

```bash
ncaa-rank export <season> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--format` | `-f` | Output format: `csv` (default) or `json` |
| `--output` | `-o` | Output file path |
| `--week` | `-w` | Export rankings as of this week |
| `--top` | `-t` | Number of teams to export (default: 25) |

```bash
# Examples
ncaa-rank export 2024 --format csv --output rankings.csv
ncaa-rank export 2024 --format json --top 130
ncaa-rank export 2024 --week 10 -o week10.csv
```

### config

Configuration management subcommands.

#### config list

List all available configuration profiles.

```bash
ncaa-rank config list
```

#### config show

Show configuration settings for a profile or the defaults.

```bash
ncaa-rank config show [profile]
```

```bash
# Examples
ncaa-rank config show              # Show default values
ncaa-rank config show predictive   # Show predictive profile
```

#### config export

Export a configuration profile to a JSON file.

```bash
ncaa-rank config export <output> [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--profile` | `-p` | Profile to export (default: defaults) |

```bash
# Examples
ncaa-rank config export my_config.json
ncaa-rank config export tuned.json --profile tuned_predictive
```

#### config create

Create a new configuration file with all levers documented as comments.

```bash
ncaa-rank config create [output] [options]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--profile` | `-p` | Base profile to start from |
| `--documented` | `-d` | Include section comments (default: true) |

```bash
# Examples
ncaa-rank config create my_config.json
ncaa-rank config create custom.json --profile predictive
```

---

## Understanding the Analysis Tools

The CLI provides several analysis commands that reveal different aspects of team performance and algorithm behavior.

### Team Decomposition

The `decompose` command shows exactly how each game contributed to a team's final rating. This is useful for understanding why a team is ranked where they are.

For each game, you see:
- **Game grade**: The base score (0.70 for win, 0.00 for loss) plus margin bonus and venue adjustment
- **Opponent rating**: The opponent's converged rating at season end
- **Contribution**: Game grade plus/minus opponent rating (the value that gets averaged)
- **Weight**: If recency is enabled, how much this game counts

Example insight: A team might have a high rating despite a loss because that loss was to a highly-rated opponent (large negative contribution) while their wins were against solid opponents (solid positive contributions). The decomposition shows this math explicitly.

This also reveals when a team's rating is "propped up" by one or two big wins against teams that turned out to be good, versus teams with consistent quality across all games.

### Team Explanation

The `explain` command provides a higher-level summary:
- Overall record and rating
- Strength of schedule (average opponent rating)
- Best wins (sorted by opponent rating)
- Worst losses (sorted by opponent rating)
- Conference breakdown

This answers questions like "Why is Team X ranked #8?" by showing their quality wins and the overall difficulty of their schedule.

### Diagnostics

The `diagnose` command evaluates how well the algorithm predicted game outcomes over a season. It computes:

**Accuracy**: Simple percentage of games where the favorite (by our rating) won.

**Brier Score**: A measure of probabilistic prediction quality. If you predict Team A has a 70% chance and they win, your Brier score for that game is (1.0 - 0.70)² = 0.09. Lower is better. A Brier score of 0.25 is random guessing; anything below 0.20 indicates meaningful predictive power.

**Calibration**: Are predicted probabilities accurate? If you predict 70% confidence across many games, do those teams actually win ~70% of the time? The diagnostics show calibration by probability bucket:

```
Bucket      Games    Win%     Expected    Error
50-60%      45       52.4%    55.0%       -2.6%
60-70%      38       68.4%    65.0%       +3.4%
70-80%      29       75.9%    75.0%       +0.9%
80-90%      18       88.9%    85.0%       +3.9%
90-100%     12       100.0%   95.0%       +5.0%
```

This reveals systematic biases. If high-confidence predictions are underperforming, you might be overconfident. If low-confidence predictions are outperforming, you might be too conservative.

**Biggest Upsets**: Games where the higher-rated team lost, sorted by rating gap. These are the games the algorithm "got wrong" and are useful for understanding limitations.

**Parameter Attribution**: Which configuration parameters most affected accuracy? This helps tune the algorithm by showing whether margin weight, venue adjustments, or other factors are helping or hurting predictions.

### Vegas Upset Analysis

The `vegas-analysis` command finds games where Vegas favorites lost outright. This is interesting for several reasons:

1. **Finding market inefficiencies**: Vegas lines reflect betting market consensus. When we disagree and are right, we may have found something the market missed.

2. **Pattern detection**: The analyzer categorizes upsets by:
   - Spread bucket (slight favorite, moderate favorite, heavy favorite)
   - Conference matchup (P5 vs G5, conference game, etc.)
   - Week of season (early, mid, late)
   - Anomaly factors (weak schedule, road favorite, etc.)

3. **Algorithm validation**: If our algorithm consistently predicts upsets that Vegas misses, that's evidence the algorithm captures something real. If we also miss them, it suggests those games were genuinely unpredictable.

Example output might show that 40% of P5-road-at-G5 upsets were predicted by our algorithm while only being 20% of total upsets—suggesting we handle that specific situation better than the market.

The `--we-got-right` filter isolates upsets where our algorithm had the underdog winning, which is the most actionable output for identifying where this system adds value over Vegas lines.

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

### Customizing the Algorithm

To create your own configuration:

```bash
# Generate a documented config file starting from a profile
ncaa-rank config create my_config.json --profile predictive

# Or export an existing profile and modify it
ncaa-rank config export my_config.json --profile balanced
```

The generated JSON file includes all parameters with their current values. Edit the file to adjust any lever:

```json
{
  "win_base": 0.70,
  "margin_weight": 0.15,
  "margin_cap": 28,
  "venue_road_win": 0.08,
  "venue_home_loss": -0.03,
  "opponent_weight": 0.9,
  "enable_conference_adj": true,
  "p5_multiplier": 1.05,
  "g5_multiplier": 0.85
}
```

Then use your config:

```bash
ncaa-rank rank 2024 --config my_config.json
ncaa-rank diagnose 2024 --config my_config.json
```

**Tuning workflow:**

1. Start with a base profile (`predictive` for forecasting, `pure_results` for minimal assumptions)
2. Run diagnostics to see current accuracy: `ncaa-rank diagnose 2024 --profile predictive`
3. Export and modify specific parameters
4. Re-run diagnostics with your config to measure impact
5. Iterate until satisfied

Common adjustments:
- **Reduce margin_weight** if blowouts are being overvalued
- **Increase g5_multiplier** if G5 teams seem underrated
- **Enable recency** if late-season form should matter more
- **Adjust venue bonuses** if road/home predictions seem miscalibrated

The diagnostics output shows parameter attribution, indicating which levers had the most impact on prediction accuracy for that season.

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

#### SP+ (ESPN)

**Methodology:** Efficiency-based rating developed by Bill Connelly. Measures points per play adjusted for opponent, game situation, and garbage time. Splits into offensive and defensive components.

**Strengths:**
- Adjusts for pace (points per play, not per game)
- Removes garbage time scoring
- Considers play-by-play context

**Weaknesses:**
- Proprietary formula, not fully reproducible
- Preseason priors can persist into late season

#### SRS (Simple Rating System)

**Methodology:** Average margin of victory adjusted for strength of schedule. Iterative calculation similar to this algorithm but using raw margin instead of game grades.

```
SRS = average_margin + average_opponent_SRS
```

**Strengths:**
- Simple and transparent
- No preseason priors
- Easy to verify

**Weaknesses:**
- Treats 28-0 and 56-28 identically
- No venue adjustment
- Susceptible to garbage time

#### Elo

**Methodology:** Rating system where teams exchange points based on expected vs actual outcomes. Originally from chess, adapted for football. After each game:

```
new_rating = old_rating + K * (actual - expected)
```

Where K is typically 20-32 and expected is based on rating difference.

**Strengths:**
- Updates incrementally after each game
- Self-correcting over time
- Well-understood mathematical properties

**Weaknesses:**
- Requires preseason initialization
- Doesn't account for margin (in basic form)
- Slow to react to mid-season changes

#### Vegas Lines

**Methodology:** Point spreads set by sportsbooks and moved by betting action. Reflects the market's best estimate of the point differential.

**Strengths:**
- Incorporates injury/suspension information
- Crowd-sourced wisdom with real money on the line
- Best historical track record for game predictions

**Weaknesses:**
- Designed to balance betting action, not predict scores
- Not available for all games
- Can reflect public perception biases

### Interpreting Validation Results

When our rankings diverge significantly from all validators, it suggests either:
1. Our algorithm found something others missed
2. Our algorithm is overweighting some factor

Common patterns for overrated teams in our system:
- **G5 teams with weak schedules** - High margins against weak opponents inflate ratings
- **Teams with quality losses early** - Convergence eventually corrects, but early-season losses to later-good teams may not be recognized
- **Teams in weak conferences** - Conference adjustment disabled in `pure_results` profile

Use `ncaa-rank validate 2024 --team <team>` for deep-dive analysis.

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
