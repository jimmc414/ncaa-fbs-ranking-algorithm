# NCAA FBS Ranking Algorithm

A bias-free college football ranking system using iterative convergence (PageRank-like). The algorithm derives team ratings purely from game outcomes, margin of victory, venue context, and opponent strength—eliminating the human voting bias inherent in traditional polls.

## Key Features

- **No Preseason Bias**: Every team starts equal; rankings are earned on the field
- **Iterative Convergence**: Ratings stabilize through repeated calculation until equilibrium
- **Configurable Engine**: 45 adjustable parameters to tune ranking behavior
- **Full Stack**: CLI + Web Dashboard + JSON/CSV Export
- **Historical Comparison**: Compare algorithm rankings to AP Poll and CFP

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure API key (get free key at https://collegefootballdata.com/key)
cp .env.example .env
# Edit .env and add your CFBD_API_KEY

# Generate rankings
ncaa-rank rank 2024 --top 25

# Explain a team's rating
ncaa-rank explain 2024 ohio-state

# Compare to AP Poll
ncaa-rank compare 2024 --poll ap

# Start web dashboard
uvicorn src.web.app:app --reload
# Open http://localhost:8000
```

---

## Algorithm Specification

### Core Formula

For each team, the rating is calculated as:

```
Rating = (1/N) * SUM[ GameGrade + OpponentFactor * OpponentRating * OpponentWeight ]
```

Where:
- **N** = Number of games played
- **GameGrade** = Base value + Margin bonus + Venue adjustment
- **OpponentFactor** = +1 for wins, -1 for losses (configurable)
- **OpponentRating** = Current rating of opponent (iteratively refined)

### The Loss Penalty (Critical Insight)

The key differentiator from other algorithms: **losses subtract opponent rating**.

```python
if is_win:
    contribution = game_grade + opponent_rating   # Beating good teams helps
else:
    contribution = game_grade - opponent_rating   # Losing to good teams hurts less
                                                  # Losing to bad teams hurts more
```

---

## Configuration Levers

The algorithm supports 45 configurable parameters organized into 11 categories.

### 1. Core Game Scoring

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `base_win_value` | Base points awarded for any win | 0.70 | 0.5-1.0 |
| `base_loss_value` | Base points for a loss | 0.00 | 0.0-0.3 |
| `margin_cap` | Maximum point differential that counts | 28 | 14-56 |
| `margin_coefficient` | Weight of margin bonus (wins only) | 0.20 | 0.0-0.4 |
| `margin_curve` | Diminishing returns curve shape | "log" | log/linear/sqrt |

**Margin Bonus Formula (Logarithmic):**
```
bonus = margin_coefficient * ln(1 + min(margin, margin_cap)) / ln(1 + margin_cap)
```

### 2. Venue Adjustments

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `road_win_bonus` | Bonus for winning on the road | +0.10 | 0.0-0.20 |
| `neutral_win_bonus` | Bonus for winning at neutral site | +0.05 | 0.0-0.15 |
| `home_loss_penalty` | Penalty for losing at home | -0.03 | -0.15-0.0 |
| `neutral_loss_penalty` | Penalty for losing at neutral site | -0.01 | -0.10-0.0 |

### 3. Opponent Influence

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `opponent_weight` | Multiplier on opponent rating contribution | 1.0 | 0.5-1.5 |
| `loss_opponent_factor` | How opponent rating affects losses | -1.0 | -1.5--0.5 |
| `second_order_weight` | Weight of opponent's SOS in contribution | 0.0 | 0.0-0.3 |

### 4. Conference Adjustments

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `enable_conference_adj` | Master switch for conference factors | false | bool |
| `conference_method` | How to calculate conference strength | "empirical" | manual/empirical |
| `p5_multiplier` | Multiplier for beating P5 opponents | 1.00 | 1.0-1.25 |
| `g5_multiplier` | Multiplier for beating G5 opponents | 1.00 | 0.75-1.0 |
| `fcs_multiplier` | Multiplier for beating FCS opponents | 0.50 | 0.3-0.7 |
| `fcs_fixed_rating` | Fixed rating assigned to FCS teams | 0.20 | 0.1-0.3 |

### 5. Schedule Strength Adjustments

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `sos_adjustment_weight` | Post-convergence SOS adjustment factor | 0.0 | 0.0-0.25 |
| `sos_method` | How to calculate SOS | "mean" | mean/median |
| `min_sos_top_10` | Minimum SOS required for top 10 | 0.0 | 0.0-0.50 |
| `min_sos_top_25` | Minimum SOS required for top 25 | 0.0 | 0.0-0.40 |
| `min_p5_games_top_10` | Minimum P5 opponents for top 10 | 0 | 0-5 |

### 6. Quality Tiers

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `enable_quality_tiers` | Enable tiered win/loss bonuses | false | bool |
| `elite_threshold` | Rating above this = elite opponent | 0.80 | 0.70-0.90 |
| `good_threshold` | Rating above this = good opponent | 0.55 | 0.45-0.65 |
| `bad_threshold` | Rating below this = bad opponent | 0.35 | 0.25-0.45 |
| `elite_win_bonus` | Bonus for beating elite team | 0.0 | 0.0-0.20 |
| `bad_loss_penalty` | Extra penalty for losing to bad team | 0.0 | 0.0-0.25 |

### 7. Recency Weighting

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `enable_recency` | Weight recent games more heavily | false | bool |
| `recency_half_life` | Weeks until game weight halves | 8 | 4-16 |
| `recency_min_weight` | Minimum weight for oldest games | 0.5 | 0.25-1.0 |

### 8. Historical Prior

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `enable_prior` | Use historical data as starting point | false | bool |
| `prior_weight` | Blend factor for prior | 0.0 | 0.0-0.30 |
| `prior_decay_weeks` | Weeks until prior fully fades | 8 | 4-16 |
| `prior_source` | What to use as prior | "last_year" | last_year/recruiting/elo |

### 9. Convergence

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `initial_rating` | Starting rating for all teams | 0.50 | 0.3-0.7 |
| `convergence_threshold` | Max delta to consider converged | 0.0001 | 0.00001-0.001 |
| `max_iterations` | Hard stop on iterations | 100 | 50-500 |

### 10. Tiebreakers

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `tie_threshold` | Rating difference to trigger tiebreaker | 0.001 | 0.0005-0.01 |
| `tiebreaker_order` | Priority order of tiebreakers | [h2h,sov,common,away] | list |

**Tiebreaker Methods:**
1. **h2h**: Head-to-head result between tied teams
2. **sov**: Strength of Victory (average rating of beaten opponents)
3. **common**: Common opponent margin differential
4. **away**: Away game win percentage

### 11. Game Filters

| Lever | Description | Default | Range |
|-------|-------------|---------|-------|
| `include_postseason` | Include bowl/playoff games | true | bool |
| `postseason_weight` | Weight multiplier for postseason games | 1.0 | 0.5-2.0 |
| `include_fcs_games` | Include games vs FCS opponents | true | bool |
| `min_games_to_rank` | Minimum games played to receive ranking | 5 | 1-8 |

---

## Algorithm Pseudocode

```
================================================================================
ALGORITHM: NCAA FBS RANKING ENGINE
================================================================================

INPUT:
    games[]          : List of completed games with scores, venues, dates
    config           : Configuration object with all lever values
    prior_ratings{}  : Optional historical ratings (if prior enabled)

OUTPUT:
    rankings[]       : Ordered list of (team, rank, rating, record, sos, sov)

--------------------------------------------------------------------------------
PHASE 1: INITIALIZATION
--------------------------------------------------------------------------------

1.1  FILTER GAMES:
     games = filter(games, where:
         - game.is_completed = true
         - if NOT config.include_postseason: game.is_postseason = false
         - if NOT config.include_fcs_games: both teams are FBS
     )

1.2  EXTRACT TEAMS:
     teams = unique(games.home_team UNION games.away_team)

1.3  INITIALIZE RATINGS:
     FOR each team in teams:
         IF config.enable_prior AND team in prior_ratings:
             ratings[team] = config.initial_rating * (1 - config.prior_weight)
                           + prior_ratings[team] * config.prior_weight
         ELSE:
             ratings[team] = config.initial_rating

1.4  BUILD GAME LOOKUP:
     FOR each team in teams:
         team_games[team] = games where team is home or away

1.5  CLASSIFY CONFERENCES (if enabled):
     IF config.enable_conference_adj:
         FOR each team:
             team_tier[team] = lookup_conference_tier(team)  -> "P5" | "G5" | "FCS"

         IF config.conference_method = "empirical":
             conference_multipliers = calculate_from_cross_conference_results(games)
         ELSE:
             conference_multipliers = {
                 "P5": config.p5_multiplier,
                 "G5": config.g5_multiplier,
                 "FCS": config.fcs_multiplier
             }

--------------------------------------------------------------------------------
PHASE 2: GAME GRADE CALCULATION
--------------------------------------------------------------------------------

FUNCTION calculate_game_grade(game, team, config):

    2.1  DETERMINE PERSPECTIVE:
         IF team = game.home_team:
             is_home = true
             team_score = game.home_score
             opp_score = game.away_score
             opponent = game.away_team
         ELSE:
             is_home = false
             team_score = game.away_score
             opp_score = game.home_score
             opponent = game.home_team

         is_win = team_score > opp_score
         margin = |team_score - opp_score|
         is_neutral = game.neutral_site

    2.2  BASE VALUE:
         IF is_win:
             grade = config.base_win_value
         ELSE:
             grade = config.base_loss_value

    2.3  MARGIN BONUS (wins only):
         IF is_win:
             capped_margin = min(margin, config.margin_cap)

             IF config.margin_curve = "log":
                 bonus = ln(1 + capped_margin) / ln(1 + config.margin_cap)
             ELSE IF config.margin_curve = "sqrt":
                 bonus = sqrt(capped_margin) / sqrt(config.margin_cap)
             ELSE:  # linear
                 bonus = capped_margin / config.margin_cap

             grade = grade + config.margin_coefficient * bonus

    2.4  VENUE ADJUSTMENT:
         IF is_win:
             IF NOT is_home AND NOT is_neutral:  # road win
                 grade = grade + config.road_win_bonus
             ELSE IF is_neutral:
                 grade = grade + config.neutral_win_bonus
         ELSE:  # loss
             IF is_home:
                 grade = grade + config.home_loss_penalty
             ELSE IF is_neutral:
                 grade = grade + config.neutral_loss_penalty

    2.5  CONFERENCE ADJUSTMENT (if enabled):
         IF config.enable_conference_adj:
             opp_tier = team_tier[opponent]
             grade = grade * conference_multipliers[opp_tier]

    2.6  RECENCY WEIGHT (if enabled):
         IF config.enable_recency:
             current_week = max(g.week for g in games)
             weeks_ago = current_week - game.week
             decay = 0.5 ^ (weeks_ago / config.recency_half_life)
             recency_weight = max(decay, config.recency_min_weight)
         ELSE:
             recency_weight = 1.0

    2.7  POSTSEASON WEIGHT:
         IF game.is_postseason:
             postseason_weight = config.postseason_weight
         ELSE:
             postseason_weight = 1.0

    RETURN {
        grade: grade,
        opponent: opponent,
        is_win: is_win,
        weight: recency_weight * postseason_weight
    }

--------------------------------------------------------------------------------
PHASE 3: ITERATIVE CONVERGENCE
--------------------------------------------------------------------------------

3.1  PRECOMPUTE GAME GRADES:
     FOR each game in games:
         FOR each team in [game.home_team, game.away_team]:
             game_results[game][team] = calculate_game_grade(game, team, config)

3.2  ITERATE UNTIL CONVERGENCE:
     iteration = 0
     converged = false

     WHILE NOT converged AND iteration < config.max_iterations:

         new_ratings = {}
         max_delta = 0

         # IMPORTANT: Process teams in sorted order for determinism
         FOR each team in sorted(teams):

             total_contribution = 0
             total_weight = 0

             FOR each game in team_games[team]:
                 result = game_results[game][team]
                 opponent = result.opponent

                 # Get opponent rating (use FCS fixed rating if applicable)
                 IF team_tier[opponent] = "FCS":
                     opp_rating = config.fcs_fixed_rating
                 ELSE:
                     opp_rating = ratings[opponent]

                 # Core contribution calculation
                 IF result.is_win:
                     opp_factor = 1.0
                 ELSE:
                     opp_factor = config.loss_opponent_factor  # typically -1.0

                 contribution = result.grade
                              + opp_factor * opp_rating * config.opponent_weight

                 # Second-order effect: opponent's schedule strength
                 IF config.second_order_weight > 0:
                     opp_sos = calculate_sos(opponent, ratings)
                     contribution = contribution + config.second_order_weight * opp_sos

                 # Quality tier bonuses/penalties
                 IF config.enable_quality_tiers:
                     IF result.is_win AND opp_rating >= config.elite_threshold:
                         contribution = contribution + config.elite_win_bonus
                     IF NOT result.is_win AND opp_rating <= config.bad_threshold:
                         contribution = contribution - config.bad_loss_penalty

                 total_contribution = total_contribution + contribution * result.weight
                 total_weight = total_weight + result.weight

             # New rating is weighted average of contributions
             IF total_weight > 0:
                 new_ratings[team] = total_contribution / total_weight
             ELSE:
                 new_ratings[team] = config.initial_rating

             # Track maximum change for convergence check
             delta = |new_ratings[team] - ratings[team]|
             max_delta = max(max_delta, delta)

         # Update ratings for next iteration
         ratings = new_ratings
         iteration = iteration + 1

         # Check convergence
         IF max_delta < config.convergence_threshold:
             converged = true

     STORE iterations_to_converge = iteration

--------------------------------------------------------------------------------
PHASE 4: POST-CONVERGENCE ADJUSTMENTS
--------------------------------------------------------------------------------

4.1  NORMALIZE TO [0, 1]:
     min_rating = min(ratings.values())
     max_rating = max(ratings.values())
     range = max_rating - min_rating

     IF range > 0:
         FOR each team in teams:
             normalized[team] = (ratings[team] - min_rating) / range
     ELSE:
         FOR each team in teams:
             normalized[team] = 0.5

4.2  CALCULATE STRENGTH METRICS:
     FOR each team in teams:
         opponents = get_all_opponents(team, games)
         sos[team] = mean(normalized[opp] for opp in opponents)

         beaten_opponents = get_opponents_team_defeated(team, games)
         IF len(beaten_opponents) > 0:
             sov[team] = mean(normalized[opp] for opp in beaten_opponents)
         ELSE:
             sov[team] = 0

4.3  SCHEDULE STRENGTH ADJUSTMENT (if enabled):
     IF config.sos_adjustment_weight > 0:
         avg_sos = mean(sos.values())

         FOR each team in teams:
             adjustment = config.sos_adjustment_weight * (sos[team] - avg_sos)
             adjusted[team] = normalized[team] * (1 + adjustment)
     ELSE:
         adjusted = normalized

4.4  COUNT WINS/LOSSES:
     FOR each team in teams:
         record[team] = count_wins_losses(team, games)

--------------------------------------------------------------------------------
PHASE 5: RANKING AND TIEBREAKERS
--------------------------------------------------------------------------------

5.1  INITIAL SORT:
     ranked_teams = sort(teams, by: adjusted[team], descending: true)

5.2  APPLY MINIMUM THRESHOLDS:
     FOR each team in ranked_teams:
         IF record[team].games_played < config.min_games_to_rank:
             MOVE team to end of rankings (unranked)

         IF position <= 10 AND sos[team] < config.min_sos_top_10:
             DEMOTE team below position 10

         IF position <= 25 AND sos[team] < config.min_sos_top_25:
             DEMOTE team below position 25

         IF config.enable_conference_adj AND position <= 10:
             p5_game_count = count_p5_opponents(team, games)
             IF p5_game_count < config.min_p5_games_top_10:
                 DEMOTE team below position 10

5.3  RESOLVE TIES:
     FOR i = 0 to len(ranked_teams) - 2:
         team_a = ranked_teams[i]
         team_b = ranked_teams[i + 1]

         IF |adjusted[team_a] - adjusted[team_b]| < config.tie_threshold:
             winner = apply_tiebreakers(team_a, team_b, config.tiebreaker_order)
             IF winner = team_b:
                 SWAP ranked_teams[i] and ranked_teams[i + 1]

     FUNCTION apply_tiebreakers(team_a, team_b, order):
         FOR each tiebreaker in order:
             IF tiebreaker = "h2h":
                 result = head_to_head(team_a, team_b, games)
                 IF result != TIE: RETURN result

             IF tiebreaker = "sov":
                 IF sov[team_a] > sov[team_b] + 0.001: RETURN team_a
                 IF sov[team_b] > sov[team_a] + 0.001: RETURN team_b

             IF tiebreaker = "common":
                 result = common_opponent_margin(team_a, team_b, games)
                 IF result != TIE: RETURN result

             IF tiebreaker = "away":
                 away_pct_a = away_win_percentage(team_a, games)
                 away_pct_b = away_win_percentage(team_b, games)
                 IF away_pct_a > away_pct_b + 0.01: RETURN team_a
                 IF away_pct_b > away_pct_a + 0.01: RETURN team_b

         RETURN TIE  # maintain original order

5.4  ASSIGN FINAL RANKS:
     FOR i = 0 to len(ranked_teams) - 1:
         team = ranked_teams[i]
         final_rank[team] = i + 1

--------------------------------------------------------------------------------
PHASE 6: OUTPUT
--------------------------------------------------------------------------------

6.1  BUILD RESULTS:
     results = []

     FOR each team in ranked_teams:
         results.append({
             team_id: team,
             rank: final_rank[team],
             rating: adjusted[team],
             wins: record[team].wins,
             losses: record[team].losses,
             games_played: record[team].games_played,
             strength_of_schedule: sos[team],
             strength_of_victory: sov[team],
             convergence_iterations: iterations_to_converge
         })

6.2  RETURN results

================================================================================
END ALGORITHM
================================================================================
```

---

## Helper Functions

### Calculate Empirical Conference Multipliers

```
FUNCTION calculate_empirical_multipliers(games):

    # Find all cross-tier games and calculate win rates
    p5_vs_g5_games = filter(games, where one team is P5 and other is G5)
    p5_wins = count(p5_vs_g5_games where P5 team won)
    p5_win_rate = p5_wins / len(p5_vs_g5_games)  # typically ~0.75

    g5_vs_fcs_games = filter(games, where one team is G5 and other is FCS)
    g5_wins = count(g5_vs_fcs_games where G5 team won)
    g5_win_rate = g5_wins / len(g5_vs_fcs_games)  # typically ~0.85

    # Convert win rates to multipliers (scaled to reasonable range)
    g5_multiplier = 1.0 - (p5_win_rate - 0.5) * 0.5
    fcs_multiplier = g5_multiplier * (1.0 - (g5_win_rate - 0.5) * 0.5)

    RETURN {
        "P5": 1.0,
        "G5": g5_multiplier,    # typically 0.80-0.90
        "FCS": fcs_multiplier   # typically 0.50-0.70
    }
```

### Head-to-Head Resolution

```
FUNCTION head_to_head(team_a, team_b, games):
    h2h_games = filter(games, where teams are team_a and team_b)

    IF len(h2h_games) = 0:
        RETURN TIE

    a_wins = count(h2h_games where team_a won)
    b_wins = count(h2h_games where team_b won)

    IF a_wins > b_wins: RETURN team_a
    IF b_wins > a_wins: RETURN team_b
    RETURN TIE
```

### Common Opponent Margin

```
FUNCTION common_opponent_margin(team_a, team_b, games):
    opponents_a = get_opponents(team_a, games)
    opponents_b = get_opponents(team_b, games)
    common = opponents_a INTERSECT opponents_b

    IF len(common) = 0:
        RETURN TIE

    margin_a = sum(get_margin(team_a, opp, games) for opp in common)
    margin_b = sum(get_margin(team_b, opp, games) for opp in common)

    IF margin_a > margin_b + 3: RETURN team_a
    IF margin_b > margin_a + 3: RETURN team_b
    RETURN TIE
```

---

## Configuration Profiles

### Profile: Pure Results (Default)

The original algorithm—purely results-based with no adjustments.

```yaml
base_win_value: 0.70
base_loss_value: 0.00
margin_cap: 28
margin_coefficient: 0.20
margin_curve: "log"
road_win_bonus: 0.10
neutral_win_bonus: 0.05
home_loss_penalty: -0.03
neutral_loss_penalty: -0.01
opponent_weight: 1.0
loss_opponent_factor: -1.0
enable_conference_adj: false
enable_quality_tiers: false
enable_recency: false
enable_prior: false
sos_adjustment_weight: 0.0
second_order_weight: 0.0
fcs_fixed_rating: 0.20
convergence_threshold: 0.0001
max_iterations: 100
tie_threshold: 0.001
tiebreaker_order: [h2h, sov, common, away]
```

### Profile: Balanced

Recommended starting point with moderate conference and quality adjustments.

```yaml
base_win_value: 0.70
margin_cap: 28
margin_coefficient: 0.20
enable_conference_adj: true
conference_method: "empirical"
enable_quality_tiers: true
elite_threshold: 0.80
elite_win_bonus: 0.05
bad_threshold: 0.35
bad_loss_penalty: 0.10
sos_adjustment_weight: 0.10
second_order_weight: 0.10
min_p5_games_top_10: 2
```

### Profile: Predictive

Optimized for bowl game prediction accuracy.

```yaml
enable_conference_adj: true
conference_method: "empirical"
g5_multiplier: 0.85
fcs_multiplier: 0.50
enable_quality_tiers: true
elite_threshold: 0.75
elite_win_bonus: 0.10
bad_threshold: 0.40
bad_loss_penalty: 0.15
sos_adjustment_weight: 0.15
second_order_weight: 0.15
min_sos_top_10: 0.45
min_sos_top_25: 0.35
min_p5_games_top_10: 3
enable_recency: true
recency_half_life: 6
recency_min_weight: 0.6
```

### Profile: Conservative

Mimics traditional poll behavior with historical priors and recency weighting.

```yaml
enable_prior: true
prior_weight: 0.20
prior_decay_weeks: 8
prior_source: "last_year"
enable_recency: true
recency_half_life: 6
recency_min_weight: 0.5
sos_adjustment_weight: 0.10
enable_quality_tiers: true
bad_loss_penalty: 0.05
```

---

## Project Structure

```
src/
├── algorithm/
│   ├── convergence.py      # Iterative rating solver
│   ├── game_grade.py       # Game grade calculation
│   └── tiebreaker.py       # Tie resolution logic
├── data/
│   ├── models.py           # Pydantic data models
│   ├── client.py           # CollegeFootballData API client
│   └── storage.py          # SQLite persistence
├── ranking/
│   ├── engine.py           # Orchestration layer
│   └── comparison.py       # Poll comparison (Spearman correlation)
├── cli/
│   └── main.py             # Typer CLI commands
└── web/
    ├── app.py              # FastAPI application
    └── templates/          # Jinja2 + HTMX templates
```

---

## CLI Reference

```bash
# Generate rankings
ncaa-rank rank <season> [--week N] [--top N]

# Explain team rating with game-by-game breakdown
ncaa-rank explain <season> <team-id> [--week N]

# Compare to AP/CFP poll
ncaa-rank compare <season> --poll ap|cfp|coaches [--week N]

# Export to file
ncaa-rank export <season> --format csv|json [--output path]
```

## Web Dashboard

```bash
# Start server
uvicorn src.web.app:app --reload

# Routes:
# /                      - Redirect to current season
# /rankings/{season}     - Full rankings page
# /teams/{id}/{season}   - Team detail with game breakdown
# /compare/{season}      - Poll comparison page
```

---

## Data Sources

- **Game Data**: [CollegeFootballData.com API](https://collegefootballdata.com/)
- **API Key**: Required in `.env` file as `CFBD_API_KEY`
- Get a free key at: https://collegefootballdata.com/key

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Current status: 257 tests, 91% coverage
```

---

## License

MIT
