# Configuration Reference

All 44 algorithm levers with descriptions, valid ranges, and effects.

## Quick Start

```bash
# Export a profile as starting point
ncaa-rank config export my_config.json --profile predictive

# Edit the file, then run with it
ncaa-rank rank 2025 --config my_config.json --top 25

# Or create from example
cp config.example.json my_config.json
```

---

## Convergence Settings

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `max_iterations` | 100 | 10-1000 | Maximum iterations before stopping |
| `convergence_threshold` | 0.0001 | 0.00001-0.01 | Stop when max rating change is below this |

**Effect:** Lower threshold = more precise but slower. Default is usually sufficient.

---

## Game Grade Settings

How individual game outcomes are scored before opponent strength is applied.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `win_base` | 0.70 | 0.5-1.0 | Base score for any win |
| `margin_weight` | 0.20 | 0.0-0.5 | Maximum bonus for margin of victory |
| `margin_cap` | 28 | 14-42 | Points of margin beyond which no extra credit |
| `margin_curve` | "log" | log/linear/sqrt | How margin bonus scales |

**Effect:**
- Higher `win_base` = wins matter more than margin
- Higher `margin_weight` = blowouts valued more
- `log` curve rewards first touchdown of margin more than third
- `linear` curve treats all margin points equally

---

## Venue Adjustments

Bonuses/penalties based on game location.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `venue_road_win` | 0.10 | 0.0-0.20 | Bonus for winning on the road |
| `venue_neutral_win` | 0.05 | 0.0-0.15 | Bonus for winning at neutral site |
| `venue_home_loss` | -0.03 | -0.10-0.0 | Penalty for losing at home |
| `venue_neutral_loss` | -0.01 | -0.05-0.0 | Penalty for losing at neutral site |

**Effect:** Positive values reward road performance, negative penalize home failures.

---

## Initialization

Starting ratings before iteration.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `initial_rating` | 0.500 | 0.3-0.7 | Starting rating for all FBS teams |
| `fcs_fixed_rating` | 0.20 | 0.1-0.4 | Fixed rating for FCS opponents (null to iterate) |

**Effect:** `fcs_fixed_rating` prevents FCS teams from inflating due to limited data. Lower values = less credit for beating FCS.

---

## Opponent Influence

How opponent quality affects your rating.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `opponent_weight` | 1.0 | 0.5-2.0 | Multiplier on opponent rating in contribution |
| `loss_opponent_factor` | 1.0 | 0.5-2.0 | Scale for loss penalty (penalty = (1-opp_rating) × factor) |
| `second_order_weight` | 0.0 | 0.0-0.5 | Weight opponent's SOS in calculation |

**Effect:**
- Higher `opponent_weight` = SOS matters more
- `loss_opponent_factor` scales the "quality losses hurt less" penalty:
  - Loss to 0.9-rated team: penalty = (1-0.9) × 1.0 = **0.1** (small)
  - Loss to 0.1-rated team: penalty = (1-0.1) × 1.0 = **0.9** (big)
- `second_order_weight` adds opponents-of-opponents consideration

---

## Conference Adjustments

P5/G5/FCS tier multipliers.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `enable_conference_adj` | false | true/false | Enable conference tier adjustments |
| `conference_method` | "empirical" | empirical/manual | How multipliers are determined |
| `p5_multiplier` | 1.0 | 0.9-1.2 | Multiplier for P5 conference teams |
| `g5_multiplier` | 1.0 | 0.7-1.1 | Multiplier for G5 conference teams |
| `fcs_multiplier` | 0.5 | 0.3-0.8 | Multiplier for FCS teams |

**Effect:** When enabled, adjusts game grades based on opponent's conference tier. Use to penalize weak scheduling.

**Predictive profile uses:** `p5_multiplier=1.05`, `g5_multiplier=0.85`

---

## Schedule Strength

SOS requirements and post-adjustments.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `sos_adjustment_weight` | 0.0 | 0.0-0.3 | Post-hoc adjustment based on SOS deviation |
| `sos_method` | "mean" | mean/median | How SOS is calculated |
| `min_sos_top_10` | 0.0 | 0.0-0.6 | Minimum SOS required for top 10 |
| `min_sos_top_25` | 0.0 | 0.0-0.5 | Minimum SOS required for top 25 |
| `min_p5_games_top_10` | 0 | 0-6 | Minimum P5 games required for top 10 |

**Effect:** SOS adjustment rewards teams with harder schedules after base rating is computed.

---

## Quality Tiers

Bonuses for elite wins, penalties for bad losses.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `enable_quality_tiers` | false | true/false | Enable quality tier system |
| `elite_threshold` | 0.80 | 0.70-0.90 | Rating threshold to be "elite" opponent |
| `good_threshold` | 0.55 | 0.45-0.65 | Rating threshold to be "good" opponent |
| `bad_threshold` | 0.35 | 0.25-0.45 | Rating below which opponent is "bad" |
| `elite_win_bonus` | 0.0 | 0.0-0.20 | Bonus added for beating elite opponent |
| `bad_loss_penalty` | 0.0 | 0.0-0.25 | Penalty subtracted for losing to bad team |

**Effect:** Rewards resume-building wins (beat top 10 teams), punishes bad losses (lose to bottom 25%).

**Predictive profile uses:** `elite_win_bonus=0.10`, `bad_loss_penalty=0.15`

---

## Recency Weighting

Weight recent games more heavily.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `enable_recency` | false | true/false | Enable recency weighting |
| `recency_half_life` | 8 | 3-12 | Weeks for weight to halve |
| `recency_min_weight` | 0.5 | 0.2-0.8 | Minimum weight for oldest games |

**Effect:** Games from 8 weeks ago count half as much. Captures "hot" teams entering playoffs.

---

## Prior (Preseason Expectations)

Incorporate preseason rankings.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `enable_prior` | false | true/false | Enable preseason prior |
| `prior_weight` | 0.0 | 0.0-0.4 | How much prior influences final rating |
| `prior_decay_weeks` | 8 | 4-16 | Weeks for prior influence to halve |

**Effect:** Allows preseason expectations to influence early-season rankings, fading over time.

---

## Tiebreakers

How ties are resolved when ratings are equal.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `tie_threshold` | 0.001 | 0.0001-0.01 | Rating difference to consider a tie |
| `tiebreaker_order` | ["h2h", "sov", "common", "away"] | any order | Order of tiebreaker criteria |

**Tiebreaker options:**
- `h2h` - Head-to-head result
- `sov` - Strength of victory (average rating of teams beaten)
- `common` - Common opponent margin
- `away` - Away win percentage

---

## Filters

What games and teams to include.

| Lever | Default | Range | Description |
|-------|---------|-------|-------------|
| `include_postseason` | true | true/false | Include bowl/playoff games |
| `postseason_weight` | 1.0 | 0.5-2.0 | Weight multiplier for postseason games |
| `include_fcs_games` | true | true/false | Include FCS games in calculation |
| `exclude_fcs_from_rankings` | false | true/false | Hide FCS teams from output |
| `min_games_to_rank` | 1 | 1-8 | Minimum games played to appear in rankings |

**Effect:**
- `exclude_fcs_from_rankings=true` removes Montana State etc. from top 25
- `min_games_to_rank=5` filters teams with insufficient data

---

## Preset Profiles

| Profile | Description | Key Settings |
|---------|-------------|--------------|
| `pure_results` | Raw algorithm, no adjustments | All defaults |
| `balanced` | Moderate conference/quality adjustments | Conference adj, small quality tiers |
| `predictive` | Optimized for predicting outcomes | Strong conference adj, quality tiers, recency |
| `conservative` | Stable rankings with prior influence | Enables prior, longer recency half-life |

```bash
# View any profile's settings
ncaa-rank config show predictive
```

---

## Example Configurations

### G5-Friendly (Reward Results Over Pedigree)
```json
{
    "enable_conference_adj": false,
    "enable_quality_tiers": true,
    "elite_win_bonus": 0.15,
    "bad_loss_penalty": 0.20,
    "exclude_fcs_from_rankings": true
}
```

### Predictive (Best for Forecasting)
```json
{
    "enable_conference_adj": true,
    "p5_multiplier": 1.05,
    "g5_multiplier": 0.85,
    "enable_quality_tiers": true,
    "elite_win_bonus": 0.10,
    "bad_loss_penalty": 0.15,
    "enable_recency": true,
    "recency_half_life": 6,
    "exclude_fcs_from_rankings": true
}
```

### Strict SOS Requirements
```json
{
    "sos_adjustment_weight": 0.20,
    "min_sos_top_10": 0.50,
    "min_sos_top_25": 0.40,
    "min_p5_games_top_10": 4,
    "min_games_to_rank": 6
}
```
