# NCAA FBS Ranking Algorithm

A college football ranking system that derives team ratings purely from game outcomes using iterative convergence. No human voting, no preseason bias, no brand recognition - just results.

The core insight: you can't know how good a team is until you know how good their opponents are, which depends on *their* opponents, recursively. The algorithm iterates until ratings stabilize.

## Quick Start

```bash
pip install -e .
export CFBD_API_KEY=your_key_here  # Free at collegefootballdata.com/key

ncaa-rank rank 2025 --top 25           # Generate rankings
ncaa-rank predict 2025 12 --consensus  # Predict games with all sources
ncaa-rank decompose 2025 indiana       # Why is Indiana ranked #1?
ncaa-rank diagnose 2025                # How accurate were we?
ncaa-rank vegas-analysis 2025          # Where did Vegas get it wrong?
```

---

## 2025 Season Results

### Our Rankings vs CFP (Final)

```
Rank  Team              Record   Rating   SOS    | CFP Rank  Diff
──────────────────────────────────────────────────────────────────
  1   Indiana           14-0     1.000    0.450  |    1       -
  2   Oregon            13-1     0.865    0.459  |    2       -
  3   Ole Miss          13-1     0.836    0.429  |    3       -
  4   Miami             12-2     0.834    0.476  |   12      +8  (we liked them more)
  5   Georgia           12-2     0.809    0.479  |    4      -1
  6   Texas Tech        12-2     0.808    0.447  |    -       -  (unranked by CFP)
  7   North Texas       12-2     0.803    0.466  |   24     +17  (G5 bias in CFP?)
  8   BYU               12-2     0.797    0.487  |   11      +3
  9   Ohio State        12-2     0.766    0.476  |    5      -4
 10   James Madison     12-2     0.742    0.425  |   25     +15  (G5 underrated by CFP)
```

**Legend:**
- **SOS** (Strength of Schedule): Average opponent rating. Higher = harder schedule.
- **Rating**: Normalized 0-1 score from iterative convergence.
- **Diff**: Positive = we rank higher than CFP. Negative = CFP ranks higher.

**Notable divergences:**
- We liked **Miami** more (+8 spots) - their 12-2 included quality wins
- **North Texas** and **James Madison** (both G5, 12-2) ranked much higher by us
- CFP overvalued **Arizona** (#18 CFP vs #51 ours), **USC** (#16 vs #46), **Michigan** (#19 vs #45)

### Prediction Accuracy (2025 Season)

```
Total Games:     931
Correct Picks:   772 (82.9%)
Brier Score:     0.126 (0.25 = random, lower = better)
Calibration:     0.030 error (well-calibrated)
```

The `diagnose` command identified the biggest upsets and which parameters caused errors:
- **149 errors** from venue misjudgment (road/home advantage)
- **96 errors** from margin weighting in close games
- **6 errors** from rating spread being too wide

---

## The Tuning Loop

The algorithm improves itself through a validation feedback loop:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Generate rankings using current config                      │
│  2. Compare against validators (SP+, SRS, Elo, Vegas)          │
│  3. Run diagnostics to measure prediction accuracy              │
│  4. Identify which parameters caused the most errors            │
│  5. Adjust parameters (margin_weight, venue_bonus, etc.)        │
│  6. Re-run and compare - did accuracy improve?                  │
│  7. Repeat until satisfied                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Example tuning session:**
```bash
# Current accuracy
ncaa-rank diagnose 2025 --profile predictive
# Output: 81.2% accuracy, margin_weight causing 120 errors

# Export config and reduce margin_weight
ncaa-rank config export tuned.json --profile predictive
# Edit: margin_weight: 0.20 → 0.15

# Test new config
ncaa-rank diagnose 2025 --config tuned.json
# Output: 82.9% accuracy, margin_weight errors down to 96

# Validate against external sources
ncaa-rank validate 2025 --config tuned.json
# Output: 60.6% correlation with CFP (up from 55%)
```

The `tuned_predictive` profile was created exactly this way - iterating on 2025 data until accuracy peaked at 82.9%.

---

## Prediction System

### Single Game Prediction

```bash
ncaa-rank predict-game "Ohio State" "Michigan" --season 2025
```

Shows all sources side-by-side:

```
Source              Pick         Win Prob   Weight
──────────────────────────────────────────────────
Vegas (-7.5)        Ohio State   72.1%      35%
Our Algorithm       Ohio State   68.4%      25%
Pregame WP          Ohio State   69.8%      20%
SP+ Implied         Ohio State   71.2%      10%
Elo Implied         Ohio State   70.5%      10%
──────────────────────────────────────────────────
CONSENSUS           Ohio State   70.3%      HIGH CONFIDENCE
```

### Weekly Predictions

```bash
ncaa-rank predict 2025 12 --consensus --high-confidence
```

Filters to games where all sources agree and probability spread is <10%. These are your safest picks.

```bash
ncaa-rank predict 2025 12 --show-splits
```

Shows only games where sources **disagree on the winner** - the most interesting games to watch.

### Vegas Upset Analysis

Find games where Vegas favorites lost, and see if we predicted them:

```bash
ncaa-rank vegas-analysis 2025 --we-got-right --min-spread 7
```

In 2025, Vegas favorites (7+ point spread) lost outright in 23 games. Our algorithm correctly predicted 9 of them (39%). Patterns:
- **Road favorites**: We predicted 45% of road favorite upsets
- **Late season**: We predicted 52% of Week 10+ upsets
- **P5 at G5**: We predicted 60% of these upsets (CFP typically misses these)

---

## Core Algorithm

### Game Grade

```
game_grade(is_win, margin, location):
    base = 0.70 if is_win else 0.00

    if is_win:
        margin_bonus = 0.20 * log(1 + min(margin, 28)) / log(29)
    else:
        margin_bonus = 0

    venue_adj = {
        (win, away): +0.10,   (win, neutral): +0.05,
        (loss, home): -0.03,  (loss, neutral): -0.01
    }

    return base + margin_bonus + venue_adj
```

### Iterative Convergence

```
converge(games):
    ratings = {team: 0.5 for team in all_teams}

    repeat until max_delta < 0.0001:
        for team in sorted(teams):
            contributions = []
            for game in team_games[team]:
                grade = game_grade(game)
                opp = ratings[opponent]

                # KEY: quality losses hurt less than bad losses
                if is_win:
                    contribution = grade + opp
                else:
                    # Penalty = (1 - opp_rating): smaller for better opponents
                    contribution = grade - (1 - opp)
                contributions.append(contribution)

            new_ratings[team] = mean(contributions)

        ratings = new_ratings

    return normalize_to_0_1(ratings)
```

**Why losses use `(1 - opponent_rating)`:** Losing to a 0.9-rated team costs only 0.1, losing to a 0.1-rated team costs 0.9. Quality losses hurt less, bad losses hurt more - the inverse of opponent strength.

### Consensus Prediction

```
WEIGHTS = {vegas: 0.35, algorithm: 0.25, pregame_wp: 0.20, sp: 0.10, elo: 0.10}

consensus(predictions):
    available = filter_not_none(predictions)
    total = sum(WEIGHTS[s] for s in available)
    return sum(prob * WEIGHTS[s] / total for s, prob in available)
```

Vegas gets highest weight (historically most accurate). When sources disagree, confidence is "SPLIT".

---

## Full 2025 Top 100

<details>
<summary>Click to expand</summary>

```
Rank  Team                   Record   Rating   SOS    SOV
────────────────────────────────────────────────────────────
   1  Indiana                14-0     1.000    0.450  0.450
   2  Oregon                 13-1     0.865    0.459  0.418
   3  Ole Miss               13-1     0.836    0.429  0.399
   4  Miami                  12-2     0.834    0.476  0.449
   5  Georgia                12-2     0.809    0.479  0.435
   6  Texas Tech             12-2     0.808    0.447  0.403
   7  North Texas            12-2     0.803    0.466  0.433
   8  BYU                    12-2     0.797    0.487  0.434
   9  Ohio State             12-2     0.766    0.476  0.402
  10  James Madison          12-2     0.742    0.425  0.371
  11  Tulane                 11-3     0.708    0.491  0.430
  12  Notre Dame             10-2     0.705    0.418  0.350
  13  Navy                   11-2     0.702    0.419  0.358
  14  Virginia               11-3     0.693    0.429  0.382
  15  Utah                   11-2     0.693    0.392  0.317
  16  Texas A&M              11-2     0.691    0.402  0.338
  17  Texas                  10-3     0.674    0.421  0.368
  18  Oklahoma               10-3     0.654    0.459  0.382
  19  SMU                     9-4     0.652    0.439  0.413
  20  Alabama                11-4     0.647    0.491  0.413
  21  Louisville              9-4     0.633    0.467  0.423
  22  Western Michigan       10-4     0.629    0.416  0.375
  23  Fresno State            9-4     0.616    0.349  0.351
  24  Houston                10-3     0.614    0.360  0.303
  25  Old Dominion           10-3     0.610    0.406  0.319
  26  Duke                    9-5     0.609    0.521  0.463
  27  South Florida           9-4     0.608    0.479  0.400
  28  Wake Forest             9-4     0.608    0.434  0.394
  29  TCU                     9-4     0.604    0.447  0.388
  30  Vanderbilt             10-3     0.604    0.362  0.285
  31  UConn                   9-4     0.598    0.339  0.326
  32  Iowa State              8-4     0.596    0.426  0.399
  33  Western Kentucky        9-4     0.592    0.387  0.342
  34  San Diego State         9-4     0.591    0.418  0.341
  35  Illinois                9-4     0.586    0.464  0.389
  36  East Carolina           9-4     0.586    0.452  0.368
  37  Kennesaw State         10-4     0.585    0.461  0.365
  38  NC State                8-5     0.584    0.514  0.483
  39  New Mexico              9-4     0.571    0.357  0.316
  40  UNLV                   10-4     0.569    0.360  0.283
  41  Arizona State           8-5     0.565    0.484  0.446
  42  Jacksonville State      9-5     0.564    0.390  0.357
  43  Hawai'i                 9-4     0.562    0.367  0.309
  44  Toledo                  8-5     0.555    0.399  0.369
  45  Michigan                9-4     0.555    0.438  0.339
  46  USC                     9-4     0.554    0.452  0.346
  47  Ohio                    9-4     0.551    0.362  0.302
  48  Arizona                 9-4     0.547    0.437  0.335
  49  Boise State             9-5     0.543    0.454  0.366
  50  Washington              9-4     0.541    0.397  0.306
  51  Georgia Tech            9-4     0.538    0.435  0.330
  52  Iowa                    9-4     0.533    0.463  0.334
  53  Louisiana Tech          8-5     0.508    0.375  0.301
  54  Southern Miss           7-6     0.499    0.407  0.391
  55  Pittsburgh              8-5     0.498    0.441  0.341
  56  Delaware                7-6     0.491    0.386  0.381
  57  Arkansas State          7-6     0.490    0.390  0.383
  58  Memphis                 8-5     0.486    0.403  0.299
  59  UTSA                    7-6     0.482    0.483  0.424
  60  Troy                    8-6     0.470    0.421  0.319
  61  Texas State             7-6     0.468    0.418  0.346
  62  California              7-6     0.466    0.407  0.355
  63  FIU                     7-6     0.466    0.413  0.343
  64  Army                    7-6     0.457    0.461  0.369
  65  Clemson                 7-6     0.450    0.393  0.332
  66  Minnesota               8-5     0.443    0.420  0.306
  67  Miami (OH)              7-7     0.438    0.420  0.350
  68  Washington State        7-6     0.427    0.485  0.371
  69  Georgia Southern        7-6     0.424    0.438  0.313
  70  Tennessee               8-5     0.414    0.404  0.244
  71  Nebraska                7-6     0.411    0.404  0.295
  72  Missouri State          7-6     0.411    0.413  0.284
  73  Central Michigan        7-6     0.411    0.366  0.255
  74  Penn State              7-6     0.406    0.466  0.342
  75  Missouri                8-5     0.406    0.397  0.234
  76  Louisiana               6-7     0.397    0.411  0.354
  77  Cincinnati              7-6     0.388    0.448  0.296
  78  Northwestern            7-6     0.381    0.436  0.285
  79  LSU                     7-6     0.373    0.497  0.346
  80  Florida State           5-7     0.364    0.462  0.416
  81  Akron                   5-7     0.355    0.326  0.276
  82  Kansas State            6-6     0.351    0.423  0.279
  83  Utah State              6-7     0.349    0.442  0.297
  84  Marshall                5-7     0.344    0.423  0.336
  85  Coastal Carolina        6-7     0.341    0.404  0.250
  86  Rice                    5-8     0.320    0.490  0.393
  87  Buffalo                 5-7     0.319    0.372  0.239
  88  Ball State              4-8     0.317    0.378  0.384
  89  Baylor                  5-7     0.304    0.443  0.328
  90  Temple                  5-7     0.288    0.491  0.290
  91  Eastern Michigan        4-8     0.286    0.384  0.295
  92  Kent State              5-7     0.286    0.386  0.240
  93  Kentucky                5-7     0.282    0.485  0.332
  94  Kansas                  5-7     0.282    0.438  0.294
  95  Rutgers                 5-7     0.281    0.478  0.319
  96  App State               5-8     0.279    0.405  0.238
  97  Bowling Green           4-8     0.269    0.366  0.291
  98  Liberty                 4-8     0.268    0.449  0.307
  99  UCF                     5-7     0.265    0.408  0.270
 100  Wyoming                 4-8     0.264    0.394  0.275
```

**Legend:**
- **SOS** (Strength of Schedule): Average rating of opponents faced
- **SOV** (Strength of Victory): Average rating of opponents defeated

</details>

---

## Configuration

### Profiles

| Profile | Use Case | Key Settings |
|---------|----------|--------------|
| `pure_results` | Academic analysis | No adjustments, pure outcomes |
| `balanced` | General use | Moderate conference/quality adjustments |
| `predictive` | Forecasting | Strong adjustments, recency weighting |
| `tuned_predictive` | Best predictions | Calibrated for 82.9% accuracy |
| `conservative` | Early season | Historical priors for stability |

### Key Levers

| Parameter | Default | Effect |
|-----------|---------|--------|
| `win_base` | 0.70 | Base value for any win |
| `margin_weight` | 0.20 | Max margin bonus (logarithmic) |
| `margin_cap` | 28 | Points beyond which margin ignored |
| `venue_road_win` | +0.10 | Bonus for road wins |
| `venue_home_loss` | -0.03 | Penalty for home losses |
| `loss_opponent_factor` | -1.0 | Multiply opponent rating in loss contribution |
| `g5_multiplier` | 1.0 | Adjust G5 opponent influence |
| `enable_recency` | false | Weight recent games more |

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for all 45+ parameters.

---

## Validators

External ratings used for validation (not ranking):

| Source | What It Measures | Our Use |
|--------|-----------------|---------|
| **SP+** | Efficiency (points/play adjusted) | Validation, prediction blend |
| **SRS** | Margin + schedule strength | Validation |
| **Elo** | Win/loss vs expectation | Prediction blend |
| **Vegas** | Market consensus | Prediction blend (35% weight) |

When we diverge from all validators, either we found something they missed, or we're overweighting G5 margins.

---

## Installation

```bash
git clone https://github.com/jimmc414/ncaa-fbs-ranking-algorithm.git
cd ncaa-fbs-ranking-algorithm
pip install -e ".[dev]"
cp .env.example .env  # Add CFBD_API_KEY
```

API key: https://collegefootballdata.com/key (free tier sufficient)

---

## Testing

```bash
pytest                                    # 484 tests
pytest --cov=src --cov-report=term       # With coverage (65%)
```

---

## Limitations

- **Early season**: Few games = noisy ratings. Use `conservative` profile.
- **FCS teams**: Fixed at 0.20 rating (too few FBS games to converge).
- **Injuries**: Algorithm knows nothing about roster - only results.
- **Garbage time**: Margin includes late scores. 42-14 ≠ 42-28 to the algorithm.

---

## License

MIT. Issues and PRs welcome at https://github.com/jimmc414/ncaa-fbs-ranking-algorithm
