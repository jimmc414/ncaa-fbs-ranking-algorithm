# CLAUDE.md - Implementation Guide for Claude Code

## Project Context

This is a NCAA football ranking algorithm that eliminates human voting bias. The system uses iterative convergence (similar to PageRank) to derive team ratings from game outcomes and opponent quality.

**Primary specification:** See `PROJECT_SPEC.md` for complete algorithm details, data models, and acceptance criteria.

---

## Implementation Priorities

### Order of Implementation

1. **Data models first** (`src/data/models.py`) - All other code depends on these
2. **API client** (`src/data/client.py`) - Need real data to test
3. **Game grade calculation** (`src/algorithm/game_grade.py`) - Core building block
4. **Convergence algorithm** (`src/algorithm/convergence.py`) - The heart of the system
5. **Ranking engine** (`src/ranking/engine.py`) - Orchestration layer
6. **CLI** (`src/cli/main.py`) - User interface
7. **Player rankings** - Optional, implement last

### Critical Implementation Details

#### 1. Quality Losses Hurt Less (CRITICAL)

The key insight: **losing to a good team should hurt less than losing to a bad team**.

```python
# CORRECT - Quality losses hurt less
if is_win:
    contribution = game_grade + opponent_rating
else:
    # Penalty = (1 - opponent_rating) × loss_opponent_factor
    # Loss to 0.9 team: -(1-0.9) = -0.1 (small penalty)
    # Loss to 0.1 team: -(1-0.1) = -0.9 (big penalty)
    opponent_weakness = 1.0 - opponent_rating
    contribution = game_grade - opponent_weakness * loss_opponent_factor

# WRONG (this is the RMI flaw)
contribution = game_grade + opponent_rating  # Losses shouldn't add opponent rating!
```

Losses always hurt, but losing to elite teams is penalized much less than losing to bad teams.

#### 2. Margin Bonus is Logarithmic

```python
import math

def margin_bonus(margin: int, is_win: bool) -> float:
    if not is_win:
        return 0.0
    capped_margin = min(margin, 28)
    return 0.20 * math.log(1 + capped_margin) / math.log(29)
```

NOT linear. The log curve rewards the first touchdown of margin more than the third.

#### 3. Convergence Must Be Deterministic

Process teams in sorted order (by team_id) every iteration. Random ordering can cause non-deterministic results.

```python
def iterate(teams: dict[str, float], games: list[Game]) -> dict[str, float]:
    new_ratings = {}
    for team_id in sorted(teams.keys()):  # MUST sort for determinism
        new_ratings[team_id] = compute_rating(team_id, teams, games)
    return new_ratings
```

#### 4. Normalization Happens AFTER Convergence

Don't normalize during iteration - only normalize final output to [0, 1] range.

```python
def normalize(ratings: dict[str, float]) -> dict[str, float]:
    min_r = min(ratings.values())
    max_r = max(ratings.values())
    return {t: (r - min_r) / (max_r - min_r) for t, r in ratings.items()}
```

---

## Code Style Requirements

### Type Hints Everywhere

```python
# Good
def compute_game_grade(
    is_win: bool,
    margin: int,
    location: Literal["home", "away", "neutral"]
) -> float:
    ...

# Bad
def compute_game_grade(is_win, margin, location):
    ...
```

### Pydantic Models for All Data

```python
from pydantic import BaseModel, Field

class Game(BaseModel):
    game_id: int
    season: int
    home_team_id: str
    away_team_id: str
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    neutral_site: bool = False
```

### Use Polars, Not Pandas

Polars is faster and has better type safety. Use it for all dataframe operations.

```python
import polars as pl

# Good
df = pl.DataFrame(games)
result = df.filter(pl.col("season") == 2024)

# Avoid pandas unless necessary for compatibility
```

### Async for API Calls

```python
import httpx

async def fetch_games(season: int) -> list[Game]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/games", params={"year": season})
        return [Game(**g) for g in response.json()]
```

---

## Testing Requirements

### Every Algorithm Component Needs Unit Tests

```python
# tests/test_game_grade.py

def test_road_blowout_win():
    grade = compute_game_grade(is_win=True, margin=28, location="away")
    assert abs(grade - 1.00) < 0.01

def test_home_loss_penalty():
    grade = compute_game_grade(is_win=False, margin=3, location="home")
    assert grade == -0.03

def test_margin_cap():
    grade_28 = compute_game_grade(is_win=True, margin=28, location="home")
    grade_60 = compute_game_grade(is_win=True, margin=60, location="home")
    assert grade_28 == grade_60  # Margin capped at 28
```

### Integration Test: 4-Team Round Robin

This is the canonical test case. Create fixture data for:
- Team A beats B, C, D
- Team B beats C, D  
- Team C beats D
- Team D loses all

Expected ranking: A > B > C > D with specific rating gaps.

### Regression Test: Compare to Historical Data

Include a test that runs the algorithm on 2023 season data and verifies:
1. Convergence occurs
2. Top 4 teams are reasonable (correlation with CFP)
3. No rating anomalies (all between 0 and 1 after normalization)

---

## API Client Implementation Notes

### Rate Limiting

CollegeFootballData allows 1000 requests/hour. Implement rate limiting:

```python
from asyncio import Semaphore, sleep

class CFBDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._semaphore = Semaphore(10)  # Max concurrent requests
        self._request_times: list[float] = []
    
    async def _rate_limit(self):
        # Ensure no more than 1000 requests per hour
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 3600]
        if len(self._request_times) >= 1000:
            sleep_time = 3600 - (now - self._request_times[0])
            await sleep(sleep_time)
        self._request_times.append(now)
```

### Caching

Cache API responses to disk to avoid redundant calls:

```python
import json
from pathlib import Path

CACHE_DIR = Path("data/cache")

def get_cached(key: str) -> dict | None:
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None

def set_cached(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps(data))
```

### Neutral Site Overrides

Some games are mislabeled. Maintain a CSV of overrides:

```csv
game_id,correct_neutral_site
401234567,true
401234568,false
```

Apply these after fetching from API.

---

## CLI Implementation

Use Typer with Rich for output:

```python
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

@app.command()
def rank(
    season: int = typer.Argument(..., help="Season year (e.g., 2024)"),
    top: int = typer.Option(25, help="Number of teams to display"),
    conference: str | None = typer.Option(None, help="Filter by conference"),
):
    """Generate rankings for a season."""
    # Implementation here
    ...

@app.command()
def explain(
    season: int,
    team: str = typer.Argument(..., help="Team name or ID"),
):
    """Show detailed rating breakdown for a team."""
    ...

if __name__ == "__main__":
    app()
```

---

## Common Pitfalls to Avoid

### 1. Don't Assume Home Team Won

```python
# WRONG
winner = game.home_team_id

# RIGHT
winner = game.home_team_id if game.home_score > game.away_score else game.away_team_id
```

### 2. Handle Ties (They Exist in Historical Data)

Before overtime rules, ties occurred. Treat as 0.35 win equivalent or skip game.

```python
if game.home_score == game.away_score:
    # Handle tie - either skip or assign 0.35 to both
    ...
```

### 3. FCS Teams Need Handling

When FBS team plays FCS team, the FCS team may not have enough data for iteration. Options:
- Assign fixed rating (e.g., 0.20) to all FCS teams
- Include FCS-vs-FCS games in network (more data work)

The config should control this:

```python
if config.fcs_fixed_rating is not None and team.division == "fcs":
    return config.fcs_fixed_rating
```

### 4. Postseason Games

Bowl games and playoffs should be included but flagged. Some analyses may want regular-season-only rankings.

```python
class Game(BaseModel):
    ...
    postseason: bool = False
    
# In ranking engine
if not include_postseason:
    games = [g for g in games if not g.postseason]
```

### 5. Week-by-Week Rankings

The algorithm should support computing rankings as of any week, not just end of season:

```python
def rank_as_of_week(season: int, week: int) -> list[TeamRating]:
    games = [g for g in all_games if g.season == season and g.week <= week]
    return compute_rankings(games)
```

---

## File-by-File Implementation Checklist

### Phase 1: Data Layer

- [ ] `src/data/models.py`
  - [ ] `Team` model
  - [ ] `Game` model  
  - [ ] `GameResult` model (derived view of Game from one team's perspective)
  - [ ] `TeamRating` model
  - [ ] `AlgorithmConfig` model

- [ ] `src/data/client.py`
  - [ ] `CFBDataClient` class
  - [ ] `fetch_teams()` method
  - [ ] `fetch_games()` method
  - [ ] Rate limiting
  - [ ] Error handling with retries

- [ ] `src/data/cache.py`
  - [ ] `get_cached()` function
  - [ ] `set_cached()` function
  - [ ] Cache invalidation by date

- [ ] `src/data/neutral_sites.py`
  - [ ] Load overrides from CSV
  - [ ] Apply overrides to game data

### Phase 2: Algorithm

- [ ] `src/algorithm/game_grade.py`
  - [ ] `compute_margin_bonus()` function
  - [ ] `compute_venue_adjustment()` function
  - [ ] `compute_game_grade()` function

- [ ] `src/algorithm/convergence.py`
  - [ ] `initialize_ratings()` function
  - [ ] `iterate_once()` function
  - [ ] `converge()` function with threshold check
  - [ ] Convergence diagnostics (iteration count, max delta history)

- [ ] `src/algorithm/tiebreaker.py`
  - [ ] `head_to_head()` function
  - [ ] `strength_of_victory()` function
  - [ ] `common_opponents()` function
  - [ ] `resolve_ties()` function

### Phase 3: Ranking Engine

- [ ] `src/ranking/engine.py`
  - [ ] `RankingEngine` class
  - [ ] `rank_season()` method
  - [ ] `rank_as_of_week()` method
  - [ ] `explain_rating()` method

- [ ] `src/ranking/output.py`
  - [ ] `format_table()` function (Rich)
  - [ ] `to_json()` function
  - [ ] `to_csv()` function

### Phase 4: CLI

- [ ] `src/cli/main.py`
  - [ ] `rank` command
  - [ ] `explain` command
  - [ ] `compare` command
  - [ ] `export` command

---

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your CFBD_API_KEY

# Run tests
pytest

# Run linting
ruff check src tests
mypy src
```

---

## Questions to Resolve During Implementation

1. **FCS handling:** Fixed rating or full iteration? Start with fixed (0.20), add full iteration later.

2. **Tie handling:** Skip games or assign partial credit? Recommend skipping for simplicity.

3. **COVID 2020 season:** Include or exclude? Include with caveat in output.

4. **Conference championship games:** Regular season or postseason? Treat as postseason.

5. **Multiple games between same teams:** Sum contributions or average? Sum (each game counts).
