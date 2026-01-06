"""FastAPI web dashboard for NCAA rankings.

Provides interactive web interface with HTMX for dynamic updates.
"""

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.data.models import AlgorithmConfig, ComparisonResult, RatingExplanation, TeamRating
from src.data.profiles import PROFILES, get_profile, get_profile_description, list_profiles
from src.ranking.comparison import RankingComparator
from src.ranking.engine import RankingEngine

# Get templates directory relative to this file
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(
    title="NCAA Rankings",
    description="Bias-free NCAA FBS football rankings using iterative convergence",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =============================================================================
# Helper Functions (can be mocked in tests)
# =============================================================================


def get_rankings(
    season: int,
    week: int | None = None,
    top: int = 25,
) -> list[TeamRating]:
    """
    Get rankings for a season.

    Args:
        season: Season year
        week: Optional week number
        top: Maximum number of teams to return

    Returns:
        List of TeamRating objects
    """
    engine = RankingEngine()

    async def _get() -> list[TeamRating]:
        if week is not None:
            rankings = await engine.rank_as_of_week(season, week, games=[])
        else:
            rankings = await engine.rank_season(season, games=[])
        return rankings[:top]

    return asyncio.run(_get())


def get_team_detail(
    team_id: str,
    season: int,
    week: int | None = None,
) -> RatingExplanation | None:
    """
    Get rating explanation for a team.

    Args:
        team_id: Team identifier
        season: Season year
        week: Optional week number

    Returns:
        RatingExplanation or None if not found
    """
    engine = RankingEngine()

    async def _get() -> RatingExplanation | None:
        try:
            return await engine.explain_rating(season, team_id, week=week, games=[])
        except Exception:
            return None

    return asyncio.run(_get())


def get_comparison(
    season: int,
    poll: str = "ap",
    week: int | None = None,
) -> ComparisonResult:
    """
    Get comparison to poll.

    Args:
        season: Season year
        poll: Poll type (ap, cfp, coaches)
        week: Optional week number

    Returns:
        ComparisonResult
    """
    engine = RankingEngine()
    comparator = RankingComparator(engine)

    async def _get() -> ComparisonResult:
        return await comparator.compare_to_poll(
            season=season,
            week=week or 15,
            poll_type=poll,
            poll_rankings=[],
            our_rankings=[],
        )

    return asyncio.run(_get())


# =============================================================================
# Routes
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def home() -> RedirectResponse:
    """Redirect to current season rankings."""
    return RedirectResponse(url="/rankings/2024", status_code=302)


@app.get("/rankings/{season}", response_class=HTMLResponse)
async def rankings_page(
    request: Request,
    season: int,
    week: Annotated[int | None, Query()] = None,
) -> HTMLResponse:
    """Rankings page with week selector."""
    rankings = get_rankings(season=season, week=week)
    weeks = list(range(1, 16))  # Weeks 1-15

    return templates.TemplateResponse(
        request,
        "rankings.html",
        {
            "rankings": rankings,
            "season": season,
            "week": week,
            "weeks": weeks,
        },
    )


@app.get("/rankings/{season}/table", response_class=HTMLResponse)
async def rankings_table(
    request: Request,
    season: int,
    week: Annotated[int | None, Query()] = None,
) -> HTMLResponse:
    """HTMX partial - just the rankings table."""
    rankings = get_rankings(season=season, week=week)

    return templates.TemplateResponse(
        request,
        "partials/rankings_table.html",
        {
            "rankings": rankings,
        },
    )


@app.get("/teams/{team_id}/{season}", response_class=HTMLResponse)
async def team_detail_page(
    request: Request,
    team_id: str,
    season: int,
    week: Annotated[int | None, Query()] = None,
) -> HTMLResponse:
    """Team detail with rating explanation."""
    team = get_team_detail(team_id=team_id, season=season, week=week)

    return templates.TemplateResponse(
        request,
        "team_detail.html",
        {
            "team": team,
            "season": season,
        },
    )


@app.get("/compare/{season}", response_class=HTMLResponse)
async def compare_page(
    request: Request,
    season: int,
    poll: Annotated[str, Query()] = "ap",
    week: Annotated[int | None, Query()] = None,
) -> HTMLResponse:
    """Compare algorithm to poll."""
    comparison = get_comparison(season=season, poll=poll, week=week)

    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "comparison": comparison,
            "poll": poll,
        },
    )


@app.get("/compare/{season}/table", response_class=HTMLResponse)
async def compare_table(
    request: Request,
    season: int,
    poll: Annotated[str, Query()] = "ap",
    week: Annotated[int | None, Query()] = None,
) -> HTMLResponse:
    """HTMX partial - comparison table."""
    comparison = get_comparison(season=season, poll=poll, week=week)

    return templates.TemplateResponse(
        request,
        "partials/comparison_table.html",
        {
            "comparison": comparison,
        },
    )


# =============================================================================
# Config Routes
# =============================================================================


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request) -> HTMLResponse:
    """Configuration page with profile selector."""
    profile_info = []
    for name in list_profiles():
        profile_info.append({
            "name": name,
            "description": get_profile_description(name),
        })

    # Get default config for form
    default_config = AlgorithmConfig()

    return templates.TemplateResponse(
        request,
        "config.html",
        {
            "profiles": profile_info,
            "config": default_config.model_dump(),
            "selected_profile": "default",
        },
    )


@app.get("/config/profile/{profile_name}", response_class=HTMLResponse)
async def config_profile(
    request: Request,
    profile_name: str,
) -> HTMLResponse:
    """HTMX partial - config form for a profile."""
    if profile_name == "default":
        config = AlgorithmConfig()
    else:
        try:
            config = get_profile(profile_name)
        except ValueError:
            config = AlgorithmConfig()

    return templates.TemplateResponse(
        request,
        "partials/config_form.html",
        {
            "config": config.model_dump(),
            "selected_profile": profile_name,
        },
    )


@app.get("/api/config/{profile_name}")
async def api_config(profile_name: str) -> dict:
    """API endpoint - get config as JSON."""
    if profile_name == "default":
        config = AlgorithmConfig()
    else:
        try:
            config = get_profile(profile_name)
        except ValueError:
            return {"error": f"Unknown profile: {profile_name}"}

    return config.model_dump()


# =============================================================================
# Run with: uvicorn src.web.app:app --reload
# =============================================================================
