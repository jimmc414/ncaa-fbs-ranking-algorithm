"""CLI interface for the NCAA ranking system.

Provides commands for generating rankings, explaining ratings,
comparing to polls, and exporting data.
"""

import asyncio
import csv
import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.data.client import CFBDataClient
from src.data.models import ComparisonResult, RatingExplanation, TeamRating
from src.ranking.comparison import RankingComparator
from src.ranking.engine import RankingEngine

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="ncaa-rank",
    help="NCAA FBS football ranking algorithm - bias-free rankings using iterative convergence.",
)
console = Console()


class ExportFormat(str, Enum):
    """Export file formats."""

    csv = "csv"
    json = "json"


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
        # Fetch games from API
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
        except ValueError as e:
            console.print(f"[red]API Error: {e}[/red]")
            return []
        except Exception as e:
            console.print(f"[red]Failed to fetch games: {e}[/red]")
            return []

        if week is not None:
            rankings = await engine.rank_as_of_week(season, week, games=games)
        else:
            rankings = await engine.rank_season(season, games=games)
        return rankings[:top]

    return asyncio.run(_get())


def get_explanation(
    season: int,
    team_id: str,
    week: int | None = None,
) -> RatingExplanation | None:
    """
    Get rating explanation for a team.

    Args:
        season: Season year
        team_id: Team identifier
        week: Optional week number

    Returns:
        RatingExplanation or None if not found
    """
    engine = RankingEngine()

    async def _get() -> RatingExplanation | None:
        try:
            # Fetch games from API
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            return await engine.explain_rating(season, team_id, week=week, games=games)
        except Exception:
            return None

    return asyncio.run(_get())


def get_comparison(
    season: int,
    poll_type: str = "ap",
    week: int | None = None,
) -> ComparisonResult:
    """
    Get comparison to poll.

    Args:
        season: Season year
        poll_type: Poll type (ap, cfp, coaches)
        week: Optional week number

    Returns:
        ComparisonResult
    """
    engine = RankingEngine()
    comparator = RankingComparator(engine)

    async def _get() -> ComparisonResult:
        # Fetch games and poll rankings from API
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            poll_rankings = await client.fetch_rankings(season, week or 15, poll_type)
        except Exception:
            games = []
            poll_rankings = []

        # Get our rankings
        if games:
            our_rankings = await engine.rank_season(season, games=games)
        else:
            our_rankings = []

        return await comparator.compare_to_poll(
            season=season,
            week=week or 15,
            poll_type=poll_type,
            poll_rankings=poll_rankings,
            our_rankings=our_rankings,
        )

    return asyncio.run(_get())


def get_default_output_path(season: int, format: str, week: int | None = None) -> Path:
    """
    Get default output path for export.

    Args:
        season: Season year
        format: File format (csv or json)
        week: Optional week number

    Returns:
        Path object for output file
    """
    if week is not None:
        return Path(f"rankings_{season}_week{week}.{format}")
    return Path(f"rankings_{season}.{format}")


# =============================================================================
# CLI Commands
# =============================================================================


@app.command()
def rank(
    season: Annotated[int, typer.Argument(help="Season year (e.g., 2024)")],
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week number")] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Number of teams to display")] = 25,
) -> None:
    """Generate rankings for a season."""
    rankings = get_rankings(season=season, week=week, top=top)

    if not rankings:
        console.print("[yellow]No rankings available for the specified parameters.[/yellow]")
        return

    # Build table
    table = Table(title=f"NCAA FBS Rankings - {season}" + (f" Week {week}" if week else ""))
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Team", style="bold")
    table.add_column("Record", justify="center")
    table.add_column("Rating", justify="right", style="green")
    table.add_column("SOS", justify="right")
    table.add_column("SOV", justify="right")

    for r in rankings:
        record = f"{r.wins}-{r.losses}"
        rating = f"{r.rating:.4f}"
        sos = f"{r.strength_of_schedule:.3f}" if r.strength_of_schedule else "-"
        sov = f"{r.strength_of_victory:.3f}" if r.strength_of_victory else "-"

        table.add_row(str(r.rank), r.team_id, record, rating, sos, sov)

    console.print(table)


@app.command()
def explain(
    season: Annotated[int, typer.Argument(help="Season year")],
    team: Annotated[str, typer.Argument(help="Team ID (e.g., ohio-state)")],
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week number")] = None,
) -> None:
    """Show detailed rating breakdown for a team."""
    explanation = get_explanation(season=season, team_id=team, week=week)

    if explanation is None:
        console.print(f"[red]Team '{team}' not found or no data available.[/red]")
        return

    # Header
    console.print()
    console.print(f"[bold cyan]{explanation.team_id.upper()}[/bold cyan]")
    console.print(f"Rank: [bold]#{explanation.rank}[/bold]")
    console.print(f"Rating: [green]{explanation.normalized_rating:.4f}[/green]")
    console.print(f"Convergence: {explanation.iterations_to_converge} iterations")
    console.print()

    # Game-by-game breakdown
    if explanation.games:
        table = Table(title="Game-by-Game Breakdown")
        table.add_column("Opponent", style="bold")
        table.add_column("Result", justify="center")
        table.add_column("Margin", justify="right")
        table.add_column("Location", justify="center")
        table.add_column("Grade", justify="right", style="cyan")
        table.add_column("Opp Rating", justify="right")
        table.add_column("Contribution", justify="right", style="green")

        for game in explanation.games:
            result = "[green]W[/green]" if game["is_win"] else "[red]L[/red]"
            margin = str(abs(game["margin"]))
            location = game["location"][:1].upper()
            grade = f"{game['game_grade']:.3f}"
            opp_rating = f"{game['opponent_rating']:.3f}"
            contribution = f"{game['contribution']:.3f}"

            table.add_row(
                game["opponent_id"],
                result,
                margin,
                location,
                grade,
                opp_rating,
                contribution,
            )

        console.print(table)
    else:
        console.print("[yellow]No games found.[/yellow]")


@app.command()
def compare(
    season: Annotated[int, typer.Argument(help="Season year")],
    poll: Annotated[str, typer.Option("--poll", "-p", help="Poll type (ap, cfp, coaches)")] = "ap",
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week number")] = None,
) -> None:
    """Compare algorithm rankings to AP/CFP poll."""
    result = get_comparison(season=season, poll_type=poll, week=week)

    # Header
    console.print()
    console.print(f"[bold cyan]Comparison to {poll.upper()} Poll[/bold cyan]")
    console.print(f"Season: {result.season}, Week: {result.week}")
    console.print()

    # Correlation
    correlation_pct = result.spearman_correlation * 100
    color = "green" if correlation_pct > 80 else "yellow" if correlation_pct > 60 else "red"
    console.print(f"Spearman Correlation: [{color}]{correlation_pct:.1f}%[/{color}]")
    console.print(f"Teams Compared: {result.teams_compared}")
    console.print()

    # Overranked teams (we rank higher than poll)
    if result.overranked:
        console.print("[bold]Overranked (we rank higher than poll):[/bold]")
        table = Table()
        table.add_column("Team", style="bold")
        table.add_column("Our Rank", justify="right", style="green")
        table.add_column("Poll Rank", justify="right", style="yellow")
        table.add_column("Diff", justify="right", style="cyan")

        for team in result.overranked[:5]:
            table.add_row(
                team["team_id"],
                str(team["our_rank"]),
                str(team["poll_rank"]),
                f"+{team['difference']}",
            )
        console.print(table)
        console.print()

    # Underranked teams (we rank lower than poll)
    if result.underranked:
        console.print("[bold]Underranked (we rank lower than poll):[/bold]")
        table = Table()
        table.add_column("Team", style="bold")
        table.add_column("Our Rank", justify="right", style="yellow")
        table.add_column("Poll Rank", justify="right", style="green")
        table.add_column("Diff", justify="right", style="cyan")

        for team in result.underranked[:5]:
            table.add_row(
                team["team_id"],
                str(team["our_rank"]),
                str(team["poll_rank"]),
                str(team["difference"]),
            )
        console.print(table)


@app.command()
def export(
    season: Annotated[int, typer.Argument(help="Season year")],
    format: Annotated[
        ExportFormat, typer.Option("--format", "-f", help="Export format")
    ] = ExportFormat.csv,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path")] = None,
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week number")] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Number of teams to export")] = 25,
) -> None:
    """Export rankings to CSV or JSON file."""
    rankings = get_rankings(season=season, week=week, top=top)

    if not rankings:
        console.print("[yellow]No rankings available to export.[/yellow]")
        return

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = get_default_output_path(season, format.value, week)

    # Export based on format
    if format == ExportFormat.csv:
        _export_csv(rankings, output_path)
    elif format == ExportFormat.json:
        _export_json(rankings, output_path)

    console.print(f"[green]Exported {len(rankings)} teams to {output_path}[/green]")


def _export_csv(rankings: list[TeamRating], path: Path) -> None:
    """Export rankings to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "team_id", "rating", "wins", "losses", "games_played", "sos", "sov"]
        )

        for r in rankings:
            writer.writerow(
                [
                    r.rank,
                    r.team_id,
                    f"{r.rating:.6f}",
                    r.wins,
                    r.losses,
                    r.games_played,
                    f"{r.strength_of_schedule:.6f}" if r.strength_of_schedule else "",
                    f"{r.strength_of_victory:.6f}" if r.strength_of_victory else "",
                ]
            )


def _export_json(rankings: list[TeamRating], path: Path) -> None:
    """Export rankings to JSON file."""
    data = [
        {
            "rank": r.rank,
            "team_id": r.team_id,
            "rating": r.rating,
            "wins": r.wins,
            "losses": r.losses,
            "games_played": r.games_played,
            "strength_of_schedule": r.strength_of_schedule,
            "strength_of_victory": r.strength_of_victory,
        }
        for r in rankings
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    app()
