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

from src.algorithm.diagnostics import (
    DiagnosticReport,
    Prediction,
    decompose_team_rating,
    generate_diagnostic_report,
    predict_game,
)
from src.validation.validators import ValidatorService
from src.validation.consensus import (
    build_game_prediction,
    organize_by_confidence,
    CONSENSUS_WEIGHTS,
)
from src.algorithm.convergence import converge, normalize_ratings
from src.data.client import CFBDataClient
from src.data.models import AlgorithmConfig, ComparisonResult, RatingExplanation, TeamRating
from src.data.profiles import PROFILES, get_profile, get_profile_description, list_profiles
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
    config: AlgorithmConfig | None = None,
) -> list[TeamRating]:
    """
    Get rankings for a season.

    Args:
        season: Season year
        week: Optional week number
        top: Maximum number of teams to return
        config: Optional AlgorithmConfig

    Returns:
        List of TeamRating objects
    """
    engine = RankingEngine(config=config)

    async def _get() -> list[TeamRating]:
        # Fetch games and teams from API
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            teams = await client.fetch_teams(fbs_only=False)  # Include FCS for identification
        except ValueError as e:
            console.print(f"[red]API Error: {e}[/red]")
            return []
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return []

        if week is not None:
            rankings = await engine.rank_as_of_week(season, week, games=games)
        else:
            rankings = await engine.rank_season(season, games=games, teams=teams)
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


def _load_config(
    profile: str | None = None,
    config_file: str | None = None,
) -> AlgorithmConfig:
    """
    Load configuration from profile or file.

    Args:
        profile: Profile name (pure_results, balanced, predictive, conservative)
        config_file: Path to JSON config file

    Returns:
        AlgorithmConfig instance
    """
    if config_file:
        try:
            with open(config_file) as f:
                data = json.load(f)
            return AlgorithmConfig(**data)
        except FileNotFoundError:
            console.print(f"[red]Config file not found: {config_file}[/red]")
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in config file: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise typer.Exit(1)

    if profile:
        try:
            return get_profile(profile)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    return AlgorithmConfig()


@app.command()
def rank(
    season: Annotated[int, typer.Argument(help="Season year (e.g., 2024)")],
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week number")] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Number of teams to display")] = 25,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile (pure_results, balanced, predictive, conservative)"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
    exclude_fcs: Annotated[
        bool,
        typer.Option("--exclude-fcs", help="Exclude FCS teams from rankings output"),
    ] = False,
) -> None:
    """Generate rankings for a season."""
    config = _load_config(profile=profile, config_file=config_file)

    # Override exclude_fcs if specified on command line
    if exclude_fcs:
        config = AlgorithmConfig(**{**config.model_dump(), "exclude_fcs_from_rankings": True})
    rankings = get_rankings(season=season, week=week, top=top, config=config)

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


# =============================================================================
# Diagnostic Commands
# =============================================================================


@app.command()
def diagnose(
    season: Annotated[int, typer.Argument(help="Season year (e.g., 2024)")],
    week: Annotated[int | None, typer.Option("--week", "-w", help="Specific week to analyze")] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
    show_upsets: Annotated[
        int,
        typer.Option("--upsets", "-u", help="Number of upsets to show"),
    ] = 5,
    show_attributions: Annotated[
        int,
        typer.Option("--attributions", "-a", help="Number of parameter attributions to show"),
    ] = 5,
) -> None:
    """Analyze prediction accuracy and identify calibration issues."""
    config = _load_config(profile=profile, config_file=config_file)

    async def _run() -> DiagnosticReport:
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None

        return generate_diagnostic_report(games, config, week=week)

    report = asyncio.run(_run())

    if report is None:
        return

    # Header
    console.print()
    title = f"[bold cyan]Diagnostic Report - {season}"
    if week:
        title += f" Week {week}"
    title += "[/bold cyan]"
    console.print(title)
    console.print()

    # Summary stats
    console.print("[bold]Prediction Accuracy[/bold]")
    accuracy_color = "green" if report.accuracy >= 0.70 else "yellow" if report.accuracy >= 0.60 else "red"
    console.print(f"  Total Games: {report.total_games}")
    console.print(f"  Correct: {report.correct_predictions}")
    console.print(f"  Wrong: {report.wrong_predictions}")
    console.print(f"  Accuracy: [{accuracy_color}]{report.accuracy:.1%}[/{accuracy_color}]")
    console.print()

    # Calibration metrics
    console.print("[bold]Calibration Metrics[/bold]")
    brier_color = "green" if report.brier_score < 0.20 else "yellow" if report.brier_score < 0.25 else "red"
    cal_color = "green" if report.calibration_error < 0.08 else "yellow" if report.calibration_error < 0.12 else "red"
    console.print(f"  Brier Score: [{brier_color}]{report.brier_score:.4f}[/{brier_color}] (lower is better, 0.25 = random)")
    console.print(f"  Calibration Error: [{cal_color}]{report.calibration_error:.4f}[/{cal_color}] (lower is better)")
    console.print()

    # Upsets
    if report.upsets and show_upsets > 0:
        console.print(f"[bold]Top {min(show_upsets, len(report.upsets))} Upsets[/bold]")
        table = Table()
        table.add_column("Game", style="bold")
        table.add_column("Predicted", justify="center")
        table.add_column("Actual", justify="center", style="green")
        table.add_column("Win Prob", justify="right")
        table.add_column("Magnitude", justify="right", style="yellow")

        for upset in report.upsets[:show_upsets]:
            game_str = f"{upset.away_team_id} @ {upset.home_team_id}"
            magnitude = f"{upset.upset_magnitude:.1%}" if upset.upset_magnitude else "-"
            table.add_row(
                game_str,
                upset.predicted_winner,
                upset.actual_winner,
                f"{upset.win_probability:.1%}",
                magnitude,
            )
        console.print(table)
        console.print()

    # Parameter attributions
    if report.parameter_attributions and show_attributions > 0:
        console.print(f"[bold]Parameter Attribution (which levers caused errors)[/bold]")
        table = Table()
        table.add_column("Parameter", style="bold")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Pattern", style="dim")
        table.add_column("Suggestion", style="cyan")

        for attr in report.parameter_attributions[:show_attributions]:
            table.add_row(
                attr.parameter,
                str(attr.error_count),
                attr.pattern_description,
                attr.suggested_adjustment,
            )
        console.print(table)
        console.print()

    # Suggestions
    if report.suggestions:
        console.print("[bold]Suggested Adjustments[/bold]")
        for suggestion in report.suggestions:
            console.print(f"  • {suggestion}")
        console.print()


@app.command()
def predict(
    season: Annotated[int, typer.Argument(help="Season year")],
    week: Annotated[int, typer.Argument(help="Week to predict")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
    min_confidence: Annotated[
        float,
        typer.Option("--min-confidence", help="Minimum win probability to show"),
    ] = 0.0,
    consensus: Annotated[
        bool,
        typer.Option("--consensus", help="Show consensus view with all sources (Vegas, SP+, Elo)"),
    ] = False,
    high_confidence_only: Annotated[
        bool,
        typer.Option("--high-confidence", help="Only show high confidence predictions"),
    ] = False,
    show_splits: Annotated[
        bool,
        typer.Option("--show-splits", help="Only show games where sources disagree"),
    ] = False,
) -> None:
    """Predict outcomes for upcoming games in a specific week."""
    config = _load_config(profile=profile, config_file=config_file)

    # Use consensus mode if requested
    if consensus or high_confidence_only or show_splits:
        _predict_consensus(
            season=season,
            week=week,
            config=config,
            high_confidence_only=high_confidence_only,
            show_splits=show_splits,
        )
        return

    async def _run() -> tuple[list[Prediction], dict[str, float]] | None:
        try:
            client = CFBDataClient.from_env()
            # Get all games to build ratings
            games = await client.fetch_games(season)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None

        # Build ratings from games before the target week
        prior_games = [g for g in games if g.week < week]
        if not prior_games:
            console.print(f"[yellow]No games before week {week} to build ratings from.[/yellow]")
            return None

        result = converge(prior_games, config)
        normalized = normalize_ratings(result.ratings)

        # Get games for the target week
        week_games = [g for g in games if g.week == week]
        if not week_games:
            console.print(f"[yellow]No games found for week {week}.[/yellow]")
            return None

        # Generate predictions
        predictions = []
        for game in week_games:
            pred = predict_game(
                game.home_team_id,
                game.away_team_id,
                normalized,
                config,
                neutral_site=game.neutral_site,
            )
            # Update with actual game info
            pred.game_id = game.game_id
            pred.season = game.season
            pred.week = game.week

            # Check if game already played
            if game.home_score > 0 or game.away_score > 0:
                if game.home_score > game.away_score:
                    pred.actual_winner = game.home_team_id
                elif game.away_score > game.home_score:
                    pred.actual_winner = game.away_team_id
                pred.was_correct = pred.predicted_winner == pred.actual_winner

            predictions.append(pred)

        return predictions, normalized

    result = asyncio.run(_run())

    if result is None:
        return

    predictions, ratings = result

    # Filter by confidence
    if min_confidence > 0:
        predictions = [p for p in predictions if p.win_probability >= min_confidence]

    # Sort by confidence descending
    predictions.sort(key=lambda p: p.win_probability, reverse=True)

    # Display
    console.print()
    console.print(f"[bold cyan]Predictions - {season} Week {week}[/bold cyan]")
    console.print()

    table = Table()
    table.add_column("Matchup", style="bold")
    table.add_column("Predicted Winner", justify="center", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Rating Gap", justify="right", style="dim")
    table.add_column("Result", justify="center")

    for pred in predictions:
        matchup = f"{pred.away_team_id} @ {pred.home_team_id}"

        # Confidence coloring
        conf = pred.win_probability
        if conf >= 0.85:
            conf_str = f"[green]{conf:.1%}[/green]"
        elif conf >= 0.70:
            conf_str = f"[yellow]{conf:.1%}[/yellow]"
        else:
            conf_str = f"[red]{conf:.1%}[/red]"

        gap_str = f"{pred.rating_gap:+.3f}"

        # Result column
        if pred.was_correct is None:
            result_str = "-"
        elif pred.was_correct:
            result_str = "[green]✓[/green]"
        else:
            result_str = f"[red]✗ ({pred.actual_winner})[/red]"

        table.add_row(
            matchup,
            pred.predicted_winner,
            conf_str,
            gap_str,
            result_str,
        )

    console.print(table)

    # Summary if games already played
    completed = [p for p in predictions if p.was_correct is not None]
    if completed:
        correct = sum(1 for p in completed if p.was_correct)
        console.print()
        console.print(f"[dim]Results: {correct}/{len(completed)} correct ({correct/len(completed):.1%})[/dim]")


def _predict_consensus(
    season: int,
    week: int,
    config: AlgorithmConfig,
    high_confidence_only: bool = False,
    show_splits: bool = False,
) -> None:
    """Display consensus predictions with all sources."""
    from src.validation.models import GamePrediction

    async def _run():
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            teams = await client.fetch_teams(fbs_only=False)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None

        # Build ratings from games before the target week
        prior_games = [g for g in games if g.week < week]
        if not prior_games:
            console.print(f"[yellow]No games before week {week} to build ratings from.[/yellow]")
            return None

        result = converge(prior_games, config)
        normalized = normalize_ratings(result.ratings)

        # Get games for the target week
        week_games = [g for g in games if g.week == week]
        if not week_games:
            console.print(f"[yellow]No games found for week {week}.[/yellow]")
            return None

        # Fetch validators and betting data
        validator_service = ValidatorService(client)
        validators = await validator_service.fetch_all_validators(season)
        betting_lines = await client.fetch_betting_lines(season, week)
        pregame_wp_data = await client.fetch_pregame_wp(season, week)

        # Build lookups
        sp_lookup = {r.team_id: r for r in validators.get("sp", [])}
        elo_lookup = {r.team_id: r for r in validators.get("elo", [])}

        lines_by_game = {}
        for line in betting_lines:
            game_id = line.get("game_id") or line.get("id")
            if game_id:
                lines_by_game[game_id] = line

        wp_by_game = {}
        for wp in pregame_wp_data:
            game_id = wp.get("game_id") or wp.get("id")
            if game_id:
                wp_by_game[game_id] = wp

        # Build predictions for each game
        predictions = []
        for game in week_games:
            home_rating = normalized.get(game.home_team_id)
            away_rating = normalized.get(game.away_team_id)

            # Get Vegas line
            line_data = lines_by_game.get(game.game_id, {})
            vegas_spread = line_data.get("spread")

            # Get pregame WP
            wp_data = wp_by_game.get(game.game_id, {})
            pregame_wp = wp_data.get("home_win_prob")

            # Get validator ratings
            home_sp = sp_lookup.get(game.home_team_id)
            away_sp = sp_lookup.get(game.away_team_id)
            home_elo = elo_lookup.get(game.home_team_id)
            away_elo = elo_lookup.get(game.away_team_id)

            pred = build_game_prediction(
                home_team_id=game.home_team_id,
                away_team_id=game.away_team_id,
                home_rating=home_rating,
                away_rating=away_rating,
                vegas_spread=vegas_spread,
                pregame_wp=pregame_wp,
                home_sp_rating=home_sp.rating if home_sp else None,
                away_sp_rating=away_sp.rating if away_sp else None,
                home_elo=home_elo.rating if home_elo else None,
                away_elo=away_elo.rating if away_elo else None,
                game_id=game.game_id,
                season=season,
                week=week,
                neutral_site=game.neutral_site,
            )

            # Check if game already played
            if game.home_score > 0 or game.away_score > 0:
                if game.home_score > game.away_score:
                    pred.actual_winner = game.home_team_id
                elif game.away_score > game.home_score:
                    pred.actual_winner = game.away_team_id
                pred.was_correct = pred.consensus_winner == pred.actual_winner

            predictions.append(pred)

        return predictions

    predictions = asyncio.run(_run())

    if predictions is None:
        return

    # Organize by confidence
    organized = organize_by_confidence(predictions)

    # Display header
    console.print()
    console.print(f"[bold cyan]PREDICTIONS: {season} Week {week} (Consensus View)[/bold cyan]")
    console.print("=" * 78)
    console.print()

    def display_bucket(games: list, title: str, style: str) -> None:
        if not games:
            return
        console.print(f"[bold {style}]{title}[/bold {style}]")
        console.print("-" * 78)

        for pred in games:
            matchup = f"{pred.away_team_id} @ {pred.home_team_id}"
            winner = pred.consensus_winner or "?"
            prob = pred.consensus_prob or 0.5

            # Get Vegas spread if available
            vegas_source = pred.get_source("vegas")
            spread_str = ""
            if vegas_source and vegas_source.available and vegas_source.spread:
                spread_str = f"Vegas {vegas_source.spread:+.1f}"

            # Result indicator
            if pred.was_correct is None:
                result_str = ""
            elif pred.was_correct:
                result_str = " [green]✓[/green]"
            else:
                result_str = f" [red]✗ ({pred.actual_winner})[/red]"

            console.print(
                f"  {matchup:40} [bold]{winner.upper():15}[/bold] "
                f"{prob:.1%}  {spread_str}{result_str}"
            )

        console.print()

    # Display by confidence level
    if not show_splits:
        if not high_confidence_only:
            display_bucket(organized.high_confidence, "★★★ HIGH CONFIDENCE", "green")
            display_bucket(organized.moderate_confidence, "★★ MODERATE CONFIDENCE", "yellow")
            display_bucket(organized.low_confidence, "★ LOW CONFIDENCE", "dim")
        else:
            display_bucket(organized.high_confidence, "★★★ HIGH CONFIDENCE", "green")
            if not organized.high_confidence:
                console.print("[yellow]No high confidence predictions this week.[/yellow]")

    # Always show splits if requested or in full consensus mode
    if show_splits or (not high_confidence_only):
        if organized.split:
            console.print("[bold red]⚠️ SOURCES DISAGREE[/bold red]")
            console.print("-" * 78)

            for pred in organized.split:
                matchup = f"{pred.away_team_id} @ {pred.home_team_id}"
                console.print(f"  {matchup}")

                # Show each source's pick
                source_picks = []
                for source_pred in pred.source_predictions:
                    if source_pred.available and source_pred.predicted_winner:
                        source_name = source_pred.source.replace("_", " ").title()
                        pick = source_pred.predicted_winner
                        prob = source_pred.win_probability or 0.5
                        source_picks.append(f"{source_name}: {pick} {prob:.0%}")

                console.print(f"    {' | '.join(source_picks)}")
                console.print()

    # Summary
    console.print("=" * 78)
    total = len(predictions)
    high = len(organized.high_confidence)
    moderate = len(organized.moderate_confidence)
    low = len(organized.low_confidence)
    split = len(organized.split)

    console.print(
        f"[dim]Summary: {high} high confidence, {moderate} moderate, "
        f"{low} low, {split} split ({total} total)[/dim]"
    )

    # Accuracy if games played
    completed = [p for p in predictions if p.was_correct is not None]
    if completed:
        correct = sum(1 for p in completed if p.was_correct)
        console.print(f"[dim]Accuracy: {correct}/{len(completed)} ({correct/len(completed):.1%})[/dim]")


@app.command()
def decompose(
    season: Annotated[int, typer.Argument(help="Season year")],
    team: Annotated[str, typer.Argument(help="Team ID (e.g., ohio-state)")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
) -> None:
    """Show detailed contribution breakdown for each game (diagnostic view)."""
    config = _load_config(profile=profile, config_file=config_file)

    async def _run():
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None

        return decompose_team_rating(team, games, config)

    breakdowns = asyncio.run(_run())

    if breakdowns is None:
        return

    if not breakdowns:
        console.print(f"[yellow]No games found for team '{team}'.[/yellow]")
        return

    console.print()
    console.print(f"[bold cyan]Rating Decomposition: {team.upper()}[/bold cyan]")
    console.print()

    # Summary stats
    total_contribution = sum(b.total_contribution for b in breakdowns)
    avg_contribution = total_contribution / len(breakdowns)
    wins = sum(1 for b in breakdowns if b.is_win)
    losses = len(breakdowns) - wins

    console.print(f"[bold]Summary[/bold]")
    console.print(f"  Record: {wins}-{losses}")
    console.print(f"  Total Contribution: {total_contribution:.3f}")
    console.print(f"  Average per Game: {avg_contribution:.3f}")
    console.print()

    # Detailed breakdown table
    table = Table(title="Game-by-Game Contribution Decomposition")
    table.add_column("Opponent", style="bold")
    table.add_column("Result", justify="center")
    table.add_column("Base", justify="right")
    table.add_column("Margin", justify="right")
    table.add_column("Venue", justify="right")
    table.add_column("= Grade", justify="right", style="cyan")
    table.add_column("Opp Rtg", justify="right")
    table.add_column("Opp Contrib", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("= Total", justify="right", style="green")

    for b in breakdowns:
        result = "[green]W[/green]" if b.is_win else "[red]L[/red]"
        loc = b.location[0].upper()
        margin_str = f"+{abs(b.margin)}" if b.is_win else f"-{abs(b.margin)}"

        table.add_row(
            b.opponent_id,
            f"{result} {margin_str} ({loc})",
            f"{b.base_result:.2f}",
            f"{b.margin_bonus:.3f}",
            f"{b.venue_adjustment:+.2f}",
            f"{b.game_grade:.3f}",
            f"{b.opponent_rating:.3f}",
            f"{b.opponent_contribution:+.3f}",
            f"{b.quality_tier_adjustment:+.2f}" if b.quality_tier_adjustment != 0 else "-",
            f"{b.total_contribution:.3f}",
        )

    console.print(table)

    # Component summary
    console.print()
    console.print("[bold]Component Totals[/bold]")
    console.print(f"  Base Results:    {sum(b.base_result for b in breakdowns):.3f}")
    console.print(f"  Margin Bonuses:  {sum(b.margin_bonus for b in breakdowns):.3f}")
    console.print(f"  Venue Adj:       {sum(b.venue_adjustment for b in breakdowns):+.3f}")
    console.print(f"  Opp Contribution:{sum(b.opponent_contribution for b in breakdowns):+.3f}")
    console.print(f"  Quality Tier:    {sum(b.quality_tier_adjustment for b in breakdowns):+.3f}")


# =============================================================================
# Validation Commands
# =============================================================================


@app.command()
def validate(
    season: Annotated[int, typer.Argument(help="Season year (e.g., 2024)")],
    week: Annotated[int | None, typer.Option("--week", "-w", help="Specific week")] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
    threshold: Annotated[
        int,
        typer.Option("--threshold", "-t", help="Rank gap threshold to flag teams"),
    ] = 20,
    team: Annotated[
        str | None,
        typer.Option("--team", help="Show detailed validation for a specific team"),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Compare against specific source (sp, srs, elo)"),
    ] = None,
) -> None:
    """Compare our rankings against external validators (SP+, SRS, Elo)."""
    config = _load_config(profile=profile, config_file=config_file)

    async def _run():
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            teams = await client.fetch_teams(fbs_only=False)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None, None, None

        # Get our rankings
        engine = RankingEngine(config=config)
        if week:
            rankings = await engine.rank_as_of_week(season, week, games=games)
        else:
            rankings = await engine.rank_season(season, games=games, teams=teams)

        # Fetch validators
        validator_service = ValidatorService(client)
        validators = await validator_service.fetch_all_validators(season)

        return rankings, validators, validator_service

    result = asyncio.run(_run())

    if result[0] is None:
        return

    rankings, validators, validator_service = result

    if not rankings:
        console.print("[yellow]No rankings available.[/yellow]")
        return

    if not validators:
        console.print("[yellow]No validators available.[/yellow]")
        return

    # Single team validation
    if team:
        team_validation = validator_service.get_team_validation(team, rankings, validators)
        if team_validation is None:
            console.print(f"[red]Team '{team}' not found.[/red]")
            return

        _display_team_validation(team_validation)
        return

    # Full validation report
    report = validator_service.compare_rankings(rankings, validators, threshold=threshold)

    # Filter to single source if requested
    if source and source in validators:
        validators = {source: validators[source]}

    _display_validation_report(report, source)


def _display_team_validation(validation) -> None:
    """Display detailed validation for a single team."""
    console.print()
    console.print(f"[bold cyan]TEAM VALIDATION: {validation.team_id.upper()}[/bold cyan]")
    console.print("=" * 70)
    console.print()

    table = Table()
    table.add_column("Source", style="bold")
    table.add_column("Rank", justify="right")
    table.add_column("Rating", justify="right")
    table.add_column("Gap vs Ours", justify="right")

    # Our ranking first
    table.add_row(
        "Our Algorithm",
        f"#{validation.our_rank}",
        f"{validation.our_rating:.3f}",
        "--",
    )

    # Each validator
    for source in ["sp", "srs", "elo"]:
        if source in validation.validator_ranks:
            rank = validation.validator_ranks[source]
            rating = validation.validator_ratings.get(source, 0)
            gap = validation.our_rank - rank

            # Color code gap
            if abs(gap) >= 20:
                gap_str = f"[red]{gap:+d}[/red]"
            elif abs(gap) >= 10:
                gap_str = f"[yellow]{gap:+d}[/yellow]"
            else:
                gap_str = f"[green]{gap:+d}[/green]"

            # Format rating based on source
            if source == "elo":
                rating_str = f"{rating:.0f}"
            else:
                rating_str = f"{rating:.1f}"

            table.add_row(
                source.upper(),
                f"#{rank}",
                rating_str,
                gap_str,
            )
        else:
            table.add_row(source.upper(), "N/A", "N/A", "N/A")

    console.print(table)
    console.print()

    # Analysis
    if validation.flagged:
        console.print("[yellow]This team is flagged for investigation.[/yellow]")
        console.print()
        console.print("[bold]Possible causes:[/bold]")
        console.print("  - Weak schedule inflating ranking")
        console.print("  - High margin bonus from blowouts vs weak teams")
        console.print("  - Low efficiency (PPA) masked by score")


def _display_validation_report(report, source_filter: str | None = None) -> None:
    """Display full validation report."""
    console.print()
    console.print(f"[bold cyan]VALIDATION REPORT: {report.season} Season[/bold cyan]")
    console.print("=" * 70)
    console.print()

    # Correlation summary
    console.print("[bold]Overall Correlation[/bold]")
    console.print("-" * 70)

    corr_table = Table(show_header=True)
    corr_table.add_column("Source", style="bold")
    corr_table.add_column("Correlation", justify="right")
    corr_table.add_column("Flagged Teams", justify="right")

    for source in ["sp", "srs", "elo"]:
        if source_filter and source != source_filter:
            continue
        if source in report.correlations:
            corr = report.correlations[source]
            flagged = report.flagged_count_by_source.get(source, 0)

            # Color code correlation
            if corr >= 0.85:
                corr_str = f"[green]{corr:.3f}[/green]"
            elif corr >= 0.70:
                corr_str = f"[yellow]{corr:.3f}[/yellow]"
            else:
                corr_str = f"[red]{corr:.3f}[/red]"

            # Color code flagged count
            if flagged == 0:
                flagged_str = "[green]0[/green]"
            elif flagged <= 3:
                flagged_str = f"[yellow]{flagged} teams[/yellow]"
            else:
                flagged_str = f"[red]{flagged} teams[/red]"

            corr_table.add_row(source.upper(), corr_str, flagged_str)

    console.print(corr_table)
    console.print()

    # Flagged teams
    if report.flagged_teams:
        console.print(f"[bold yellow]FLAGGED TEAMS ({len(report.flagged_teams)} teams with gap >= 20)[/bold yellow]")
        console.print("-" * 70)

        flagged_table = Table()
        flagged_table.add_column("Team", style="bold")
        flagged_table.add_column("Our Rank", justify="right")
        flagged_table.add_column("SP+", justify="right")
        flagged_table.add_column("SRS", justify="right")
        flagged_table.add_column("Elo", justify="right")
        flagged_table.add_column("Max Gap", justify="right", style="yellow")

        for team in report.flagged_teams[:15]:  # Show top 15
            sp_rank = team.validator_ranks.get("sp", "-")
            srs_rank = team.validator_ranks.get("srs", "-")
            elo_rank = team.validator_ranks.get("elo", "-")

            flagged_table.add_row(
                team.team_id,
                f"#{team.our_rank}",
                f"#{sp_rank}" if sp_rank != "-" else "-",
                f"#{srs_rank}" if srs_rank != "-" else "-",
                f"#{elo_rank}" if elo_rank != "-" else "-",
                str(team.max_gap),
            )

        console.print(flagged_table)
        console.print()

    # Patterns
    if report.patterns:
        console.print("[bold]Pattern Analysis[/bold]")
        for pattern in report.patterns:
            console.print(f"  - {pattern}")
        console.print()

    console.print(f"[dim]Use `ncaa-rank validate {report.season} --team <team-id>` for detailed breakdown[/dim]")


# =============================================================================
# Vegas Upset Analysis Commands
# =============================================================================


@app.command("vegas-analysis")
def vegas_analysis(
    season: Annotated[int, typer.Argument(help="Season year (e.g., 2024)")],
    week: Annotated[int | None, typer.Option("--week", "-w", help="Specific week to analyze")] = None,
    min_spread: Annotated[
        float,
        typer.Option("--min-spread", "-m", help="Minimum spread to consider (filters toss-ups)"),
    ] = 3.0,
    we_got_right: Annotated[
        bool,
        typer.Option("--we-got-right", help="Only show upsets we correctly predicted"),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
    export: Annotated[
        str | None,
        typer.Option("--export", "-e", help="Export results to JSON file"),
    ] = None,
) -> None:
    """Analyze games where Vegas got it wrong to find patterns and anomalies."""
    from src.validation.upset_analyzer import UpsetAnalyzer

    config = _load_config(profile=profile, config_file=config_file)

    async def _run():
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            teams = await client.fetch_teams(fbs_only=False)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None

        # Get our rankings
        engine = RankingEngine(config=config)
        if week:
            rankings = await engine.rank_as_of_week(season, week, games=games)
        else:
            rankings = await engine.rank_season(season, games=games, teams=teams)

        # Run upset analysis
        analyzer = UpsetAnalyzer(client)
        report = await analyzer.analyze_season(
            season=season,
            our_ratings=rankings,
            week=week,
            min_spread=min_spread,
        )

        return report

    report = asyncio.run(_run())

    if report is None:
        return

    # Display the report
    _display_vegas_analysis_report(report, we_got_right_only=we_got_right)

    # Export if requested
    if export:
        _export_vegas_analysis(report, export)


def _display_vegas_analysis_report(report, we_got_right_only: bool = False) -> None:
    """Display Vegas upset analysis report."""
    console.print()
    title = f"VEGAS UPSET ANALYSIS: {report.season} Season"
    if report.week:
        title += f" (Week {report.week})"
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("=" * 78)
    console.print()

    # Season Summary
    console.print("[bold]Season Summary[/bold]")
    console.print("-" * 78)

    total_with_lines = report.vegas_correct + report.vegas_wrong
    if total_with_lines > 0:
        vegas_pct = report.vegas_correct / total_with_lines * 100
    else:
        vegas_pct = 0

    console.print(f"  Total FBS Games:        {report.total_games}")
    console.print(f"  Games with Vegas Lines: {total_with_lines}")
    console.print(f"  Vegas Correct (SU):     {report.vegas_correct} ({vegas_pct:.1f}%)")
    console.print(f"  Vegas Wrong (Upsets):   {report.vegas_wrong} ({100-vegas_pct:.1f}%)")
    console.print()

    # Our performance on upsets
    if report.vegas_wrong > 0:
        our_pct = report.we_predicted_upset / report.vegas_wrong * 100
        console.print("[bold]  Our Performance on Upsets:[/bold]")
        console.print(f"  ├─ We Predicted Upset:   {report.we_predicted_upset} ({our_pct:.1f}%) [green]✓[/green]")
        console.print(f"  └─ We Also Wrong:        {report.we_also_wrong} ({100-our_pct:.1f}%)")
        console.print()
        console.print(f"  [bold yellow]★ EDGE:[/bold yellow] On games Vegas got wrong, we were right {our_pct:.1f}%")
    console.print()
    console.print("=" * 78)
    console.print()

    # Pattern Analysis by Spread Bucket
    if report.by_spread_bucket:
        console.print("[bold]Pattern Analysis: By Spread Bucket[/bold]")
        console.print("-" * 78)

        spread_table = Table()
        spread_table.add_column("Spread Range", style="bold", width=18)
        spread_table.add_column("Upsets", justify="right", width=10)
        spread_table.add_column("We Got Right", justify="right", width=15)
        spread_table.add_column("Our Rate", justify="right", width=12)

        for bucket in ["toss-up (<3)", "slight (3-7)", "moderate (7-14)", "heavy (14+)"]:
            stats = report.by_spread_bucket.get(bucket)
            if stats and stats.upsets > 0:
                rate = stats.we_predicted_rate * 100
                rate_str = f"{rate:.1f}%"
                if rate >= 40:
                    rate_str = f"[green]{rate_str} ✓[/green]"
                elif rate >= 25:
                    rate_str = f"[yellow]{rate_str}[/yellow]"

                spread_table.add_row(
                    bucket.title(),
                    str(stats.upsets),
                    str(stats.we_predicted),
                    rate_str,
                )

        console.print(spread_table)
        console.print()

    # Pattern Analysis by Conference Matchup
    if report.by_conference_matchup:
        console.print("[bold]Pattern Analysis: By Conference Matchup[/bold]")
        console.print("-" * 78)

        conf_table = Table()
        conf_table.add_column("Matchup Type", style="bold", width=15)
        conf_table.add_column("Upsets", justify="right", width=10)
        conf_table.add_column("We Got Right", justify="right", width=15)
        conf_table.add_column("Our Rate", justify="right", width=12)

        for matchup, stats in sorted(
            report.by_conference_matchup.items(),
            key=lambda x: x[1].upsets,
            reverse=True,
        ):
            if stats.upsets > 0:
                rate = stats.we_predicted_rate * 100
                rate_str = f"{rate:.1f}%"
                if rate >= 40:
                    rate_str = f"[green]{rate_str} ✓[/green]"
                elif rate >= 25:
                    rate_str = f"[yellow]{rate_str}[/yellow]"

                # Highlight interesting patterns
                matchup_str = matchup
                if matchup == "P5vG5" and stats.upsets >= 5:
                    matchup_str = f"[cyan]{matchup}[/cyan]"

                conf_table.add_row(
                    matchup_str,
                    str(stats.upsets),
                    str(stats.we_predicted),
                    rate_str,
                )

        console.print(conf_table)
        console.print()

    # Anomaly Factors
    if report.anomaly_factors:
        console.print("[bold]Top Anomaly Factors (What Vegas Missed)[/bold]")
        console.print("-" * 78)

        factor_table = Table()
        factor_table.add_column("Factor", style="bold", width=35)
        factor_table.add_column("In Upsets", justify="right", width=12)
        factor_table.add_column("We Predicted", justify="right", width=15)

        for factor in report.anomaly_factors[:7]:  # Top 7
            rate = factor.we_predicted_rate * 100
            rate_str = f"{rate:.1f}%"
            if rate >= 40:
                rate_str = f"[green]{rate_str} ✓[/green]"
            elif rate >= 25:
                rate_str = f"[yellow]{rate_str}[/yellow]"

            factor_table.add_row(
                factor.description[:35],
                str(factor.occurrences),
                rate_str,
            )

        console.print(factor_table)
        console.print()

    console.print("=" * 78)
    console.print()

    # Upsets We Called
    upsets_to_show = report.upsets_we_called if we_got_right_only else report.biggest_upsets
    title = "★ Upsets We Correctly Predicted" if we_got_right_only else "★ Biggest Upsets"

    if upsets_to_show:
        console.print(f"[bold yellow]{title}[/bold yellow]")
        console.print("-" * 78)

        for i, analysis in enumerate(upsets_to_show[:5], 1):
            upset = analysis.upset
            marker = "[green]✓[/green]" if analysis.we_predicted_upset else "[red]✗[/red]"

            console.print(
                f"  {i}. Week {upset.week}: [bold]{upset.actual_winner.upper()}[/bold] "
                f"(+{abs(upset.spread):.1f}) def. {upset.actual_loser.upper()}"
            )
            console.print(
                f"     Our Pick: {analysis.our_pick.upper()} {analysis.our_prob*100:.1f}% {marker} | "
                f"Vegas: {upset.vegas_favorite.upper()} {upset.implied_prob*100:.0f}%"
            )

            if analysis.anomaly_factors:
                factors_str = ", ".join(f.replace("_", " ") for f in analysis.anomaly_factors[:3])
                console.print(f"     [dim]Factors: {factors_str}[/dim]")

            console.print()

    console.print("=" * 78)


def _export_vegas_analysis(report, filepath: str) -> None:
    """Export Vegas analysis to JSON file."""
    export_data = {
        "season": report.season,
        "week": report.week,
        "summary": {
            "total_games": report.total_games,
            "games_with_lines": report.games_with_lines,
            "vegas_correct": report.vegas_correct,
            "vegas_wrong": report.vegas_wrong,
            "vegas_accuracy": report.vegas_accuracy,
            "we_predicted_upset": report.we_predicted_upset,
            "we_also_wrong": report.we_also_wrong,
            "our_upset_accuracy": report.our_upset_accuracy,
        },
        "by_spread_bucket": {
            k: {"upsets": v.upsets, "we_predicted": v.we_predicted, "rate": v.we_predicted_rate}
            for k, v in report.by_spread_bucket.items()
        },
        "by_conference_matchup": {
            k: {"upsets": v.upsets, "we_predicted": v.we_predicted, "rate": v.we_predicted_rate}
            for k, v in report.by_conference_matchup.items()
        },
        "anomaly_factors": [
            {
                "name": f.name,
                "description": f.description,
                "occurrences": f.occurrences,
                "we_predicted_rate": f.we_predicted_rate,
            }
            for f in report.anomaly_factors
        ],
        "upsets_we_called": [
            {
                "game_id": a.upset.game_id,
                "week": a.upset.week,
                "favorite": a.upset.vegas_favorite,
                "underdog": a.upset.vegas_underdog,
                "spread": a.upset.spread,
                "winner": a.upset.actual_winner,
                "our_pick": a.our_pick,
                "our_prob": a.our_prob,
                "factors": a.anomaly_factors,
            }
            for a in report.upsets_we_called
        ],
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[green]Analysis exported to: {filepath}[/green]")


@app.command("predict-game")
def predict_game_cmd(
    team1: Annotated[str, typer.Argument(help="First team (e.g., 'Ohio State' or ohio-state)")],
    team2: Annotated[str, typer.Argument(help="Second team (e.g., 'Michigan' or michigan)")],
    season: Annotated[int, typer.Option("--season", "-s", help="Season year")] = 2024,
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week number")] = None,
    neutral: Annotated[bool, typer.Option("--neutral", "-n", help="Neutral site game")] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile"),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to JSON config file"),
    ] = None,
) -> None:
    """Predict a game with all sources (our algorithm, Vegas, SP+, Elo)."""
    config = _load_config(profile=profile, config_file=config_file)

    # Normalize team names
    home_team = team2.lower().replace(" ", "-").replace("&", "and")
    away_team = team1.lower().replace(" ", "-").replace("&", "and")

    async def _run():
        try:
            client = CFBDataClient.from_env()
            games = await client.fetch_games(season)
            teams = await client.fetch_teams(fbs_only=False)
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None

        # Get our rankings
        engine = RankingEngine(config=config)
        if week:
            rankings = await engine.rank_as_of_week(season, week, games=games)
        else:
            rankings = await engine.rank_season(season, games=games, teams=teams)

        # Build rating lookup
        rating_lookup = {r.team_id: r for r in rankings}

        # Fetch validators and betting data
        validator_service = ValidatorService(client)
        validators = await validator_service.fetch_all_validators(season)

        # Try to find betting lines for specific game
        betting_lines = await client.fetch_betting_lines(season, week)
        pregame_wp_data = await client.fetch_pregame_wp(season, week)

        # Build lookups
        sp_lookup = {r.team_id: r for r in validators.get("sp", [])}
        elo_lookup = {r.team_id: r for r in validators.get("elo", [])}
        srs_lookup = {r.team_id: r for r in validators.get("srs", [])}

        # Find betting line for this game
        vegas_spread = None
        pregame_wp = None

        for line in betting_lines:
            if (line["home_team_id"] == home_team and line["away_team_id"] == away_team) or \
               (line["home_team_id"] == away_team and line["away_team_id"] == home_team):
                if line["home_team_id"] == home_team:
                    vegas_spread = line.get("spread")
                else:
                    vegas_spread = -line.get("spread") if line.get("spread") else None
                break

        for wp in pregame_wp_data:
            if (wp["home_team_id"] == home_team and wp["away_team_id"] == away_team) or \
               (wp["home_team_id"] == away_team and wp["away_team_id"] == home_team):
                if wp["home_team_id"] == home_team:
                    pregame_wp = wp.get("home_win_prob")
                else:
                    pregame_wp = 1 - wp.get("home_win_prob") if wp.get("home_win_prob") else None
                break

        # Get ratings
        home_rating = rating_lookup.get(home_team)
        away_rating = rating_lookup.get(away_team)

        home_sp = sp_lookup.get(home_team)
        away_sp = sp_lookup.get(away_team)

        home_elo_data = elo_lookup.get(home_team)
        away_elo_data = elo_lookup.get(away_team)

        # Build prediction
        prediction = build_game_prediction(
            home_team_id=home_team,
            away_team_id=away_team,
            home_rating=home_rating.rating if home_rating else None,
            away_rating=away_rating.rating if away_rating else None,
            vegas_spread=vegas_spread,
            pregame_wp=pregame_wp,
            home_sp_rating=home_sp.rating if home_sp else None,
            away_sp_rating=away_sp.rating if away_sp else None,
            home_elo=home_elo_data.rating if home_elo_data else None,
            away_elo=away_elo_data.rating if away_elo_data else None,
            season=season,
            week=week,
            neutral_site=neutral,
        )

        # Include validator ranks for display
        home_srs = srs_lookup.get(home_team)
        away_srs = srs_lookup.get(away_team)
        context = {
            "home_rank": home_rating.rank if home_rating else None,
            "away_rank": away_rating.rank if away_rating else None,
            "home_sp_rank": home_sp.rank if home_sp else None,
            "away_sp_rank": away_sp.rank if away_sp else None,
            "home_srs_rank": home_srs.rank if home_srs else None,
            "away_srs_rank": away_srs.rank if away_srs else None,
            "home_elo_rank": home_elo_data.rank if home_elo_data else None,
            "away_elo_rank": away_elo_data.rank if away_elo_data else None,
        }

        return prediction, context

    result = asyncio.run(_run())

    if result is None:
        return

    prediction, context = result

    _display_game_prediction(prediction, context, neutral)


def _display_game_prediction(prediction, context, neutral_site: bool) -> None:
    """Display comprehensive game prediction with all sources."""
    console.print()
    title = f"GAME PREDICTION: {prediction.away_team_id.upper()} @ {prediction.home_team_id.upper()}"
    if neutral_site:
        title += " (Neutral Site)"
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("=" * 75)
    console.print()

    # Source predictions table
    table = Table()
    table.add_column("Source", style="bold", width=25)
    table.add_column("Pick", justify="center", width=15)
    table.add_column("Win Prob", justify="right", width=10)
    table.add_column("Weight", justify="right", width=8)
    table.add_column("Contrib", justify="right", width=10)

    total_contrib = 0.0
    available_weight = 0.0

    for source_pred in prediction.source_predictions:
        if source_pred.available:
            winner = source_pred.predicted_winner or "-"
            prob = f"{source_pred.win_probability:.1%}" if source_pred.win_probability else "-"
            weight = f"{source_pred.weight:.0%}"

            # Calculate contribution
            if source_pred.win_probability:
                contrib = source_pred.win_probability * source_pred.weight
                total_contrib += contrib
                available_weight += source_pred.weight
                contrib_str = f"{contrib:.1%}"
            else:
                contrib_str = "-"

            # Add spread info for Vegas
            source_name = source_pred.source.replace("_", " ").title()
            if source_pred.source == "vegas" and source_pred.spread is not None:
                source_name = f"Vegas ({source_pred.spread:+.1f})"

            table.add_row(source_name, winner, prob, weight, contrib_str)
        else:
            source_name = source_pred.source.replace("_", " ").title()
            table.add_row(
                source_name,
                "[dim]Pending[/dim]",
                "[dim]--[/dim]",
                f"{source_pred.weight:.0%}",
                "[dim]--[/dim]",
            )

    # Consensus row
    table.add_section()
    conf_emoji = {"HIGH": "[green]HIGH[/green]", "MODERATE": "[yellow]MODERATE[/yellow]",
                  "LOW": "[red]LOW[/red]", "SPLIT": "[red]SPLIT[/red]", "UNKNOWN": "[dim]UNKNOWN[/dim]"}

    table.add_row(
        "[bold]CONSENSUS[/bold]",
        f"[bold]{prediction.consensus_winner}[/bold]",
        f"[bold]{prediction.consensus_prob:.1%}[/bold]",
        "100%",
        conf_emoji.get(prediction.confidence, prediction.confidence),
    )

    console.print(table)
    console.print()

    # Agreement line
    console.print(f"[dim]Agreement: {prediction.sources_agreeing}/{prediction.sources_available} sources favor {prediction.consensus_winner}[/dim]")
    console.print()

    # Rating comparison
    console.print("[bold]Rating Comparison[/bold]")
    console.print("-" * 75)

    home = prediction.home_team_id
    away = prediction.away_team_id

    home_parts = []
    away_parts = []

    if prediction.home_rating:
        home_parts.append(f"{prediction.home_rating:.3f} rating")
    if context.get("home_rank"):
        home_parts.append(f"Our #{context['home_rank']}")
    if context.get("home_sp_rank"):
        home_parts.append(f"SP+ #{context['home_sp_rank']}")
    if context.get("home_srs_rank"):
        home_parts.append(f"SRS #{context['home_srs_rank']}")
    if context.get("home_elo_rank"):
        home_parts.append(f"Elo #{context['home_elo_rank']}")

    if prediction.away_rating:
        away_parts.append(f"{prediction.away_rating:.3f} rating")
    if context.get("away_rank"):
        away_parts.append(f"Our #{context['away_rank']}")
    if context.get("away_sp_rank"):
        away_parts.append(f"SP+ #{context['away_sp_rank']}")
    if context.get("away_srs_rank"):
        away_parts.append(f"SRS #{context['away_srs_rank']}")
    if context.get("away_elo_rank"):
        away_parts.append(f"Elo #{context['away_elo_rank']}")

    console.print(f"  {home.upper()}: {' | '.join(home_parts) if home_parts else 'N/A'}")
    console.print(f"  {away.upper()}: {' | '.join(away_parts) if away_parts else 'N/A'}")
    if prediction.rating_gap is not None:
        console.print(f"  Gap: {prediction.rating_gap:+.3f} favoring {prediction.consensus_winner}")


# =============================================================================
# Config Commands
# =============================================================================

config_app = typer.Typer(help="Configuration management commands.")
app.add_typer(config_app, name="config")


@config_app.command("list")
def config_list() -> None:
    """List all available configuration profiles."""
    console.print()
    console.print("[bold cyan]Available Configuration Profiles[/bold cyan]")
    console.print()

    for name in list_profiles():
        desc = get_profile_description(name)
        console.print(f"  [bold]{name}[/bold]")
        console.print(f"    {desc}")
        console.print()


@config_app.command("show")
def config_show(
    profile: Annotated[
        str | None,
        typer.Argument(help="Profile name to show (or 'default' for defaults)"),
    ] = None,
) -> None:
    """Show configuration settings for a profile."""
    if profile:
        try:
            config = get_profile(profile)
            title = f"Configuration: {profile}"
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        config = AlgorithmConfig()
        title = "Configuration: Default"

    console.print()
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print()

    # Group settings by category
    categories = {
        "Core": ["max_iterations", "convergence_threshold", "win_base", "margin_weight", "margin_cap"],
        "Venue": ["venue_road_win", "venue_neutral_win", "venue_home_loss", "venue_neutral_loss"],
        "Opponent": ["opponent_weight", "loss_opponent_factor", "second_order_weight"],
        "Conference": ["enable_conference_adj", "conference_method", "p5_multiplier", "g5_multiplier", "fcs_multiplier"],
        "SOS": ["sos_adjustment_weight", "sos_method", "min_sos_top_10", "min_sos_top_25", "min_p5_games_top_10"],
        "Quality Tiers": ["enable_quality_tiers", "elite_threshold", "good_threshold", "bad_threshold", "elite_win_bonus", "bad_loss_penalty"],
        "Recency": ["enable_recency", "recency_half_life", "recency_min_weight"],
        "Prior": ["enable_prior", "prior_weight", "prior_decay_weeks"],
        "Margin Curve": ["margin_curve"],
        "Tiebreakers": ["tie_threshold", "tiebreaker_order"],
        "Filters": ["include_postseason", "postseason_weight", "include_fcs_games", "exclude_fcs_from_rankings", "min_games_to_rank"],
    }

    config_dict = config.model_dump()

    for category, fields in categories.items():
        console.print(f"[bold]{category}[/bold]")
        for field in fields:
            value = config_dict.get(field, "N/A")
            console.print(f"  {field}: {value}")
        console.print()


@config_app.command("export")
def config_export(
    output: Annotated[str, typer.Argument(help="Output file path")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile to export"),
    ] = None,
) -> None:
    """Export configuration to JSON file."""
    if profile:
        try:
            config = get_profile(profile)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        config = AlgorithmConfig()

    with open(output, "w") as f:
        f.write(config.model_dump_json(indent=2))

    console.print(f"[green]Configuration exported to {output}[/green]")


@config_app.command("create")
def config_create(
    output: Annotated[str, typer.Argument(help="Output file path")] = "my_config.json",
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Base profile to start from"),
    ] = None,
    documented: Annotated[
        bool,
        typer.Option("--documented", "-d", help="Include section comments"),
    ] = True,
) -> None:
    """Create a new configuration file with all levers documented."""
    if profile:
        try:
            config = get_profile(profile)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        config = AlgorithmConfig()

    config_dict = config.model_dump()

    if documented:
        # Create documented structure with section markers
        documented_config = {
            "_documentation": "See CONFIG_REFERENCE.md for detailed descriptions of each lever",
            "_profile_base": profile or "default",

            "___CONVERGENCE___": "Core algorithm iteration settings",
            "max_iterations": config_dict["max_iterations"],
            "convergence_threshold": config_dict["convergence_threshold"],

            "___GAME_GRADE___": "How individual games are scored",
            "win_base": config_dict["win_base"],
            "margin_weight": config_dict["margin_weight"],
            "margin_cap": config_dict["margin_cap"],
            "margin_curve": config_dict["margin_curve"],

            "___VENUE___": "Home/Away/Neutral adjustments",
            "venue_road_win": config_dict["venue_road_win"],
            "venue_neutral_win": config_dict["venue_neutral_win"],
            "venue_home_loss": config_dict["venue_home_loss"],
            "venue_neutral_loss": config_dict["venue_neutral_loss"],

            "___INITIALIZATION___": "Starting ratings",
            "initial_rating": config_dict["initial_rating"],
            "fcs_fixed_rating": config_dict["fcs_fixed_rating"],

            "___OPPONENT___": "How opponent strength affects rating",
            "opponent_weight": config_dict["opponent_weight"],
            "loss_opponent_factor": config_dict["loss_opponent_factor"],
            "second_order_weight": config_dict["second_order_weight"],

            "___CONFERENCE___": "P5/G5/FCS tier adjustments",
            "enable_conference_adj": config_dict["enable_conference_adj"],
            "conference_method": config_dict["conference_method"],
            "p5_multiplier": config_dict["p5_multiplier"],
            "g5_multiplier": config_dict["g5_multiplier"],
            "fcs_multiplier": config_dict["fcs_multiplier"],

            "___SCHEDULE_STRENGTH___": "SOS requirements and adjustments",
            "sos_adjustment_weight": config_dict["sos_adjustment_weight"],
            "sos_method": config_dict["sos_method"],
            "min_sos_top_10": config_dict["min_sos_top_10"],
            "min_sos_top_25": config_dict["min_sos_top_25"],
            "min_p5_games_top_10": config_dict["min_p5_games_top_10"],

            "___QUALITY_TIERS___": "Bonuses for elite wins, penalties for bad losses",
            "enable_quality_tiers": config_dict["enable_quality_tiers"],
            "elite_threshold": config_dict["elite_threshold"],
            "good_threshold": config_dict["good_threshold"],
            "bad_threshold": config_dict["bad_threshold"],
            "elite_win_bonus": config_dict["elite_win_bonus"],
            "bad_loss_penalty": config_dict["bad_loss_penalty"],

            "___RECENCY___": "Weight recent games more heavily",
            "enable_recency": config_dict["enable_recency"],
            "recency_half_life": config_dict["recency_half_life"],
            "recency_min_weight": config_dict["recency_min_weight"],

            "___PRIOR___": "Preseason expectations influence",
            "enable_prior": config_dict["enable_prior"],
            "prior_weight": config_dict["prior_weight"],
            "prior_decay_weeks": config_dict["prior_decay_weeks"],

            "___TIEBREAKERS___": "How ties are resolved",
            "tie_threshold": config_dict["tie_threshold"],
            "tiebreaker_order": config_dict["tiebreaker_order"],

            "___FILTERS___": "What games/teams to include",
            "include_postseason": config_dict["include_postseason"],
            "postseason_weight": config_dict["postseason_weight"],
            "include_fcs_games": config_dict["include_fcs_games"],
            "exclude_fcs_from_rankings": config_dict["exclude_fcs_from_rankings"],
            "min_games_to_rank": config_dict["min_games_to_rank"],
        }
        output_dict = documented_config
    else:
        output_dict = config_dict

    with open(output, "w") as f:
        json.dump(output_dict, f, indent=2)

    console.print(f"[green]Configuration created: {output}[/green]")
    console.print(f"[dim]Edit the file to customize, then run:[/dim]")
    console.print(f"  ncaa-rank rank 2025 --config {output} --top 25")


if __name__ == "__main__":
    app()
