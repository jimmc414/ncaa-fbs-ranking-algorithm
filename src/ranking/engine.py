"""Ranking engine that orchestrates the full ranking pipeline.

The RankingEngine ties together:
- Convergence algorithm (computes raw ratings)
- Tiebreaker logic (resolves ties)
- TeamRating generation (final output with rank, SOS, SOV)
"""

from src.algorithm.convergence import converge, normalize_ratings
from src.algorithm.game_grade import compute_game_grade_for_result
from src.algorithm.tiebreaker import resolve_ties, strength_of_victory
from src.data.models import (
    AlgorithmConfig,
    Game,
    RatingExplanation,
    TeamRating,
)


class RankingEngine:
    """Orchestrates the ranking pipeline."""

    def __init__(
        self,
        config: AlgorithmConfig | None = None,
    ):
        """
        Initialize the ranking engine.

        Args:
            config: Algorithm configuration (uses defaults if None)
        """
        self.config = config or AlgorithmConfig()

    async def rank_season(
        self,
        season: int,
        include_postseason: bool = True,
        games: list[Game] | None = None,
    ) -> list[TeamRating]:
        """
        Generate rankings for a full season.

        Args:
            season: Season year
            include_postseason: Whether to include postseason games
            games: List of games (if None, would fetch from API)

        Returns:
            List of TeamRating ordered by rank
        """
        if games is None:
            games = []

        if not games:
            return []

        # Filter postseason games if needed
        if not include_postseason:
            games = [g for g in games if not g.postseason]

        if not games:
            return []

        # Run convergence algorithm
        result = converge(games, self.config)
        raw_ratings = result.ratings

        if not raw_ratings:
            return []

        # Normalize ratings to [0, 1]
        normalized = normalize_ratings(raw_ratings)

        # Create (team_id, rating) tuples sorted by rating descending
        teams_with_ratings = sorted(
            [(tid, rating) for tid, rating in normalized.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Resolve ties
        ordered_team_ids = resolve_ties(
            teams_with_ratings,
            games,
            normalized,
            threshold=0.001,
        )

        # Build game results lookup for win/loss counting
        team_games = self._build_team_games(games)

        # Generate TeamRating objects
        rankings = []
        for rank, team_id in enumerate(ordered_team_ids, start=1):
            team_game_list = team_games.get(team_id, [])
            wins, losses = self._count_wins_losses(team_id, team_game_list)

            sos = self._compute_sos(team_id, games, normalized)
            sov = strength_of_victory(team_id, games, normalized)

            rating = TeamRating(
                team_id=team_id,
                season=season,
                week=max(g.week for g in games),
                rating=normalized[team_id],
                rank=rank,
                games_played=len(team_game_list),
                wins=wins,
                losses=losses,
                strength_of_schedule=sos,
                strength_of_victory=sov,
            )
            rankings.append(rating)

        return rankings

    async def rank_as_of_week(
        self,
        season: int,
        week: int,
        include_postseason: bool = True,
        games: list[Game] | None = None,
    ) -> list[TeamRating]:
        """
        Generate rankings through a specific week.

        Args:
            season: Season year
            week: Week number (inclusive)
            include_postseason: Whether to include postseason games
            games: List of games (if None, would fetch from API)

        Returns:
            List of TeamRating ordered by rank
        """
        if games is None:
            games = []

        # Filter to games through this week
        filtered_games = [g for g in games if g.week <= week]

        return await self.rank_season(
            season,
            include_postseason=include_postseason,
            games=filtered_games,
        )

    async def rank_all_weeks(
        self,
        season: int,
        include_postseason: bool = True,
        games: list[Game] | None = None,
    ) -> dict[int, list[TeamRating]]:
        """
        Generate week-by-week ranking progression.

        Args:
            season: Season year
            include_postseason: Whether to include postseason games
            games: List of games (if None, would fetch from API)

        Returns:
            Dict mapping week number to rankings
        """
        if games is None:
            games = []

        if not games:
            return {}

        # Find all weeks with games
        weeks = sorted(set(g.week for g in games))

        result = {}
        for week in weeks:
            rankings = await self.rank_as_of_week(
                season,
                week,
                include_postseason=include_postseason,
                games=games,
            )
            if rankings:
                result[week] = rankings

        return result

    async def explain_rating(
        self,
        season: int,
        team_id: str,
        week: int | None = None,
        games: list[Game] | None = None,
    ) -> RatingExplanation:
        """
        Get detailed breakdown of how a team's rating was computed.

        Args:
            season: Season year
            team_id: Team to explain
            week: Optional week (uses full season if None)
            games: List of games (if None, would fetch from API)

        Returns:
            RatingExplanation with game-by-game breakdown
        """
        if games is None:
            games = []

        # Filter by week if specified
        if week is not None:
            games = [g for g in games if g.week <= week]

        # Run convergence
        result = converge(games, self.config)
        raw_ratings = result.ratings
        normalized = normalize_ratings(raw_ratings)

        # Get team's rank
        teams_with_ratings = sorted(
            [(tid, rating) for tid, rating in normalized.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        ordered_team_ids = resolve_ties(
            teams_with_ratings,
            games,
            normalized,
            threshold=0.001,
        )
        rank = ordered_team_ids.index(team_id) + 1 if team_id in ordered_team_ids else 0

        # Build game-by-game breakdown
        game_contributions = []
        total_contribution = 0.0

        for game in games:
            if game.home_team_id != team_id and game.away_team_id != team_id:
                continue

            home_result, away_result = game.to_results()
            team_result = home_result if game.home_team_id == team_id else away_result

            game_grade = compute_game_grade_for_result(team_result, self.config)
            opp_rating = normalized.get(team_result.opponent_id, 0.5)

            if team_result.is_win:
                contribution = game_grade + opp_rating
            else:
                contribution = game_grade - opp_rating

            total_contribution += contribution

            game_contributions.append({
                "game_id": game.game_id,
                "opponent_id": team_result.opponent_id,
                "is_win": team_result.is_win,
                "margin": team_result.margin,
                "location": team_result.location,
                "game_grade": game_grade,
                "opponent_rating": opp_rating,
                "contribution": contribution,
            })

        return RatingExplanation(
            team_id=team_id,
            season=season,
            week=week or (max(g.week for g in games) if games else 0),
            final_rating=raw_ratings.get(team_id, 0.0),
            normalized_rating=normalized.get(team_id, 0.0),
            rank=rank,
            games=game_contributions,
            total_contribution=total_contribution,
            iterations_to_converge=result.iterations,
        )

    def _build_team_games(self, games: list[Game]) -> dict[str, list[Game]]:
        """Build a lookup of games by team."""
        team_games: dict[str, list[Game]] = {}
        for game in games:
            team_games.setdefault(game.home_team_id, []).append(game)
            team_games.setdefault(game.away_team_id, []).append(game)
        return team_games

    def _count_wins_losses(
        self,
        team_id: str,
        games: list[Game],
    ) -> tuple[int, int]:
        """Count wins and losses for a team."""
        wins = 0
        losses = 0

        for game in games:
            if game.home_team_id == team_id:
                if game.home_score > game.away_score:
                    wins += 1
                elif game.away_score > game.home_score:
                    losses += 1
            else:  # away team
                if game.away_score > game.home_score:
                    wins += 1
                elif game.home_score > game.away_score:
                    losses += 1

        return wins, losses

    def _compute_sos(
        self,
        team_id: str,
        games: list[Game],
        ratings: dict[str, float],
    ) -> float:
        """
        Compute strength of schedule (average rating of all opponents).

        Args:
            team_id: Team to compute SOS for
            games: List of all games
            ratings: Dict of team ratings

        Returns:
            Average rating of opponents played
        """
        opponents: list[float] = []

        for game in games:
            if game.home_team_id == team_id:
                opp = game.away_team_id
            elif game.away_team_id == team_id:
                opp = game.home_team_id
            else:
                continue

            if opp in ratings:
                opponents.append(ratings[opp])

        if not opponents:
            return 0.0

        return sum(opponents) / len(opponents)
