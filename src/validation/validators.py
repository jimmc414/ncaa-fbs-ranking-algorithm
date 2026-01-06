"""Service for fetching and analyzing external validators."""

import asyncio
from typing import TYPE_CHECKING

from scipy.stats import spearmanr

from src.validation.models import (
    TeamValidation,
    ValidationReport,
    ValidatorRating,
)

if TYPE_CHECKING:
    from src.data.client import CFBDataClient
    from src.data.models import TeamRating


class ValidatorService:
    """Fetch and analyze external validators (SP+, SRS, Elo)."""

    def __init__(self, client: "CFBDataClient"):
        """
        Initialize the validator service.

        Args:
            client: CFBDataClient for API access
        """
        self.client = client

    async def fetch_all_validators(
        self,
        season: int,
    ) -> dict[str, list[ValidatorRating]]:
        """
        Fetch all validator ratings in parallel.

        Args:
            season: Season year

        Returns:
            Dict mapping source name to list of ValidatorRating objects
        """
        results = await asyncio.gather(
            self._fetch_sp(season),
            self._fetch_srs(season),
            self._fetch_elo(season),
            return_exceptions=True,
        )

        validators = {}

        # SP+
        if not isinstance(results[0], Exception):
            validators["sp"] = results[0]

        # SRS
        if not isinstance(results[1], Exception):
            validators["srs"] = results[1]

        # Elo
        if not isinstance(results[2], Exception):
            validators["elo"] = results[2]

        return validators

    async def _fetch_sp(self, season: int) -> list[ValidatorRating]:
        """Fetch and parse SP+ ratings."""
        data = await self.client.fetch_sp_ratings(season)
        ratings = []
        for item in data:
            if item.get("ranking") is not None:
                ratings.append(
                    ValidatorRating(
                        team_id=item["team_id"],
                        source="sp",
                        rating=item.get("rating", 0.0),
                        rank=item.get("ranking"),
                        offense=item.get("offense"),
                        defense=item.get("defense"),
                    )
                )
        return ratings

    async def _fetch_srs(self, season: int) -> list[ValidatorRating]:
        """Fetch and parse SRS ratings."""
        data = await self.client.fetch_srs_ratings(season)
        ratings = []
        for item in data:
            if item.get("ranking") is not None:
                ratings.append(
                    ValidatorRating(
                        team_id=item["team_id"],
                        source="srs",
                        rating=item.get("rating", 0.0),
                        rank=item.get("ranking"),
                    )
                )
        return ratings

    async def _fetch_elo(self, season: int) -> list[ValidatorRating]:
        """Fetch and parse Elo ratings."""
        data = await self.client.fetch_elo_ratings(season)
        # Elo doesn't have ranking, we need to compute it
        sorted_data = sorted(data, key=lambda x: x.get("elo", 0), reverse=True)
        ratings = []
        for rank, item in enumerate(sorted_data, start=1):
            if item.get("elo") is not None:
                ratings.append(
                    ValidatorRating(
                        team_id=item["team_id"],
                        source="elo",
                        rating=item.get("elo", 0.0),
                        rank=rank,
                    )
                )
        return ratings

    def compare_rankings(
        self,
        our_rankings: list["TeamRating"],
        validators: dict[str, list[ValidatorRating]],
        threshold: int = 20,
    ) -> ValidationReport:
        """
        Compare our rankings to external validators.

        Args:
            our_rankings: Our generated rankings
            validators: Dict of validator source -> ratings
            threshold: Rank gap threshold to flag a team

        Returns:
            ValidationReport with correlations and flagged teams
        """
        # Build lookup for our rankings
        our_lookup = {r.team_id: r for r in our_rankings}

        # Build lookups for each validator
        validator_lookups: dict[str, dict[str, ValidatorRating]] = {}
        for source, ratings in validators.items():
            validator_lookups[source] = {r.team_id: r for r in ratings}

        # Calculate correlations
        correlations = {}
        for source, ratings in validators.items():
            correlation = self._calculate_correlation(our_rankings, ratings)
            if correlation is not None:
                correlations[source] = correlation

        # Find flagged teams
        flagged_teams = []
        flagged_count_by_source: dict[str, int] = {source: 0 for source in validators}

        for our_rating in our_rankings:
            team_id = our_rating.team_id
            validator_ranks: dict[str, int] = {}
            validator_ratings: dict[str, float] = {}
            max_gap = 0

            for source, lookup in validator_lookups.items():
                if team_id in lookup:
                    val_rating = lookup[team_id]
                    if val_rating.rank is not None:
                        validator_ranks[source] = val_rating.rank
                        validator_ratings[source] = val_rating.rating
                        gap = abs(our_rating.rank - val_rating.rank)
                        if gap > max_gap:
                            max_gap = gap

            # Check if flagged
            flagged = False
            flag_sources = 0
            for source, rank in validator_ranks.items():
                gap = abs(our_rating.rank - rank)
                if gap >= threshold:
                    flagged = True
                    flag_sources += 1
                    flagged_count_by_source[source] += 1

            team_validation = TeamValidation(
                team_id=team_id,
                our_rank=our_rating.rank,
                our_rating=our_rating.rating,
                validator_ranks=validator_ranks,
                validator_ratings=validator_ratings,
                max_gap=max_gap,
                flagged=flagged,
            )

            if flagged:
                flagged_teams.append(team_validation)

        # Sort flagged teams by max gap
        flagged_teams.sort(key=lambda t: t.max_gap, reverse=True)

        # Detect patterns
        patterns = self._detect_patterns(flagged_teams, our_lookup)

        return ValidationReport(
            season=our_rankings[0].season if our_rankings else 0,
            week=our_rankings[0].week if our_rankings else None,
            correlations=correlations,
            flagged_teams=flagged_teams,
            flagged_count_by_source=flagged_count_by_source,
            patterns=patterns,
            total_teams_compared=len(our_rankings),
        )

    def _calculate_correlation(
        self,
        our_rankings: list["TeamRating"],
        validator_ratings: list[ValidatorRating],
    ) -> float | None:
        """Calculate Spearman rank correlation."""
        # Find common teams
        our_lookup = {r.team_id: r.rank for r in our_rankings}
        validator_lookup = {r.team_id: r.rank for r in validator_ratings if r.rank is not None}

        common_teams = set(our_lookup.keys()) & set(validator_lookup.keys())
        if len(common_teams) < 10:
            return None

        our_ranks = []
        val_ranks = []
        for team_id in common_teams:
            our_ranks.append(our_lookup[team_id])
            val_ranks.append(validator_lookup[team_id])

        correlation, _ = spearmanr(our_ranks, val_ranks)
        return float(correlation) if correlation is not None else None

    def _detect_patterns(
        self,
        flagged_teams: list[TeamValidation],
        our_lookup: dict[str, "TeamRating"],
    ) -> list[str]:
        """Detect common patterns among flagged teams."""
        patterns = []

        if not flagged_teams:
            return patterns

        # Count overrated vs underrated
        overrated = sum(1 for t in flagged_teams if t.get_gap("sp") is not None and t.get_gap("sp") < 0)
        underrated = len(flagged_teams) - overrated

        if overrated > underrated * 2:
            patterns.append(f"{overrated}/{len(flagged_teams)} flagged teams appear overrated by our algorithm")
        elif underrated > overrated * 2:
            patterns.append(f"{underrated}/{len(flagged_teams)} flagged teams appear underrated by our algorithm")

        # Check for weak SOS pattern
        weak_sos_count = 0
        for t in flagged_teams:
            rating = our_lookup.get(t.team_id)
            if rating and rating.strength_of_schedule and rating.strength_of_schedule < 0.45:
                weak_sos_count += 1

        if weak_sos_count >= len(flagged_teams) * 0.6 and weak_sos_count >= 2:
            patterns.append(f"{weak_sos_count}/{len(flagged_teams)} flagged teams have weak SOS (<0.45)")

        return patterns

    def get_team_validation(
        self,
        team_id: str,
        our_rankings: list["TeamRating"],
        validators: dict[str, list[ValidatorRating]],
    ) -> TeamValidation | None:
        """
        Get detailed validation for a single team.

        Args:
            team_id: Team ID to look up
            our_rankings: Our generated rankings
            validators: Dict of validator source -> ratings

        Returns:
            TeamValidation for the team, or None if not found
        """
        # Find in our rankings
        our_rating = None
        for r in our_rankings:
            if r.team_id == team_id:
                our_rating = r
                break

        if our_rating is None:
            return None

        # Build validator lookups
        validator_ranks: dict[str, int] = {}
        validator_ratings: dict[str, float] = {}
        max_gap = 0

        for source, ratings in validators.items():
            for r in ratings:
                if r.team_id == team_id:
                    if r.rank is not None:
                        validator_ranks[source] = r.rank
                        validator_ratings[source] = r.rating
                        gap = abs(our_rating.rank - r.rank)
                        if gap > max_gap:
                            max_gap = gap
                    break

        return TeamValidation(
            team_id=team_id,
            our_rank=our_rating.rank,
            our_rating=our_rating.rating,
            validator_ranks=validator_ranks,
            validator_ratings=validator_ratings,
            max_gap=max_gap,
            flagged=max_gap >= 20,
        )
