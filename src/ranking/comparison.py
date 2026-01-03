"""Historical comparison module for comparing algorithm rankings to polls.

Compares our algorithm's rankings to AP, CFP, or Coaches polls using
Spearman rank correlation and identifies over/underranked teams.
"""

import math

from scipy.stats import spearmanr

from src.data.models import ComparisonResult, TeamRating
from src.ranking.engine import RankingEngine


class RankingComparator:
    """Compares algorithm rankings to historical polls."""

    def __init__(self, engine: RankingEngine):
        """
        Initialize the comparator.

        Args:
            engine: RankingEngine for generating algorithm rankings
        """
        self.engine = engine

    @staticmethod
    def _compute_spearman(
        our_ranks: list[int],
        poll_ranks: list[int],
    ) -> float:
        """
        Compute Spearman rank correlation coefficient.

        Args:
            our_ranks: List of our ranking positions
            poll_ranks: List of poll ranking positions

        Returns:
            Correlation coefficient between -1 and 1.
            Returns 0.0 for empty lists, 1.0 for single element.
        """
        if not our_ranks or not poll_ranks:
            return 0.0

        if len(our_ranks) == 1:
            return 1.0

        if len(our_ranks) != len(poll_ranks):
            raise ValueError("Rank lists must have the same length")

        correlation, _ = spearmanr(our_ranks, poll_ranks)

        # Handle NaN (can occur with constant arrays)
        if math.isnan(correlation):
            return 1.0

        return float(correlation)

    async def compare_to_poll(
        self,
        season: int,
        week: int,
        poll_type: str = "ap",
        poll_rankings: list[dict] | None = None,
        our_rankings: list[TeamRating] | None = None,
    ) -> ComparisonResult:
        """
        Compare algorithm rankings to a poll.

        Args:
            season: Season year
            week: Week number
            poll_type: Poll type ("ap", "cfp", or "coaches")
            poll_rankings: Optional pre-fetched poll rankings (for testing)
            our_rankings: Optional pre-computed algorithm rankings (for testing)

        Returns:
            ComparisonResult with correlation and over/underranked teams
        """
        # Use provided rankings or generate them
        if our_rankings is None:
            our_rankings = []

        if poll_rankings is None:
            poll_rankings = []

        # Build lookup for our rankings
        our_rank_lookup: dict[str, int] = {
            r.team_id: r.rank for r in our_rankings
        }

        # Build lookup for poll rankings
        poll_rank_lookup: dict[str, int] = {
            r["team_id"]: r["rank"] for r in poll_rankings
        }

        # Find teams that appear in both rankings
        common_teams = set(our_rank_lookup.keys()) & set(poll_rank_lookup.keys())

        if not common_teams:
            return ComparisonResult(
                season=season,
                week=week,
                poll_type=poll_type,
                spearman_correlation=0.0,
                teams_compared=0,
                overranked=[],
                underranked=[],
            )

        # Extract matched ranks in consistent order
        sorted_teams = sorted(common_teams)
        our_ranks = [our_rank_lookup[t] for t in sorted_teams]
        poll_ranks = [poll_rank_lookup[t] for t in sorted_teams]

        # Compute correlation
        correlation = self._compute_spearman(our_ranks, poll_ranks)

        # Identify over/underranked teams
        overranked: list[dict] = []
        underranked: list[dict] = []

        for team_id in sorted_teams:
            our_rank = our_rank_lookup[team_id]
            poll_rank = poll_rank_lookup[team_id]
            difference = poll_rank - our_rank  # Positive = we rank higher

            if difference > 0:
                # We rank them higher (lower number) than poll
                overranked.append({
                    "team_id": team_id,
                    "our_rank": our_rank,
                    "poll_rank": poll_rank,
                    "difference": difference,
                })
            elif difference < 0:
                # We rank them lower (higher number) than poll
                underranked.append({
                    "team_id": team_id,
                    "our_rank": our_rank,
                    "poll_rank": poll_rank,
                    "difference": difference,
                })

        # Sort by absolute difference (largest first)
        overranked.sort(key=lambda x: x["difference"], reverse=True)
        underranked.sort(key=lambda x: abs(x["difference"]), reverse=True)

        return ComparisonResult(
            season=season,
            week=week,
            poll_type=poll_type,
            spearman_correlation=correlation,
            teams_compared=len(common_teams),
            overranked=overranked,
            underranked=underranked,
        )

    async def find_biggest_differences(
        self,
        season: int,
        week: int,
        poll_type: str = "ap",
        limit: int = 10,
        poll_rankings: list[dict] | None = None,
        our_rankings: list[TeamRating] | None = None,
    ) -> list[dict]:
        """
        Find teams with the biggest rank differences.

        Args:
            season: Season year
            week: Week number
            poll_type: Poll type ("ap", "cfp", or "coaches")
            limit: Maximum number of results to return
            poll_rankings: Optional pre-fetched poll rankings (for testing)
            our_rankings: Optional pre-computed algorithm rankings (for testing)

        Returns:
            List of dicts with team_id, our_rank, poll_rank, difference
            sorted by absolute difference (largest first)
        """
        # Use provided rankings or generate them
        if our_rankings is None:
            our_rankings = []

        if poll_rankings is None:
            poll_rankings = []

        # Build lookups
        our_rank_lookup: dict[str, int] = {
            r.team_id: r.rank for r in our_rankings
        }

        poll_rank_lookup: dict[str, int] = {
            r["team_id"]: r["rank"] for r in poll_rankings
        }

        # Find common teams
        common_teams = set(our_rank_lookup.keys()) & set(poll_rank_lookup.keys())

        # Build differences list
        differences: list[dict] = []
        for team_id in common_teams:
            our_rank = our_rank_lookup[team_id]
            poll_rank = poll_rank_lookup[team_id]
            difference = poll_rank - our_rank

            differences.append({
                "team_id": team_id,
                "our_rank": our_rank,
                "poll_rank": poll_rank,
                "difference": difference,
            })

        # Sort by absolute difference (largest first)
        differences.sort(key=lambda x: abs(x["difference"]), reverse=True)

        # Apply limit
        return differences[:limit]
