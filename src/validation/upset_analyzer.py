"""Vegas upset analyzer - Find patterns in games where Vegas got it wrong."""

import asyncio
from collections import defaultdict

from src.data.client import CFBDataClient
from src.data.models import Game, TeamRating
from src.validation.consensus import spread_to_probability, rating_gap_to_probability
from src.validation.models import (
    AnomalyFactor,
    PatternReport,
    UpsetAnalysis,
    UpsetStats,
    ValidatorRating,
    VegasUpset,
)


# P5 conferences for classification
P5_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
G5_CONFERENCES = {"American Athletic", "Mountain West", "Sun Belt", "MAC", "Conference USA"}


def get_conference_tier(conference: str | None) -> str:
    """Classify conference into P5, G5, FCS, or Independent."""
    if conference is None:
        return "FCS"
    if conference in P5_CONFERENCES:
        return "P5"
    if conference in G5_CONFERENCES:
        return "G5"
    if conference == "FBS Independents":
        return "IND"
    return "FCS"


def get_week_range(week: int) -> str:
    """Categorize week into early/mid/late season."""
    if week <= 4:
        return "early (1-4)"
    elif week <= 9:
        return "mid (5-9)"
    else:
        return "late (10+)"


class UpsetAnalyzer:
    """Analyze games where Vegas favorites lost to find patterns."""

    def __init__(self, client: CFBDataClient):
        self.client = client

    async def find_upsets(
        self,
        games: list[Game],
        betting_lines: list[dict],
        min_spread: float = 3.0,
    ) -> list[VegasUpset]:
        """
        Find all games where the Vegas favorite lost outright.

        Args:
            games: List of completed games
            betting_lines: Betting lines data from API
            min_spread: Minimum spread to consider (filters out toss-ups)

        Returns:
            List of VegasUpset objects
        """
        upsets = []

        # Build lookup for betting lines by game_id
        lines_by_game = {}
        for line in betting_lines:
            game_id = line.get("game_id") or line.get("id")
            if game_id and line.get("spread"):
                lines_by_game[game_id] = line

        for game in games:
            line_data = lines_by_game.get(game.game_id)
            if not line_data:
                continue

            spread = line_data.get("spread")
            if spread is None:
                continue

            # Spread is typically from home team perspective
            # Negative spread = home team favored
            abs_spread = abs(spread)
            if abs_spread < min_spread:
                continue  # Skip toss-ups

            # Determine favorite/underdog
            if spread < 0:
                # Home team is favorite
                vegas_favorite = game.home_team_id
                vegas_underdog = game.away_team_id
            else:
                # Away team is favorite
                vegas_favorite = game.away_team_id
                vegas_underdog = game.home_team_id

            # Determine actual winner
            if game.home_score > game.away_score:
                actual_winner = game.home_team_id
                actual_loser = game.away_team_id
                final_margin = game.home_score - game.away_score
            elif game.away_score > game.home_score:
                actual_winner = game.away_team_id
                actual_loser = game.home_team_id
                final_margin = game.away_score - game.home_score
            else:
                continue  # Skip ties

            # Check if this was an upset (underdog won)
            if actual_winner == vegas_underdog:
                # Calculate implied probability from spread
                implied_prob = spread_to_probability(spread if spread < 0 else -spread)

                upset = VegasUpset(
                    game_id=game.game_id,
                    season=game.season,
                    week=game.week,
                    vegas_favorite=vegas_favorite,
                    vegas_underdog=vegas_underdog,
                    spread=-abs_spread,  # Always negative (from favorite's perspective)
                    implied_prob=implied_prob,
                    actual_winner=actual_winner,
                    actual_loser=actual_loser,
                    final_margin=final_margin,
                    home_team=game.home_team_id,
                    neutral_site=game.neutral_site,
                )
                upsets.append(upset)

        return upsets

    def analyze_upset(
        self,
        upset: VegasUpset,
        our_ratings: dict[str, TeamRating],
        teams: dict[str, dict],
        validators: dict[str, list[ValidatorRating]] | None = None,
    ) -> UpsetAnalysis:
        """
        Perform deep analysis of a single upset.

        Args:
            upset: The upset to analyze
            our_ratings: Our team ratings keyed by team_id
            teams: Team info dict with conference data
            validators: Optional validator ratings (SP+, SRS, Elo)

        Returns:
            UpsetAnalysis with detailed breakdown
        """
        favorite_rating = our_ratings.get(upset.vegas_favorite)
        underdog_rating = our_ratings.get(upset.vegas_underdog)

        # Calculate our prediction
        if favorite_rating and underdog_rating:
            # Calculate win probability for home team
            is_favorite_home = upset.vegas_favorite == upset.home_team
            if is_favorite_home:
                home_rating = favorite_rating.rating
                away_rating = underdog_rating.rating
            else:
                home_rating = underdog_rating.rating
                away_rating = favorite_rating.rating

            home_prob = rating_gap_to_probability(
                home_rating - away_rating,
                home_advantage=0.0 if upset.neutral_site else 0.03,
            )

            # Determine our pick
            if is_favorite_home:
                our_favorite_prob = home_prob
            else:
                our_favorite_prob = 1 - home_prob

            if our_favorite_prob > 0.5:
                our_pick = upset.vegas_favorite
                our_prob = our_favorite_prob
            else:
                our_pick = upset.vegas_underdog
                our_prob = 1 - our_favorite_prob

            we_predicted_upset = our_pick == upset.vegas_underdog

            fav_rating_val = favorite_rating.rating
            und_rating_val = underdog_rating.rating
            rating_gap = fav_rating_val - und_rating_val

            fav_sos = favorite_rating.strength_of_schedule or 0.5
            und_sos = underdog_rating.strength_of_schedule or 0.5
            sos_gap = fav_sos - und_sos
        else:
            our_pick = upset.vegas_favorite  # Default to favorite if no ratings
            our_prob = 0.5
            we_predicted_upset = False
            fav_rating_val = 0.5
            und_rating_val = 0.5
            rating_gap = 0.0
            fav_sos = 0.5
            und_sos = 0.5
            sos_gap = 0.0

        # Determine conference matchup
        fav_team = teams.get(upset.vegas_favorite, {})
        und_team = teams.get(upset.vegas_underdog, {})
        fav_conf = fav_team.get("conference")
        und_conf = und_team.get("conference")
        fav_tier = get_conference_tier(fav_conf)
        und_tier = get_conference_tier(und_conf)
        conference_matchup = f"{fav_tier}v{und_tier}"
        is_conference_game = fav_conf == und_conf and fav_conf is not None

        # Check validator predictions
        sp_pick = None
        srs_pick = None
        elo_pick = None
        validators_predicting_upset = 0

        if validators:
            # SP+ prediction
            sp_ratings = {r.team_id: r for r in validators.get("sp", [])}
            fav_sp = sp_ratings.get(upset.vegas_favorite)
            und_sp = sp_ratings.get(upset.vegas_underdog)
            if fav_sp and und_sp:
                if und_sp.rating > fav_sp.rating:
                    sp_pick = upset.vegas_underdog
                    validators_predicting_upset += 1
                else:
                    sp_pick = upset.vegas_favorite

            # SRS prediction
            srs_ratings = {r.team_id: r for r in validators.get("srs", [])}
            fav_srs = srs_ratings.get(upset.vegas_favorite)
            und_srs = srs_ratings.get(upset.vegas_underdog)
            if fav_srs and und_srs:
                if und_srs.rating > fav_srs.rating:
                    srs_pick = upset.vegas_underdog
                    validators_predicting_upset += 1
                else:
                    srs_pick = upset.vegas_favorite

            # Elo prediction
            elo_ratings = {r.team_id: r for r in validators.get("elo", [])}
            fav_elo = elo_ratings.get(upset.vegas_favorite)
            und_elo = elo_ratings.get(upset.vegas_underdog)
            if fav_elo and und_elo:
                if und_elo.rating > fav_elo.rating:
                    elo_pick = upset.vegas_underdog
                    validators_predicting_upset += 1
                else:
                    elo_pick = upset.vegas_favorite

        # Identify anomaly factors
        anomaly_factors = self._identify_anomaly_factors(
            upset=upset,
            rating_gap=rating_gap,
            sos_gap=sos_gap,
            fav_sos=fav_sos,
            we_predicted=we_predicted_upset,
            conference_matchup=conference_matchup,
        )

        return UpsetAnalysis(
            upset=upset,
            our_pick=our_pick,
            our_prob=our_prob,
            we_predicted_upset=we_predicted_upset,
            sp_pick=sp_pick,
            srs_pick=srs_pick,
            elo_pick=elo_pick,
            validators_predicting_upset=validators_predicting_upset,
            favorite_rating=fav_rating_val,
            underdog_rating=und_rating_val,
            rating_gap=rating_gap,
            favorite_sos=fav_sos,
            underdog_sos=und_sos,
            sos_gap=sos_gap,
            conference_matchup=conference_matchup,
            is_conference_game=is_conference_game,
            anomaly_factors=anomaly_factors,
        )

    def _identify_anomaly_factors(
        self,
        upset: VegasUpset,
        rating_gap: float,
        sos_gap: float,
        fav_sos: float,
        we_predicted: bool,
        conference_matchup: str,
    ) -> list[str]:
        """Identify factors that may have contributed to the upset."""
        factors = []

        # Weak favorite SOS
        if fav_sos < 0.45:
            factors.append("weak_favorite_sos")

        # Small rating gap (we had them close)
        if rating_gap < 0.05:
            factors.append("small_rating_gap")

        # We disagreed with Vegas
        if we_predicted:
            factors.append("we_predicted_upset")

        # P5 at G5 (historically upset-prone)
        if conference_matchup == "P5vG5" and not upset.favorite_was_home:
            factors.append("p5_road_at_g5")

        # Late season
        if upset.week >= 10:
            factors.append("late_season")

        # Large spread (big upsets are notable)
        if abs(upset.spread) >= 10:
            factors.append("large_spread_upset")

        # Favorite was away (road favorites are vulnerable)
        if not upset.favorite_was_home and not upset.neutral_site:
            factors.append("road_favorite")

        return factors

    def generate_pattern_report(
        self,
        upsets: list[UpsetAnalysis],
        all_games_count: int,
        games_with_lines_count: int,
        season: int,
        week: int | None = None,
    ) -> PatternReport:
        """
        Generate aggregate pattern analysis from upset data.

        Args:
            upsets: List of analyzed upsets
            all_games_count: Total games in the period
            games_with_lines_count: Games that had betting lines
            season: Season year
            week: Optional specific week

        Returns:
            PatternReport with aggregate analysis
        """
        report = PatternReport(
            season=season,
            week=week,
            total_games=all_games_count,
            games_with_lines=games_with_lines_count,
            vegas_wrong=len(upsets),
            vegas_correct=games_with_lines_count - len(upsets),
        )

        # Count our predictions
        report.we_predicted_upset = sum(1 for u in upsets if u.we_predicted_upset)
        report.we_also_wrong = len(upsets) - report.we_predicted_upset

        # Initialize stats buckets
        spread_buckets = defaultdict(lambda: UpsetStats())
        conf_buckets = defaultdict(lambda: UpsetStats())
        week_buckets = defaultdict(lambda: UpsetStats())
        factor_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "predicted": 0})

        # Aggregate stats
        for analysis in upsets:
            upset = analysis.upset

            # By spread bucket
            bucket = upset.spread_bucket
            spread_buckets[bucket].upsets += 1
            if analysis.we_predicted_upset:
                spread_buckets[bucket].we_predicted += 1

            # By conference matchup
            conf = analysis.conference_matchup
            conf_buckets[conf].upsets += 1
            if analysis.we_predicted_upset:
                conf_buckets[conf].we_predicted += 1

            # By week range
            week_range = get_week_range(upset.week)
            week_buckets[week_range].upsets += 1
            if analysis.we_predicted_upset:
                week_buckets[week_range].we_predicted += 1

            # Count anomaly factors
            for factor in analysis.anomaly_factors:
                factor_counts[factor]["total"] += 1
                if analysis.we_predicted_upset:
                    factor_counts[factor]["predicted"] += 1

        # Convert to report format
        report.by_spread_bucket = dict(spread_buckets)
        report.by_conference_matchup = dict(conf_buckets)
        report.by_week_range = dict(week_buckets)

        # Build anomaly factor list
        anomaly_factors = []
        factor_descriptions = {
            "weak_favorite_sos": "Favorite had weak schedule (SOS < 0.45)",
            "small_rating_gap": "Our ratings had teams close (gap < 0.05)",
            "we_predicted_upset": "We predicted the underdog to win",
            "p5_road_at_g5": "P5 team playing road game at G5 venue",
            "late_season": "Late season game (week 10+)",
            "large_spread_upset": "Large spread (10+ point favorite lost)",
            "road_favorite": "Favorite was playing on the road",
        }

        for factor_name, counts in sorted(
            factor_counts.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            if factor_name == "we_predicted_upset":
                continue  # Skip this meta-factor

            anomaly_factors.append(
                AnomalyFactor(
                    name=factor_name,
                    description=factor_descriptions.get(factor_name, factor_name),
                    occurrences=counts["total"],
                    we_predicted_when_present=counts["predicted"],
                )
            )

        report.anomaly_factors = anomaly_factors

        # Highlight upsets we called
        report.upsets_we_called = sorted(
            [u for u in upsets if u.we_predicted_upset],
            key=lambda u: abs(u.upset.spread),
            reverse=True,
        )[:10]  # Top 10 by spread

        # Biggest upsets overall
        report.biggest_upsets = sorted(
            upsets,
            key=lambda u: abs(u.upset.spread),
            reverse=True,
        )[:10]

        return report

    async def analyze_season(
        self,
        season: int,
        our_ratings: list[TeamRating],
        week: int | None = None,
        min_spread: float = 3.0,
    ) -> PatternReport:
        """
        Full analysis of Vegas upsets for a season.

        Args:
            season: Season year
            our_ratings: Our team ratings
            week: Optional specific week to analyze
            min_spread: Minimum spread to consider

        Returns:
            PatternReport with full analysis
        """
        # Fetch data in parallel
        games_task = self.client.fetch_games(season)
        lines_task = self.client.fetch_betting_lines(season, week)
        teams_task = self.client.fetch_teams(season)

        try:
            validators_task = asyncio.gather(
                self.client.fetch_sp_ratings(season),
                self.client.fetch_srs_ratings(season),
                self.client.fetch_elo_ratings(season),
                return_exceptions=True,
            )
        except Exception:
            validators_task = None

        games, lines, teams_list = await asyncio.gather(games_task, lines_task, teams_task)

        # Build teams lookup
        teams = {t.team_id: {"conference": t.conference} for t in teams_list}

        # Build ratings lookup
        ratings_dict = {r.team_id: r for r in our_ratings}

        # Filter games by week if specified
        if week is not None:
            games = [g for g in games if g.week == week]

        # Find upsets
        upsets = await self.find_upsets(games, lines, min_spread)

        # Fetch validators if available
        validators = None
        if validators_task:
            try:
                validator_results = await validators_task
                validators = {}

                # SP+
                if not isinstance(validator_results[0], Exception):
                    validators["sp"] = [
                        ValidatorRating(
                            team_id=r.get("team") or r.get("team_id", ""),
                            source="sp",
                            rating=r.get("rating", 0),
                            rank=r.get("ranking"),
                        )
                        for r in validator_results[0]
                        if r.get("team") or r.get("team_id")
                    ]

                # SRS
                if not isinstance(validator_results[1], Exception):
                    validators["srs"] = [
                        ValidatorRating(
                            team_id=r.get("team") or r.get("team_id", ""),
                            source="srs",
                            rating=r.get("rating", 0),
                            rank=r.get("ranking"),
                        )
                        for r in validator_results[1]
                        if r.get("team") or r.get("team_id")
                    ]

                # Elo
                if not isinstance(validator_results[2], Exception):
                    validators["elo"] = [
                        ValidatorRating(
                            team_id=r.get("team") or r.get("team_id", ""),
                            source="elo",
                            rating=r.get("elo", 0),
                        )
                        for r in validator_results[2]
                        if r.get("team") or r.get("team_id")
                    ]
            except Exception:
                validators = None

        # Analyze each upset
        analyses = [
            self.analyze_upset(upset, ratings_dict, teams, validators) for upset in upsets
        ]

        # Generate report
        return self.generate_pattern_report(
            upsets=analyses,
            all_games_count=len(games),
            games_with_lines_count=len([g for g in games if g.game_id in {l.get("game_id") or l.get("id") for l in lines}]),
            season=season,
            week=week,
        )
