"""Tests for SQLite storage layer."""

import pytest
from datetime import date, datetime, timedelta


class TestStorageInitialization:
    """Tests for storage initialization."""

    @pytest.mark.asyncio
    async def test_storage_creates_tables(self, in_memory_db):
        """Storage.initialize() creates all required tables."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        # Check that tables exist
        async with storage._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in await cursor.fetchall()}

        await storage.close()

        expected_tables = {
            "teams",
            "games",
            "neutral_site_overrides",
            "ratings",
            "historical_rankings",
            "api_cache",
        }
        assert expected_tables.issubset(tables)

    @pytest.mark.asyncio
    async def test_storage_idempotent_init(self, in_memory_db):
        """Multiple initialize() calls don't fail."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()
        await storage.initialize()  # Should not raise
        await storage.close()


class TestTeamOperations:
    """Tests for team CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_team(self, in_memory_db):
        """Save team, retrieve by ID."""
        from src.data.storage import Storage
        from src.data.models import Team

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        team = Team(
            team_id="ohio-state",
            name="Ohio State Buckeyes",
            conference="Big Ten",
            division="fbs",
        )
        await storage.save_teams([team])

        retrieved = await storage.get_team("ohio-state")
        await storage.close()

        assert retrieved is not None
        assert retrieved.team_id == "ohio-state"
        assert retrieved.name == "Ohio State Buckeyes"
        assert retrieved.conference == "Big Ten"
        assert retrieved.division == "fbs"

    @pytest.mark.asyncio
    async def test_get_teams_by_division(self, in_memory_db):
        """Filter teams by FBS/FCS."""
        from src.data.storage import Storage
        from src.data.models import Team

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        teams = [
            Team(team_id="ohio-state", name="Ohio State", conference="Big Ten", division="fbs"),
            Team(team_id="michigan", name="Michigan", conference="Big Ten", division="fbs"),
            Team(team_id="ndsu", name="North Dakota State", conference="MVC", division="fcs"),
        ]
        await storage.save_teams(teams)

        fbs_teams = await storage.get_fbs_teams()
        await storage.close()

        assert len(fbs_teams) == 2
        assert all(t.division == "fbs" for t in fbs_teams)

    @pytest.mark.asyncio
    async def test_upsert_team(self, in_memory_db):
        """Update existing team data."""
        from src.data.storage import Storage
        from src.data.models import Team

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        # Initial save
        team_v1 = Team(
            team_id="usc", name="USC Trojans", conference="Pac-12", division="fbs"
        )
        await storage.save_teams([team_v1])

        # Update (conference changed)
        team_v2 = Team(
            team_id="usc", name="USC Trojans", conference="Big Ten", division="fbs"
        )
        await storage.save_teams([team_v2])

        retrieved = await storage.get_team("usc")
        await storage.close()

        assert retrieved.conference == "Big Ten"

    @pytest.mark.asyncio
    async def test_get_nonexistent_team(self, in_memory_db):
        """Getting nonexistent team returns None."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        result = await storage.get_team("nonexistent")
        await storage.close()

        assert result is None


class TestGameOperations:
    """Tests for game CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_games_batch(self, in_memory_db):
        """Bulk insert games efficiently."""
        from src.data.storage import Storage
        from src.data.models import Game

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        games = [
            Game(
                game_id=i,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id=f"team-{i}",
                away_team_id=f"team-{i+100}",
                home_score=28,
                away_score=14,
            )
            for i in range(100)
        ]
        await storage.save_games(games)

        all_games = await storage.get_games(season=2024)
        await storage.close()

        assert len(all_games) == 100

    @pytest.mark.asyncio
    async def test_get_games_by_season(self, in_memory_db):
        """Retrieve all games for a season."""
        from src.data.storage import Storage
        from src.data.models import Game

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        games = [
            Game(
                game_id=1,
                season=2024,
                week=1,
                game_date=date(2024, 8, 31),
                home_team_id="ohio-state",
                away_team_id="akron",
                home_score=52,
                away_score=6,
            ),
            Game(
                game_id=2,
                season=2023,
                week=1,
                game_date=date(2023, 9, 2),
                home_team_id="ohio-state",
                away_team_id="indiana",
                home_score=23,
                away_score=3,
            ),
        ]
        await storage.save_games(games)

        games_2024 = await storage.get_games(season=2024)
        games_2023 = await storage.get_games(season=2023)
        await storage.close()

        assert len(games_2024) == 1
        assert len(games_2023) == 1
        assert games_2024[0].season == 2024
        assert games_2023[0].season == 2023

    @pytest.mark.asyncio
    async def test_get_games_by_week(self, in_memory_db):
        """Retrieve games up to specific week."""
        from src.data.storage import Storage
        from src.data.models import Game

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        games = [
            Game(
                game_id=i,
                season=2024,
                week=i,
                game_date=date(2024, 8, 31) + timedelta(weeks=i - 1),
                home_team_id="team-a",
                away_team_id="team-b",
                home_score=28,
                away_score=14,
            )
            for i in range(1, 13)
        ]
        await storage.save_games(games)

        games_through_week_6 = await storage.get_games(season=2024, week=6)
        await storage.close()

        assert len(games_through_week_6) == 6
        assert all(g.week <= 6 for g in games_through_week_6)

    @pytest.mark.asyncio
    async def test_get_games_exclude_postseason(self, in_memory_db):
        """Exclude postseason games when requested."""
        from src.data.storage import Storage
        from src.data.models import Game

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        games = [
            Game(
                game_id=1,
                season=2024,
                week=12,
                game_date=date(2024, 11, 30),
                home_team_id="ohio-state",
                away_team_id="michigan",
                home_score=28,
                away_score=20,
                postseason=False,
            ),
            Game(
                game_id=2,
                season=2024,
                week=16,
                game_date=date(2025, 1, 1),
                home_team_id="ohio-state",
                away_team_id="texas",
                home_score=35,
                away_score=28,
                postseason=True,
            ),
        ]
        await storage.save_games(games)

        all_games = await storage.get_games(season=2024, include_postseason=True)
        regular_only = await storage.get_games(season=2024, include_postseason=False)
        await storage.close()

        assert len(all_games) == 2
        assert len(regular_only) == 1
        assert regular_only[0].postseason is False


class TestNeutralSiteOverrides:
    """Tests for neutral site override operations."""

    @pytest.mark.asyncio
    async def test_apply_neutral_site_override(self, in_memory_db):
        """Override changes game's neutral_site flag."""
        from src.data.storage import Storage
        from src.data.models import Game

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        # Save game initially as not neutral
        game = Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="georgia",
            away_team_id="clemson",
            home_score=24,
            away_score=21,
            neutral_site=False,
        )
        await storage.save_games([game])

        # Apply override
        await storage.apply_neutral_override(game_id=1, neutral_site=True)

        # Retrieve and verify
        games = await storage.get_games(season=2024)
        await storage.close()

        assert len(games) == 1
        assert games[0].neutral_site is True


class TestRatingOperations:
    """Tests for rating CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_ratings(self, in_memory_db):
        """Store computed ratings."""
        from src.data.storage import Storage
        from src.data.models import TeamRating

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        ratings = [
            TeamRating(
                team_id="georgia",
                season=2024,
                week=12,
                rating=0.9432,
                rank=1,
                games_played=11,
                wins=11,
                losses=0,
                strength_of_schedule=0.71,
            ),
            TeamRating(
                team_id="ohio-state",
                season=2024,
                week=12,
                rating=0.9187,
                rank=2,
                games_played=11,
                wins=10,
                losses=1,
            ),
        ]
        await storage.save_ratings(ratings)

        retrieved = await storage.get_ratings(season=2024, week=12)
        await storage.close()

        assert len(retrieved) == 2
        assert retrieved[0].rank == 1  # Ordered by rank

    @pytest.mark.asyncio
    async def test_get_latest_ratings(self, in_memory_db):
        """Get most recent ratings for season."""
        from src.data.storage import Storage
        from src.data.models import TeamRating

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        # Save ratings for multiple weeks
        for week in [10, 11, 12]:
            rating = TeamRating(
                team_id="georgia",
                season=2024,
                week=week,
                rating=0.90 + week * 0.01,
                rank=1,
                games_played=week,
                wins=week,
                losses=0,
            )
            await storage.save_ratings([rating])

        latest = await storage.get_ratings(season=2024)
        await storage.close()

        assert len(latest) == 1
        assert latest[0].week == 12
        assert latest[0].rating == pytest.approx(1.02)

    @pytest.mark.asyncio
    async def test_get_ratings_by_week(self, in_memory_db):
        """Get ratings snapshot for specific week."""
        from src.data.storage import Storage
        from src.data.models import TeamRating

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        for week in [10, 11, 12]:
            rating = TeamRating(
                team_id="georgia",
                season=2024,
                week=week,
                rating=0.90 + week * 0.01,
                rank=1,
                games_played=week,
                wins=week,
                losses=0,
            )
            await storage.save_ratings([rating])

        week_11 = await storage.get_ratings(season=2024, week=11)
        await storage.close()

        assert len(week_11) == 1
        assert week_11[0].week == 11


class TestHistoricalRankings:
    """Tests for historical poll ranking operations."""

    @pytest.mark.asyncio
    async def test_save_ap_poll(self, in_memory_db):
        """Store AP Poll rankings."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        rankings = [
            {"team_id": "georgia", "rank": 1, "points": 1550, "first_place_votes": 62},
            {"team_id": "ohio-state", "rank": 2, "points": 1480, "first_place_votes": 1},
        ]
        await storage.save_historical_rankings(
            poll_type="ap", season=2024, week=12, rankings=rankings
        )

        retrieved = await storage.get_historical_rankings(
            poll_type="ap", season=2024, week=12
        )
        await storage.close()

        assert len(retrieved) == 2
        assert retrieved[0]["rank"] == 1

    @pytest.mark.asyncio
    async def test_save_cfp_rankings(self, in_memory_db):
        """Store CFP rankings."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        rankings = [
            {"team_id": "oregon", "rank": 1},
            {"team_id": "ohio-state", "rank": 2},
            {"team_id": "texas", "rank": 3},
        ]
        await storage.save_historical_rankings(
            poll_type="cfp", season=2024, week=12, rankings=rankings
        )

        retrieved = await storage.get_historical_rankings(
            poll_type="cfp", season=2024, week=12
        )
        await storage.close()

        assert len(retrieved) == 3

    @pytest.mark.asyncio
    async def test_get_historical_rankings_empty(self, in_memory_db):
        """Getting nonexistent rankings returns empty list."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        result = await storage.get_historical_rankings(
            poll_type="ap", season=2024, week=1
        )
        await storage.close()

        assert result == []


class TestCacheOperations:
    """Tests for API response caching."""

    @pytest.mark.asyncio
    async def test_cache_api_response(self, in_memory_db):
        """Cache and retrieve API responses."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        data = {"teams": [{"id": 1, "name": "Ohio State"}]}
        await storage.set_cached("teams_fbs_2024", data, ttl_hours=24)

        retrieved = await storage.get_cached("teams_fbs_2024")
        await storage.close()

        assert retrieved == data

    @pytest.mark.asyncio
    async def test_cache_expiry(self, in_memory_db):
        """Expired cache entries not returned."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        data = {"teams": []}
        # Set with very short TTL
        await storage.set_cached("expired_key", data, ttl_hours=0)

        # Should return None since TTL is 0
        retrieved = await storage.get_cached("expired_key")
        await storage.close()

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_cache_nonexistent_key(self, in_memory_db):
        """Getting nonexistent cache key returns None."""
        from src.data.storage import Storage

        storage = Storage(db_path=in_memory_db)
        await storage.initialize()

        result = await storage.get_cached("nonexistent")
        await storage.close()

        assert result is None
