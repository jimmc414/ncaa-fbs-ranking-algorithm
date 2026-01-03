"""SQLite storage layer for NCAA ranking data."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from src.data.models import Game, Team, TeamRating


class Storage:
    """Async SQLite storage for teams, games, ratings, and cache."""

    def __init__(self, db_path: str | Path = "data/ncaa_rankings.db"):
        self.db_path = str(db_path)
        self._connection: aiosqlite.Connection | None = None

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get or create a database connection."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
            await self._connection.execute("PRAGMA foreign_keys = ON")
        yield self._connection

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with self._get_connection() as conn:
            await conn.executescript(
                """
                -- Teams table
                CREATE TABLE IF NOT EXISTS teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    conference TEXT,
                    division TEXT CHECK(division IN ('fbs', 'fcs')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Games table
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    game_date DATE NOT NULL,
                    home_team_id TEXT NOT NULL,
                    away_team_id TEXT NOT NULL,
                    home_score INTEGER NOT NULL CHECK(home_score >= 0),
                    away_score INTEGER NOT NULL CHECK(away_score >= 0),
                    neutral_site BOOLEAN DEFAULT FALSE,
                    postseason BOOLEAN DEFAULT FALSE,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Neutral site overrides
                CREATE TABLE IF NOT EXISTS neutral_site_overrides (
                    game_id INTEGER PRIMARY KEY,
                    correct_neutral_site BOOLEAN NOT NULL,
                    reason TEXT
                );

                -- Computed ratings
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    games_played INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    losses INTEGER NOT NULL,
                    strength_of_schedule REAL,
                    strength_of_victory REAL,
                    average_game_grade REAL,
                    iterations_to_converge INTEGER,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_id, season, week)
                );

                -- Historical poll data
                CREATE TABLE IF NOT EXISTS historical_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    poll_type TEXT NOT NULL CHECK(poll_type IN ('ap', 'cfp', 'coaches')),
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    team_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    points INTEGER,
                    first_place_votes INTEGER,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(poll_type, season, week, team_id)
                );

                -- API cache
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                );

                -- Indices
                CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
                CREATE INDEX IF NOT EXISTS idx_ratings_season_week ON ratings(season, week);
                CREATE INDEX IF NOT EXISTS idx_historical_poll_season
                    ON historical_rankings(poll_type, season, week);
                """
            )
            await conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    # ==================== Team Operations ====================

    async def save_teams(self, teams: list[Team]) -> None:
        """Save or update teams."""
        async with self._get_connection() as conn:
            await conn.executemany(
                """
                INSERT INTO teams (team_id, name, conference, division, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(team_id) DO UPDATE SET
                    name = excluded.name,
                    conference = excluded.conference,
                    division = excluded.division,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [(t.team_id, t.name, t.conference, t.division) for t in teams],
            )
            await conn.commit()

    async def get_team(self, team_id: str) -> Team | None:
        """Get a team by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT team_id, name, conference, division FROM teams WHERE team_id = ?",
                (team_id,),
            )
            row = await cursor.fetchone()
            if row:
                return Team(
                    team_id=row["team_id"],
                    name=row["name"],
                    conference=row["conference"],
                    division=row["division"],
                )
            return None

    async def get_fbs_teams(self) -> list[Team]:
        """Get all FBS teams."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT team_id, name, conference, division FROM teams WHERE division = 'fbs'"
            )
            rows = await cursor.fetchall()
            return [
                Team(
                    team_id=row["team_id"],
                    name=row["name"],
                    conference=row["conference"],
                    division=row["division"],
                )
                for row in rows
            ]

    # ==================== Game Operations ====================

    async def save_games(self, games: list[Game]) -> None:
        """Save or update games."""
        async with self._get_connection() as conn:
            await conn.executemany(
                """
                INSERT INTO games (
                    game_id, season, week, game_date, home_team_id, away_team_id,
                    home_score, away_score, neutral_site, postseason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    season = excluded.season,
                    week = excluded.week,
                    game_date = excluded.game_date,
                    home_team_id = excluded.home_team_id,
                    away_team_id = excluded.away_team_id,
                    home_score = excluded.home_score,
                    away_score = excluded.away_score,
                    neutral_site = excluded.neutral_site,
                    postseason = excluded.postseason
                """,
                [
                    (
                        g.game_id,
                        g.season,
                        g.week,
                        g.game_date.isoformat(),
                        g.home_team_id,
                        g.away_team_id,
                        g.home_score,
                        g.away_score,
                        g.neutral_site,
                        g.postseason,
                    )
                    for g in games
                ],
            )
            await conn.commit()

    async def get_games(
        self,
        season: int,
        week: int | None = None,
        include_postseason: bool = True,
    ) -> list[Game]:
        """Get games for a season, optionally filtered by week."""
        async with self._get_connection() as conn:
            query = """
                SELECT
                    g.game_id, g.season, g.week, g.game_date,
                    g.home_team_id, g.away_team_id,
                    g.home_score, g.away_score,
                    COALESCE(o.correct_neutral_site, g.neutral_site) as neutral_site,
                    g.postseason
                FROM games g
                LEFT JOIN neutral_site_overrides o ON g.game_id = o.game_id
                WHERE g.season = ?
            """
            params: list = [season]

            if week is not None:
                query += " AND g.week <= ?"
                params.append(week)

            if not include_postseason:
                query += " AND g.postseason = FALSE"

            query += " ORDER BY g.week, g.game_id"

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [
                Game(
                    game_id=row["game_id"],
                    season=row["season"],
                    week=row["week"],
                    game_date=row["game_date"],
                    home_team_id=row["home_team_id"],
                    away_team_id=row["away_team_id"],
                    home_score=row["home_score"],
                    away_score=row["away_score"],
                    neutral_site=bool(row["neutral_site"]),
                    postseason=bool(row["postseason"]),
                )
                for row in rows
            ]

    async def apply_neutral_override(
        self, game_id: int, neutral_site: bool, reason: str | None = None
    ) -> None:
        """Apply a neutral site override to a game."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO neutral_site_overrides (game_id, correct_neutral_site, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    correct_neutral_site = excluded.correct_neutral_site,
                    reason = excluded.reason
                """,
                (game_id, neutral_site, reason),
            )
            await conn.commit()

    # ==================== Rating Operations ====================

    async def save_ratings(self, ratings: list[TeamRating]) -> None:
        """Save computed ratings."""
        async with self._get_connection() as conn:
            await conn.executemany(
                """
                INSERT INTO ratings (
                    team_id, season, week, rating, rank, games_played,
                    wins, losses, strength_of_schedule, strength_of_victory,
                    average_game_grade
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, season, week) DO UPDATE SET
                    rating = excluded.rating,
                    rank = excluded.rank,
                    games_played = excluded.games_played,
                    wins = excluded.wins,
                    losses = excluded.losses,
                    strength_of_schedule = excluded.strength_of_schedule,
                    strength_of_victory = excluded.strength_of_victory,
                    average_game_grade = excluded.average_game_grade,
                    computed_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        r.team_id,
                        r.season,
                        r.week,
                        r.rating,
                        r.rank,
                        r.games_played,
                        r.wins,
                        r.losses,
                        r.strength_of_schedule,
                        r.strength_of_victory,
                        r.average_game_grade,
                    )
                    for r in ratings
                ],
            )
            await conn.commit()

    async def get_ratings(
        self, season: int, week: int | None = None
    ) -> list[TeamRating]:
        """Get ratings for a season. If week is None, get latest week."""
        async with self._get_connection() as conn:
            if week is None:
                # Get latest week with ratings
                cursor = await conn.execute(
                    "SELECT MAX(week) FROM ratings WHERE season = ?", (season,)
                )
                row = await cursor.fetchone()
                if row[0] is None:
                    return []
                week = row[0]

            cursor = await conn.execute(
                """
                SELECT team_id, season, week, rating, rank, games_played,
                       wins, losses, strength_of_schedule, strength_of_victory,
                       average_game_grade
                FROM ratings
                WHERE season = ? AND week = ?
                ORDER BY rank
                """,
                (season, week),
            )
            rows = await cursor.fetchall()

            return [
                TeamRating(
                    team_id=row["team_id"],
                    season=row["season"],
                    week=row["week"],
                    rating=row["rating"],
                    rank=row["rank"],
                    games_played=row["games_played"],
                    wins=row["wins"],
                    losses=row["losses"],
                    strength_of_schedule=row["strength_of_schedule"],
                    strength_of_victory=row["strength_of_victory"],
                    average_game_grade=row["average_game_grade"],
                )
                for row in rows
            ]

    # ==================== Historical Rankings ====================

    async def save_historical_rankings(
        self,
        poll_type: str,
        season: int,
        week: int,
        rankings: list[dict],
    ) -> None:
        """Save historical poll rankings."""
        async with self._get_connection() as conn:
            await conn.executemany(
                """
                INSERT INTO historical_rankings (
                    poll_type, season, week, team_id, rank, points, first_place_votes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(poll_type, season, week, team_id) DO UPDATE SET
                    rank = excluded.rank,
                    points = excluded.points,
                    first_place_votes = excluded.first_place_votes
                """,
                [
                    (
                        poll_type,
                        season,
                        week,
                        r["team_id"],
                        r["rank"],
                        r.get("points"),
                        r.get("first_place_votes"),
                    )
                    for r in rankings
                ],
            )
            await conn.commit()

    async def get_historical_rankings(
        self, poll_type: str, season: int, week: int
    ) -> list[dict]:
        """Get historical rankings for a poll/season/week."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT team_id, rank, points, first_place_votes
                FROM historical_rankings
                WHERE poll_type = ? AND season = ? AND week = ?
                ORDER BY rank
                """,
                (poll_type, season, week),
            )
            rows = await cursor.fetchall()

            return [
                {
                    "team_id": row["team_id"],
                    "rank": row["rank"],
                    "points": row["points"],
                    "first_place_votes": row["first_place_votes"],
                }
                for row in rows
            ]

    # ==================== Cache Operations ====================

    async def set_cached(
        self, key: str, data: dict, ttl_hours: int = 24
    ) -> None:
        """Cache an API response."""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO api_cache (cache_key, response_data, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_data = excluded.response_data,
                    fetched_at = CURRENT_TIMESTAMP,
                    expires_at = excluded.expires_at
                """,
                (key, json.dumps(data), expires_at.isoformat()),
            )
            await conn.commit()

    async def get_cached(self, key: str) -> dict | None:
        """Get a cached API response if not expired."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT response_data, expires_at
                FROM api_cache
                WHERE cache_key = ?
                """,
                (key,),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            # Check expiry
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now() >= expires_at:
                return None

            return json.loads(row["response_data"])
