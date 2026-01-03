"""Shared pytest fixtures for NCAA ranking tests."""

import pytest
from datetime import date
from pathlib import Path
import json


@pytest.fixture
def sample_teams():
    """Sample team data for testing."""
    from src.data.models import Team

    return [
        Team(team_id="ohio-state", name="Ohio State Buckeyes", conference="Big Ten", division="fbs"),
        Team(team_id="michigan", name="Michigan Wolverines", conference="Big Ten", division="fbs"),
        Team(team_id="alabama", name="Alabama Crimson Tide", conference="SEC", division="fbs"),
        Team(team_id="georgia", name="Georgia Bulldogs", conference="SEC", division="fbs"),
    ]


@pytest.fixture
def sample_games():
    """Sample game data for testing."""
    from src.data.models import Game

    return [
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
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="michigan",
            away_team_id="fresno-state",
            home_score=30,
            away_score=10,
        ),
    ]


@pytest.fixture
def four_team_round_robin():
    """
    Canonical test case: 4-team round robin.

    A beats B, C, D
    B beats C, D
    C beats D
    D loses all

    Expected order: A > B > C > D
    """
    from src.data.models import Game

    return [
        # Week 1: A beats B, C beats D
        Game(
            game_id=1,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="A",
            away_team_id="B",
            home_score=28,
            away_score=14,
        ),
        Game(
            game_id=2,
            season=2024,
            week=1,
            game_date=date(2024, 8, 31),
            home_team_id="C",
            away_team_id="D",
            home_score=21,
            away_score=14,
        ),
        # Week 2: A beats C, B beats D
        Game(
            game_id=3,
            season=2024,
            week=2,
            game_date=date(2024, 9, 7),
            home_team_id="A",
            away_team_id="C",
            home_score=35,
            away_score=17,
        ),
        Game(
            game_id=4,
            season=2024,
            week=2,
            game_date=date(2024, 9, 7),
            home_team_id="B",
            away_team_id="D",
            home_score=24,
            away_score=10,
        ),
        # Week 3: A beats D, B beats C
        Game(
            game_id=5,
            season=2024,
            week=3,
            game_date=date(2024, 9, 14),
            home_team_id="A",
            away_team_id="D",
            home_score=42,
            away_score=7,
        ),
        Game(
            game_id=6,
            season=2024,
            week=3,
            game_date=date(2024, 9, 14),
            home_team_id="B",
            away_team_id="C",
            home_score=17,
            away_score=14,
        ),
    ]


@pytest.fixture
def algorithm_config():
    """Default algorithm configuration for testing."""
    from src.data.models import AlgorithmConfig

    return AlgorithmConfig()


@pytest.fixture
def in_memory_db():
    """In-memory SQLite database path for isolated testing."""
    return ":memory:"


@pytest.fixture
async def storage(in_memory_db):
    """Storage instance with clean in-memory database."""
    from src.data.storage import Storage

    storage = Storage(db_path=in_memory_db)
    await storage.initialize()
    yield storage
    await storage.close()
