"""Tests for CollegeFootballData API client."""

import time
from datetime import date
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from src.data.models import Game, Team


class TestResponseParsing:
    """Tests for parsing API responses into Pydantic models."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_teams_response(self):
        """Parse teams API response into list[Team]."""
        from src.data.client import CFBDataClient

        # Sample API response structure from CollegeFootballData
        mock_response = [
            {
                "id": 57,
                "school": "Ohio State",
                "mascot": "Buckeyes",
                "abbreviation": "OSU",
                "conference": "Big Ten",
                "classification": "fbs",
            },
            {
                "id": 130,
                "school": "Michigan",
                "mascot": "Wolverines",
                "abbreviation": "MICH",
                "conference": "Big Ten",
                "classification": "fbs",
            },
        ]

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        teams = await client.fetch_teams()

        assert len(teams) == 2
        assert isinstance(teams[0], Team)
        assert teams[0].name == "Ohio State"
        assert teams[0].conference == "Big Ten"
        assert teams[0].division == "fbs"

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_teams_fbs_only_filter(self):
        """Filter to FBS teams only when fbs_only=True."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 57,
                "school": "Ohio State",
                "mascot": "Buckeyes",
                "abbreviation": "OSU",
                "conference": "Big Ten",
                "classification": "fbs",
            },
            {
                "id": 9999,
                "school": "Youngstown State",
                "mascot": "Penguins",
                "abbreviation": "YSU",
                "conference": "Missouri Valley",
                "classification": "fcs",
            },
        ]

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        teams = await client.fetch_teams(fbs_only=True)

        assert len(teams) == 1
        assert teams[0].division == "fbs"

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_teams_include_fcs(self):
        """Include FCS teams when fbs_only=False."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 57,
                "school": "Ohio State",
                "mascot": "Buckeyes",
                "abbreviation": "OSU",
                "conference": "Big Ten",
                "classification": "fbs",
            },
            {
                "id": 9999,
                "school": "Youngstown State",
                "mascot": "Penguins",
                "abbreviation": "YSU",
                "conference": "Missouri Valley",
                "classification": "fcs",
            },
        ]

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        teams = await client.fetch_teams(fbs_only=False)

        assert len(teams) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_games_response(self):
        """Parse games API response into list[Game]."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 401628401,
                "season": 2024,
                "week": 1,
                "startDate": "2024-08-31T00:00:00.000Z",
                "homeId": 57,
                "homeTeam": "Ohio State",
                "homePoints": 52,
                "homeClassification": "fbs",
                "awayId": 2006,
                "awayTeam": "Akron",
                "awayPoints": 6,
                "awayClassification": "fbs",
                "neutralSite": False,
                "seasonType": "regular",
            },
            {
                "id": 401628402,
                "season": 2024,
                "week": 1,
                "startDate": "2024-08-31T00:00:00.000Z",
                "homeId": 130,
                "homeTeam": "Michigan",
                "homePoints": 30,
                "homeClassification": "fbs",
                "awayId": 278,
                "awayTeam": "Fresno State",
                "awayPoints": 10,
                "awayClassification": "fbs",
                "neutralSite": False,
                "seasonType": "regular",
            },
        ]

        respx.get("https://api.collegefootballdata.com/games").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        games = await client.fetch_games(season=2024)

        assert len(games) == 2
        assert isinstance(games[0], Game)
        assert games[0].game_id == 401628401
        assert games[0].season == 2024
        assert games[0].week == 1
        assert games[0].home_score == 52
        assert games[0].away_score == 6
        assert games[0].postseason is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_games_with_postseason(self):
        """Identify postseason games correctly."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 401628500,
                "season": 2024,
                "week": 16,
                "startDate": "2024-12-28T00:00:00.000Z",
                "homeId": 57,
                "homeTeam": "Ohio State",
                "homePoints": 42,
                "homeClassification": "fbs",
                "awayId": 130,
                "awayTeam": "Michigan",
                "awayPoints": 35,
                "awayClassification": "fbs",
                "neutralSite": True,
                "seasonType": "postseason",
            },
        ]

        respx.get("https://api.collegefootballdata.com/games").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        games = await client.fetch_games(season=2024, season_type="postseason")

        assert len(games) == 1
        assert games[0].postseason is True
        assert games[0].neutral_site is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_games_team_id_normalization(self):
        """Team IDs are normalized to lowercase with hyphens."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 401628401,
                "season": 2024,
                "week": 1,
                "startDate": "2024-08-31T00:00:00.000Z",
                "homeId": 57,
                "homeTeam": "Ohio State",
                "homePoints": 52,
                "homeClassification": "fbs",
                "awayId": 2006,
                "awayTeam": "Akron",
                "awayPoints": 6,
                "awayClassification": "fbs",
                "neutralSite": False,
                "seasonType": "regular",
            },
        ]

        respx.get("https://api.collegefootballdata.com/games").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        games = await client.fetch_games(season=2024)

        assert games[0].home_team_id == "ohio-state"
        assert games[0].away_team_id == "akron"

    @respx.mock
    @pytest.mark.asyncio
    async def test_parse_rankings_response(self):
        """Parse rankings API response into list[dict]."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "season": 2024,
                "week": 12,
                "seasonType": "regular",
                "polls": [
                    {
                        "poll": "AP Top 25",
                        "ranks": [
                            {"rank": 1, "school": "Oregon", "points": 1550, "firstPlaceVotes": 61},
                            {"rank": 2, "school": "Ohio State", "points": 1482, "firstPlaceVotes": 1},
                            {"rank": 3, "school": "Texas", "points": 1430, "firstPlaceVotes": 0},
                        ],
                    }
                ],
            }
        ]

        respx.get("https://api.collegefootballdata.com/rankings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        rankings = await client.fetch_rankings(season=2024, week=12, poll_type="ap")

        assert len(rankings) == 3
        assert rankings[0]["rank"] == 1
        assert rankings[0]["team_id"] == "oregon"
        assert rankings[0]["points"] == 1550
        assert rankings[0]["first_place_votes"] == 61

    @respx.mock
    @pytest.mark.asyncio
    async def test_handle_missing_optional_fields(self):
        """Handle missing optional fields gracefully."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 57,
                "school": "Ohio State",
                "mascot": "Buckeyes",
                "classification": "fbs",
                # Missing: conference, abbreviation
            },
        ]

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key")
        teams = await client.fetch_teams()

        assert len(teams) == 1
        assert teams[0].conference is None


class TestRateLimiting:
    """Tests for rate limiting (1000 requests/hour)."""

    @pytest.mark.asyncio
    async def test_tracks_request_count(self):
        """Client tracks request count per hour."""
        from src.data.client import CFBDataClient

        client = CFBDataClient(api_key="test-key")
        assert client.request_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_increments_request_count(self):
        """Each API call increments request count."""
        from src.data.client import CFBDataClient

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=[])
        )

        client = CFBDataClient(api_key="test-key")
        await client.fetch_teams()
        await client.fetch_teams()

        assert client.request_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_resets_after_hour(self):
        """Request count resets after an hour."""
        from src.data.client import CFBDataClient

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=[])
        )

        client = CFBDataClient(api_key="test-key")
        await client.fetch_teams()
        assert client.request_count == 1

        # Simulate time passing (1 hour + 1 second)
        client._window_start = time.time() - 3601
        await client.fetch_teams()

        # Should have reset and this is the first request of new window
        assert client.request_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_rate_limit_exceeded(self):
        """Raises exception when approaching rate limit."""
        from src.data.client import CFBDataClient, RateLimitError

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=[])
        )

        client = CFBDataClient(api_key="test-key")
        # Simulate being at the rate limit
        client._request_count = 999
        client._window_start = time.time()

        # One more should still work
        await client.fetch_teams()
        assert client.request_count == 1000

        # Next one should raise
        with pytest.raises(RateLimitError) as exc_info:
            await client.fetch_teams()

        assert "rate limit" in str(exc_info.value).lower()


class TestRetryLogic:
    """Tests for retry with exponential backoff on server errors."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        """Retry on 500 Internal Server Error."""
        from src.data.client import CFBDataClient

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.side_effect = [
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(200, json=[{"id": 57, "school": "Ohio State", "mascot": "Buckeyes", "classification": "fbs"}]),
        ]

        client = CFBDataClient(api_key="test-key")
        teams = await client.fetch_teams()

        assert len(teams) == 1
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_502_503_504(self):
        """Retry on 502, 503, 504 errors."""
        from src.data.client import CFBDataClient

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.side_effect = [
            httpx.Response(502, json={"error": "Bad Gateway"}),
            httpx.Response(503, json={"error": "Service Unavailable"}),
            httpx.Response(504, json={"error": "Gateway Timeout"}),
            httpx.Response(200, json=[]),
        ]

        client = CFBDataClient(api_key="test-key")
        teams = await client.fetch_teams()

        assert route.call_count == 4

    @respx.mock
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Raise after max retries exceeded."""
        from src.data.client import CFBDataClient, APIError

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.side_effect = [
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(500, json={"error": "Internal Server Error"}),  # 4th attempt
        ]

        client = CFBDataClient(api_key="test-key", max_retries=3)

        with pytest.raises(APIError) as exc_info:
            await client.fetch_teams()

        assert "500" in str(exc_info.value) or "retry" in str(exc_info.value).lower()
        assert route.call_count == 4  # Initial + 3 retries

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        """Don't retry on 4xx client errors."""
        from src.data.client import CFBDataClient, APIError

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))

        client = CFBDataClient(api_key="test-key")

        with pytest.raises(APIError) as exc_info:
            await client.fetch_teams()

        assert "401" in str(exc_info.value)
        assert route.call_count == 1  # No retries

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_retry_on_404(self):
        """Don't retry on 404 Not Found."""
        from src.data.client import CFBDataClient, APIError

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.mock(return_value=httpx.Response(404, json={"error": "Not Found"}))

        client = CFBDataClient(api_key="test-key")

        with pytest.raises(APIError):
            await client.fetch_teams()

        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Verify exponential backoff between retries."""
        from src.data.client import CFBDataClient

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.side_effect = [
            httpx.Response(500, json={"error": "Error"}),
            httpx.Response(500, json={"error": "Error"}),
            httpx.Response(200, json=[]),
        ]

        client = CFBDataClient(api_key="test-key", base_delay=0.01)

        start = time.time()
        await client.fetch_teams()
        elapsed = time.time() - start

        # Should have waited: 0.01 + 0.02 = 0.03 seconds minimum
        # Allow some tolerance for test execution
        assert elapsed >= 0.02


class TestCacheIntegration:
    """Tests for cache integration with Storage layer."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_checks_cache_before_api_call(self, storage):
        """Check cache before making API call."""
        from src.data.client import CFBDataClient

        # Pre-populate cache
        await storage.set_cached(
            "teams_fbs_only",
            [{"team_id": "ohio-state", "name": "Ohio State", "conference": "Big Ten", "division": "fbs"}],
        )

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.mock(return_value=httpx.Response(200, json=[]))

        client = CFBDataClient(api_key="test-key", storage=storage)
        teams = await client.fetch_teams()

        # Should use cached data, not make API call
        assert route.call_count == 0
        assert len(teams) == 1
        assert teams[0].team_id == "ohio-state"

    @respx.mock
    @pytest.mark.asyncio
    async def test_stores_response_in_cache(self, storage):
        """Store API response in cache after fetch."""
        from src.data.client import CFBDataClient

        mock_response = [
            {"id": 57, "school": "Ohio State", "mascot": "Buckeyes", "conference": "Big Ten", "classification": "fbs"},
        ]

        respx.get("https://api.collegefootballdata.com/teams").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = CFBDataClient(api_key="test-key", storage=storage)
        await client.fetch_teams()

        # Check cache was populated
        cached = await storage.get_cached("teams_fbs_only")
        assert cached is not None
        assert len(cached) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_respects_cache_ttl(self, storage):
        """Expired cache entries are not used."""
        from src.data.client import CFBDataClient

        # Pre-populate cache with expired entry (TTL of -1 hours = already expired)
        await storage.set_cached(
            "teams_fbs_only",
            [{"team_id": "old-data", "name": "Old", "conference": None, "division": "fbs"}],
            ttl_hours=-1,  # Already expired
        )

        mock_response = [
            {"id": 57, "school": "Ohio State", "mascot": "Buckeyes", "conference": "Big Ten", "classification": "fbs"},
        ]

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.mock(return_value=httpx.Response(200, json=mock_response))

        client = CFBDataClient(api_key="test-key", storage=storage)
        teams = await client.fetch_teams()

        # Should fetch fresh data since cache expired
        assert route.call_count == 1
        assert teams[0].team_id == "ohio-state"

    @respx.mock
    @pytest.mark.asyncio
    async def test_games_cache_by_season(self, storage):
        """Games are cached by season."""
        from src.data.client import CFBDataClient

        mock_response = [
            {
                "id": 401628401,
                "season": 2024,
                "week": 1,
                "start_date": "2024-08-31T00:00:00.000Z",
                "home_id": 57,
                "home_team": "Ohio State",
                "home_points": 52,
                "away_id": 2006,
                "away_team": "Akron",
                "away_points": 6,
                "neutral_site": False,
                "season_type": "regular",
            },
        ]

        route = respx.get("https://api.collegefootballdata.com/games")
        route.mock(return_value=httpx.Response(200, json=mock_response))

        client = CFBDataClient(api_key="test-key", storage=storage)
        await client.fetch_games(season=2024)

        # Check cache key includes season
        cached = await storage.get_cached("games_2024_both")
        assert cached is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_works_without_storage(self):
        """Client works without storage (no caching)."""
        from src.data.client import CFBDataClient

        mock_response = [
            {"id": 57, "school": "Ohio State", "mascot": "Buckeyes", "conference": "Big Ten", "classification": "fbs"},
        ]

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.mock(return_value=httpx.Response(200, json=mock_response))

        client = CFBDataClient(api_key="test-key")  # No storage
        teams = await client.fetch_teams()
        teams2 = await client.fetch_teams()

        # Should make API call each time since no cache
        assert route.call_count == 2


class TestAuthentication:
    """Tests for API authentication."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_bearer_token_in_header(self):
        """API key is sent as Bearer token in Authorization header."""
        from src.data.client import CFBDataClient

        route = respx.get("https://api.collegefootballdata.com/teams")
        route.mock(return_value=httpx.Response(200, json=[]))

        client = CFBDataClient(api_key="my-secret-key")
        await client.fetch_teams()

        # Verify Authorization header
        request = route.calls[0].request
        assert request.headers.get("Authorization") == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_loads_api_key_from_env(self):
        """Can load API key from environment variable."""
        from src.data.client import CFBDataClient

        with patch.dict("os.environ", {"CFBD_API_KEY": "env-key"}):
            client = CFBDataClient.from_env()
            assert client.api_key == "env-key"

    @pytest.mark.asyncio
    async def test_raises_if_no_api_key(self):
        """Raises if no API key provided and not in env."""
        from src.data.client import CFBDataClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                CFBDataClient.from_env()

            assert "cfbd_api_key" in str(exc_info.value).lower()
