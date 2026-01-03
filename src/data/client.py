"""CollegeFootballData API client with caching and rate limiting."""

import asyncio
import os
import time
from datetime import date

import httpx

from src.data.models import Game, Team
from src.data.storage import Storage


class APIError(Exception):
    """Raised when API request fails."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"API Error {status_code}: {message}")


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class CFBDataClient:
    """Client for CollegeFootballData.com API."""

    BASE_URL = "https://api.collegefootballdata.com"
    RATE_LIMIT = 1000  # requests per hour

    def __init__(
        self,
        api_key: str,
        storage: Storage | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Initialize the API client.

        Args:
            api_key: CollegeFootballData API key
            storage: Optional Storage instance for caching
            max_retries: Maximum number of retry attempts for 5xx errors
            base_delay: Base delay in seconds for exponential backoff
        """
        self.api_key = api_key
        self.storage = storage
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Rate limiting state
        self._request_count = 0
        self._window_start = time.time()

    @classmethod
    def from_env(cls, storage: Storage | None = None) -> "CFBDataClient":
        """
        Create client from environment variable.

        Args:
            storage: Optional Storage instance for caching

        Returns:
            CFBDataClient instance

        Raises:
            ValueError: If CFBD_API_KEY not set in environment
        """
        api_key = os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise ValueError("CFBD_API_KEY environment variable not set. Get a key at https://collegefootballdata.com")
        return cls(api_key=api_key, storage=storage)

    @property
    def request_count(self) -> int:
        """Current request count in the rate limit window."""
        return self._request_count

    def _check_rate_limit(self) -> None:
        """
        Check and update rate limiting state.

        Raises:
            RateLimitError: If rate limit would be exceeded
        """
        now = time.time()

        # Reset window if hour has passed
        if now - self._window_start >= 3600:
            self._request_count = 0
            self._window_start = now

        # Check if we'd exceed the limit
        if self._request_count >= self.RATE_LIMIT:
            seconds_until_reset = 3600 - (now - self._window_start)
            raise RateLimitError(
                f"Rate limit of {self.RATE_LIMIT} requests/hour exceeded. "
                f"Resets in {seconds_until_reset:.0f} seconds."
            )

    async def _make_request(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> list:
        """
        Make an API request with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/teams")
            params: Optional query parameters

        Returns:
            JSON response as list

        Raises:
            APIError: If request fails after retries
            RateLimitError: If rate limit exceeded
        """
        self._check_rate_limit()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        attempt = 0
        last_error = None

        async with httpx.AsyncClient() as client:
            while attempt <= self.max_retries:
                try:
                    response = await client.get(url, params=params, headers=headers)

                    # Success
                    if response.status_code == 200:
                        self._request_count += 1
                        return response.json()

                    # Client error - don't retry
                    if 400 <= response.status_code < 500:
                        raise APIError(response.status_code, response.text)

                    # Server error - retry with backoff
                    if response.status_code >= 500:
                        last_error = APIError(response.status_code, response.text)
                        if attempt < self.max_retries:
                            delay = self.base_delay * (2 ** attempt)
                            await asyncio.sleep(delay)
                        attempt += 1
                        continue

                except httpx.RequestError as e:
                    last_error = APIError(0, str(e))
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    attempt += 1
                    continue

        # All retries exhausted
        if last_error:
            raise last_error
        raise APIError(0, "Unknown error")

    @staticmethod
    def _normalize_team_id(team_name: str) -> str:
        """
        Normalize team name to ID format.

        Args:
            team_name: Team name (e.g., "Ohio State")

        Returns:
            Normalized ID (e.g., "ohio-state")
        """
        return team_name.lower().replace(" ", "-").replace("&", "and")

    async def fetch_teams(self, fbs_only: bool = True) -> list[Team]:
        """
        Fetch teams from the API.

        Args:
            fbs_only: If True, filter to FBS teams only

        Returns:
            List of Team objects
        """
        cache_key = f"teams_{'fbs_only' if fbs_only else 'all'}"

        # Check cache first
        if self.storage:
            cached = await self.storage.get_cached(cache_key)
            if cached:
                return [Team(**t) for t in cached]

        # Fetch from API
        data = await self._make_request("/teams")

        # Parse and filter
        teams = []
        for item in data:
            classification = item.get("classification", "").lower()
            if fbs_only and classification != "fbs":
                continue

            team = Team(
                team_id=self._normalize_team_id(item["school"]),
                name=item["school"],
                conference=item.get("conference"),
                division="fbs" if classification == "fbs" else "fcs",
            )
            teams.append(team)

        # Cache the result
        if self.storage:
            await self.storage.set_cached(
                cache_key,
                [t.model_dump() for t in teams],
                ttl_hours=24,
            )

        return teams

    async def fetch_games(
        self,
        season: int,
        season_type: str = "both",
    ) -> list[Game]:
        """
        Fetch games for a season.

        Args:
            season: Season year (e.g., 2024)
            season_type: "regular", "postseason", or "both"

        Returns:
            List of Game objects
        """
        cache_key = f"games_{season}_{season_type}"

        # Check cache first
        if self.storage:
            cached = await self.storage.get_cached(cache_key)
            if cached:
                return [Game(**g) for g in cached]

        # Build params
        params = {"year": season}
        if season_type == "regular":
            params["seasonType"] = "regular"
        elif season_type == "postseason":
            params["seasonType"] = "postseason"
        # "both" = no seasonType filter

        # Fetch from API
        data = await self._make_request("/games", params=params)

        # Parse games
        games = []
        for item in data:
            # Skip games without scores (not yet played)
            if item.get("homePoints") is None or item.get("awayPoints") is None:
                continue

            # Filter for FBS games only (at least one team must be FBS)
            home_class = (item.get("homeClassification") or "").lower()
            away_class = (item.get("awayClassification") or "").lower()
            if home_class != "fbs" and away_class != "fbs":
                continue

            # Parse date (API uses camelCase: startDate)
            date_str = item.get("startDate", "")
            if date_str:
                game_date = date.fromisoformat(date_str[:10])
            else:
                continue

            is_postseason = item.get("seasonType", "").lower() == "postseason"

            game = Game(
                game_id=item["id"],
                season=item["season"],
                week=item["week"],
                game_date=game_date,
                home_team_id=self._normalize_team_id(item["homeTeam"]),
                away_team_id=self._normalize_team_id(item["awayTeam"]),
                home_score=item["homePoints"],
                away_score=item["awayPoints"],
                neutral_site=item.get("neutralSite", False),
                postseason=is_postseason,
            )
            games.append(game)

        # Cache the result
        if self.storage:
            await self.storage.set_cached(
                cache_key,
                [g.model_dump(mode="json") for g in games],
                ttl_hours=1,  # Games can update more frequently
            )

        return games

    async def fetch_rankings(
        self,
        season: int,
        week: int,
        poll_type: str = "ap",
    ) -> list[dict]:
        """
        Fetch historical poll rankings.

        Args:
            season: Season year
            week: Week number
            poll_type: "ap", "cfp", or "coaches"

        Returns:
            List of ranking dicts with team_id, rank, points, first_place_votes
        """
        cache_key = f"rankings_{season}_{week}_{poll_type}"

        # Check cache first
        if self.storage:
            cached = await self.storage.get_cached(cache_key)
            if cached:
                return cached

        # Fetch from API
        params = {"year": season, "week": week}
        data = await self._make_request("/rankings", params=params)

        # Find the right poll
        poll_name_map = {
            "ap": "AP Top 25",
            "cfp": "Playoff Committee Rankings",
            "coaches": "Coaches Poll",
        }
        target_poll = poll_name_map.get(poll_type, poll_type)

        rankings = []
        for week_data in data:
            for poll in week_data.get("polls", []):
                if poll.get("poll") == target_poll:
                    for rank_entry in poll.get("ranks", []):
                        rankings.append({
                            "team_id": self._normalize_team_id(rank_entry["school"]),
                            "rank": rank_entry["rank"],
                            "points": rank_entry.get("points"),
                            "first_place_votes": rank_entry.get("firstPlaceVotes"),
                        })
                    break

        # Cache the result
        if self.storage:
            await self.storage.set_cached(cache_key, rankings, ttl_hours=24)

        return rankings
