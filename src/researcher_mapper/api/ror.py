"""ROR (Research Organization Registry) API client."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BASE_URL = "https://api.ror.org/organizations"


class RorClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self.client = httpx.Client(timeout=timeout)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    def _get(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.client.get(BASE_URL, params=params or {})
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, page: int = 1) -> dict[str, Any]:
        return self._get(params={"query": query, "page": page})

    def get_by_id(self, ror_id: str) -> dict[str, Any]:
        """Fetch a single organization by ROR ID (full URL or bare ID)."""
        clean = ror_id.replace("https://ror.org/", "").strip("/")
        resp = self.client.get(f"{BASE_URL}/{clean}")
        resp.raise_for_status()
        return resp.json()

    def normalize_institution_name(self, raw_name: str) -> dict[str, Any] | None:
        """Return the best-matching ROR record for a raw institution string."""
        results = self.search(raw_name).get("items", [])
        if not results:
            return None
        return results[0]
