"""Crossref REST API client — DOI resolution and metadata backfill."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from researcher_mapper.settings import settings

BASE_URL = "https://api.crossref.org"


class CrossrefClient:
    def __init__(self, mailto: str | None = None, timeout: float = 30.0) -> None:
        self.mailto = mailto or settings.crossref_mailto
        self.client = httpx.Client(timeout=timeout)

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = dict(extra or {})
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.client.get(f"{BASE_URL}{path}", params=self._params(params))
        resp.raise_for_status()
        return resp.json()

    def get_work(self, doi: str) -> dict[str, Any]:
        clean = doi.replace("https://doi.org/", "").lstrip("/")
        return self._get(f"/works/{clean}")

    def search_works(self, bibliographic: str, rows: int = 10) -> dict[str, Any]:
        return self._get("/works", params={"query.bibliographic": bibliographic, "rows": rows})

    def extract_abstract(self, work_message: dict) -> str | None:
        """Pull abstract text from a Crossref work message, stripping JATS tags."""
        raw = work_message.get("abstract", "")
        if not raw:
            return None
        import re
        return re.sub(r"<[^>]+>", " ", raw).strip() or None
