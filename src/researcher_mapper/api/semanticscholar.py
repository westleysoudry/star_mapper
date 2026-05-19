"""Semantic Scholar Academic Graph API client."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from researcher_mapper.settings import settings

BASE_URL = "https://api.semanticscholar.org/graph/v1"

_AUTHOR_FIELDS = (
    "name,affiliations,paperCount,citationCount,hIndex,"
    "papers.title,papers.year,papers.citationCount,papers.externalIds"
)
_PAPER_FIELDS = (
    "title,year,authors,referenceCount,citationCount,"
    "references.paperId,citations.paperId,externalIds,fieldsOfStudy"
)


class SemanticScholarClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"x-api-key": self.api_key}
        return {}

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.client.get(
            f"{BASE_URL}{path}",
            params=params or {},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def search_author(self, query: str, limit: int = 10) -> dict[str, Any]:
        return self._get("/author/search", params={"query": query, "limit": limit})

    def get_author(self, author_id: str, fields: str = _AUTHOR_FIELDS) -> dict[str, Any]:
        return self._get(f"/author/{author_id}", params={"fields": fields})

    def get_paper(self, paper_id: str, fields: str = _PAPER_FIELDS) -> dict[str, Any]:
        return self._get(f"/paper/{paper_id}", params={"fields": fields})

    def get_paper_by_doi(self, doi: str, fields: str = _PAPER_FIELDS) -> dict[str, Any]:
        clean = doi.replace("https://doi.org/", "").lstrip("/")
        return self._get(f"/paper/DOI:{clean}", params={"fields": fields})

    def get_author_papers(self, author_id: str, limit: int = 100) -> dict[str, Any]:
        return self._get(
            f"/author/{author_id}/papers",
            params={"fields": "title,year,citationCount,externalIds", "limit": limit},
        )
