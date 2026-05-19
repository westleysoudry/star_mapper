"""OpenAlex API client.

Uses the polite pool (mailto param) and cursor-based pagination.
Rate limit: stay under 10 req/s; tenacity handles transient 429/5xx.

Retry policy: only 429 and 5xx are retried.  400 Bad Request is a caller
error (wrong filter format) and will never succeed on retry — failing fast
prevents burning the 5-attempt quota on every bad concept ID.
"""
from __future__ import annotations

import re
import time
from typing import Any, Generator

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from researcher_mapper.settings import settings

BASE_URL = "https://api.openalex.org"
_POLITE_DELAY = 0.12  # ~8 req/s — comfortably inside the 10 req/s soft cap



def _is_retryable(exc: BaseException) -> bool:
    """Retry only on rate-limit (429) and transient server errors (5xx)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TimeoutException)


class OpenAlexClient:
    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.email = email or settings.openalex_email
        self.api_key = api_key or settings.openalex_api_key
        self.client = httpx.Client(timeout=timeout)
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _POLITE_DELAY:
            time.sleep(_POLITE_DELAY - elapsed)
        self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["mailto"] = self.email
        resp = self.client.get(f"{BASE_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    # ── concept ID normalisation ───────────────────────────────────────────────

    @staticmethod
    def _normalize_concept_id(raw: str) -> str:
        """
        Normalise a concept ID to full OpenAlex URL form.

        OpenAlex may return concept IDs in several formats depending on context:
          • Full URL:     https://openalex.org/C41008148
          • Prefixed:     C41008148
          • Numeric only: 41008148   ← common in x_concepts on author records

        The filter API requires the full URL (or at minimum the C-prefixed form).
        """
        raw = raw.strip()
        if raw.startswith("https://openalex.org/"):
            return raw
        # Single letter prefix: C/T/W/A/I/V (concept, topic, work, author…)
        if re.match(r"^[CTWAIVctwaiv]\d+$", raw):
            return f"https://openalex.org/{raw.upper()}"
        # Bare numeric — assume concept (C prefix)
        if re.match(r"^\d+$", raw):
            return f"https://openalex.org/C{raw}"
        return raw  # pass through; caller will discover the error

    # ── single-entity endpoints ────────────────────────────────────────────────

    def get_author(self, author_id: str) -> dict[str, Any]:
        """Fetch a single author record by OpenAlex ID (short or full URL)."""
        oid = author_id.split("/")[-1]
        return self._get(f"/authors/{oid}")

    def get_work(self, work_id: str) -> dict[str, Any]:
        oid = work_id.split("/")[-1]
        return self._get(f"/works/{oid}")

    def get_institution(self, institution_id: str) -> dict[str, Any]:
        oid = institution_id.split("/")[-1]
        return self._get(f"/institutions/{oid}")

    # ── search endpoints ───────────────────────────────────────────────────────

    def search_authors(
        self,
        name: str,
        institution_name: str | None = None,
        per_page: int = 25,
    ) -> list[dict[str, Any]]:
        filt = f"display_name.search:{name}"
        data = self._get("/authors", params={"filter": filt, "per-page": per_page})
        return data.get("results", [])

    def search_institutions(self, name: str, per_page: int = 25) -> list[dict[str, Any]]:
        data = self._get(
            "/institutions",
            params={"filter": f"display_name.search:{name}", "per-page": per_page},
        )
        return data.get("results", [])

    def get_recent_institution_authors(
        self,
        institution_id: str,
        from_year: int = 2022,
        per_page_works: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Find authors who published from a specific institution recently.

        Uses work-level ``authorships.institutions`` (set at submission time),
        which is far more reliable for identifying *current* faculty than
        ``affiliations.institution.id`` — the latter covers career history and
        returns honorary-degree recipients, former students, and visitors who
        have never been full-time faculty.
        """
        oid = institution_id.split("/")[-1]
        try:
            data = self._get(
                "/works",
                params={
                    "filter": f"institutions.id:{oid},from_publication_date:{from_year}-01-01",
                    "per-page": min(per_page_works, 200),
                    "select": "authorships",
                },
            )
        except Exception:
            return []

        authors: dict[str, dict[str, Any]] = {}
        for work in data.get("results", []):
            for auth in work.get("authorships", []):
                author = auth.get("author") or {}
                aid = author.get("id")
                if not aid or aid in authors:
                    continue
                insts = auth.get("institutions") or []
                if any(i.get("id", "").split("/")[-1] == oid for i in insts):
                    authors[aid] = author
        return list(authors.values())

    def get_recent_country_authors(
        self,
        country_code: str,
        from_year: int = 2022,
        per_page_works: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Find authors who published from *country_code* institutions recently.

        Uses work-level ``authorships.institutions`` — the institution listed
        on the paper itself — rather than ``affiliations.institution.*`` which
        covers career history.  A researcher who published from IL in 2023 is
        *currently* at an IL institution by definition, so this is a much more
        reliable current-affiliation signal than historical affiliation filters.

        Returns a flat list of minimal author dicts (id, display_name).
        """
        try:
            data = self._get(
                "/works",
                params={
                    "filter": (
                        f"institutions.country_code:{country_code},"
                        f"from_publication_date:{from_year}-01-01"
                    ),
                    "per-page": min(per_page_works, 200),
                    "select": "authorships",
                },
            )
        except Exception:
            return []

        authors: dict[str, dict[str, Any]] = {}
        for work in data.get("results", []):
            for auth in work.get("authorships", []):
                author = auth.get("author") or {}
                aid = author.get("id")
                if not aid or aid in authors:
                    continue
                insts = auth.get("institutions") or []
                if any(i.get("country_code") == country_code for i in insts):
                    authors[aid] = author
        return list(authors.values())

    def get_israel_institutions(
        self,
        per_page: int = 200,
        sort: str = "cited_by_count:desc",
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"filter": "country_code:IL", "per-page": per_page}
        if sort:
            params["sort"] = sort
        data = self._get("/institutions", params=params)
        return data.get("results", [])

    def get_topics(self, search_term: str, per_page: int = 25) -> list[dict[str, Any]]:
        data = self._get(
            "/topics",
            params={"filter": f"display_name.search:{search_term}", "per-page": per_page},
        )
        return data.get("results", [])

    # ── paginated work fetch ───────────────────────────────────────────────────

    def iter_author_works(
        self,
        author_id: str,
        from_year: int = 2014,
        max_works: int = 500,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield all works for an author using cursor pagination."""
        oid = author_id.split("/")[-1]
        filt = f"author.id:{oid},from_publication_date:{from_year}-01-01"
        cursor = "*"
        fetched = 0
        while cursor and fetched < max_works:
            data = self._get(
                "/works",
                params={
                    "filter": filt,
                    "per-page": 200,
                    "cursor": cursor,
                    "select": (
                        "id,doi,title,publication_year,primary_location,"
                        "authorships,topics,referenced_works,cited_by_count"
                    ),
                },
            )
            results = data.get("results", [])
            if not results:
                break
            for work in results:
                yield work
                fetched += 1
                if fetched >= max_works:
                    break
            cursor = data.get("meta", {}).get("next_cursor")

    def get_author_works(
        self,
        author_id: str,
        from_year: int = 2014,
        max_works: int = 500,
    ) -> list[dict[str, Any]]:
        return list(self.iter_author_works(author_id, from_year=from_year, max_works=max_works))

    # ── candidate retrieval helpers ────────────────────────────────────────────

    def get_authors_by_concept(
        self,
        concept_id: str,
        country_code: str | None = None,
        per_page: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Return authors whose works are tagged with the given concept.

        The ``x_concepts.id`` filter on /authors is no longer supported by the
        OpenAlex API (returns 400 for all concept IDs), so we go straight to the
        works-based fallback: fetch recent works by concept and extract authors.
        """
        normalized = self._normalize_concept_id(concept_id)
        return self._authors_from_works(
            concept_id=normalized, country_code=country_code, per_page=per_page
        )

    def get_authors_by_topic(
        self,
        topic_id: str,
        country_code: str | None = None,
        per_page: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Return authors working on the given topic, sorted by citation count.

        Tries the /authors endpoint first (returns richly-sorted results by
        cited_by_count).  Falls back to works-based extraction on 400/error.
        For country-filtered queries prefer ``get_authors_by_topic_and_country``.
        """
        normalized = self._normalize_concept_id(topic_id)
        short_id = normalized.split("/")[-1]
        filt = f"topics.id:{short_id}"
        if country_code:
            filt += f",affiliations.institution.country_code:{country_code}"
        try:
            data = self._get(
                "/authors",
                params={
                    "filter": filt,
                    "sort": "cited_by_count:desc",
                    "per-page": per_page,
                },
            )
            results = data.get("results", [])
            if results:
                return results
        except httpx.HTTPStatusError:
            pass
        return self._authors_from_works(
            concept_id=normalized, country_code=country_code,
            per_page=per_page, field="topics.id",
        )

    def get_authors_by_topic_and_country(
        self,
        topic_id: str,
        country_code: str,
        per_page: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Return authors affiliated with *country_code* institutions who also work
        on *topic_id*.

        The /authors endpoint does not support ``last_known_institution.*`` as a
        filter field; ``affiliations.institution.country_code`` is the correct
        filterable attribute (covers current and recent affiliations).

        Falls back to works-based extraction on 400.
        """
        short_id = self._normalize_concept_id(topic_id).split("/")[-1]
        try:
            data = self._get(
                "/authors",
                params={
                    "filter": (
                        f"topics.id:{short_id},"
                        f"affiliations.institution.country_code:{country_code}"
                    ),
                    "sort": "cited_by_count:desc",
                    "per-page": per_page,
                },
            )
            results = data.get("results", [])
            if results:
                return results
        except httpx.HTTPStatusError:
            pass
        # Fall back: extract from works, post-filter by work-level affiliation
        return self._authors_from_works(
            concept_id=f"https://openalex.org/{short_id}",
            country_code=country_code,
            per_page=per_page,
            field="topics.id",
        )

    def _authors_from_works(
        self,
        concept_id: str,
        country_code: str | None = None,
        per_page: int = 25,
        field: str = "concepts.id",
    ) -> list[dict[str, Any]]:
        """
        Discover authors by fetching recent works tagged with concept_id and
        extracting unique authors from those works' authorships.

        ``country_code`` is applied as a post-filter (cheaper than a compound
        works filter, which has different semantics on the works endpoint).
        """
        short_id = concept_id.split("/")[-1]   # e.g. "C41008148" or "T12345"
        works_filter = f"{field}:{short_id},from_publication_date:2020-01-01"
        try:
            data = self._get(
                "/works",
                params={
                    "filter": works_filter,
                    "per-page": min(per_page * 3, 50),
                    "select": "authorships",
                },
            )
        except httpx.HTTPStatusError:
            return []

        authors: dict[str, dict[str, Any]] = {}
        for work in data.get("results", []):
            for auth in work.get("authorships", []):
                author = auth.get("author") or {}
                aid = author.get("id")
                if not aid or aid in authors:
                    continue
                if country_code:
                    insts = auth.get("institutions") or []
                    if not any(i.get("country_code") == country_code for i in insts):
                        continue
                authors[aid] = author   # minimal dict; full summary fetched later

        return list(authors.values())[:per_page]

    def get_top_authors_by_country(
        self,
        country_code: str,
        per_page: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Return top-cited authors affiliated with institutions in the given country.

        Uses ``affiliations.institution.country_code`` — the correct filterable
        attribute on /authors (``last_known_institution.*`` is not filterable).
        Covers researchers with any affiliation in that country, not just current;
        relevance is enforced downstream by topic scoring.
        """
        data = self._get(
            "/authors",
            params={
                "filter": f"affiliations.institution.country_code:{country_code}",
                "sort": "cited_by_count:desc",
                "per-page": per_page,
            },
        )
        return data.get("results", [])

    def get_authors_at_institution(
        self,
        institution_id: str,
        per_page: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Return top-cited authors affiliated with the given institution.

        Uses ``affiliations.institution.id`` — the correct filterable attribute
        on /authors (``last_known_institution.id`` is not filterable).
        """
        oid = institution_id.split("/")[-1]
        data = self._get(
            "/authors",
            params={
                "filter": f"affiliations.institution.id:{oid}",
                "sort": "cited_by_count:desc",
                "per-page": per_page,
            },
        )
        return data.get("results", [])

    def get_citing_authors(
        self,
        work_id: str,
        per_page: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Return authors who have works that cite the given work.

        Sorted by ``cited_by_count:desc`` so the most-cited citing papers
        (written by prominent researchers) are fetched first.  Without this
        sort the default ordering is by recency, which biases toward recent
        grad-student papers and misses established researchers who cited the
        work years ago.
        """
        oid = work_id.split("/")[-1]
        data = self._get(
            "/works",
            params={
                "filter": f"cites:{oid}",
                "sort": "cited_by_count:desc",
                "per-page": per_page,
                "select": "id,authorships",
            },
        )
        authors: dict[str, dict] = {}
        for work in data.get("results", []):
            for auth in work.get("authorships", []):
                a = auth.get("author", {})
                aid = a.get("id")
                if aid:
                    authors[aid] = a
        return list(authors.values())

    def get_author_summary(self, author_id: str) -> dict[str, Any]:
        """Fetch an author profile with fields useful for quick candidate scoring."""
        oid = author_id.split("/")[-1]
        # No select parameter — requesting specific fields (especially x_concepts
        # or last_known_institution) causes 400 on some API versions.
        return self._get(f"/authors/{oid}")
