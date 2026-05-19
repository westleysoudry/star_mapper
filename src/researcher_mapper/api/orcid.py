"""ORCID Public API client (no authentication required for public records)."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BASE_URL = "https://pub.orcid.org/v3.0"


class OrcidClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self.client = httpx.Client(
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.client.get(f"{BASE_URL}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, rows: int = 10, start: int = 0) -> dict[str, Any]:
        """Full-text search across ORCID public records."""
        return self._get("/search", params={"q": query, "rows": rows, "start": start})

    def get_record(self, orcid_id: str) -> dict[str, Any]:
        """Retrieve the full public record for an ORCID iD."""
        clean = orcid_id.replace("https://orcid.org/", "").strip("/")
        return self._get(f"/{clean}/record")

    def get_works(self, orcid_id: str) -> dict[str, Any]:
        clean = orcid_id.replace("https://orcid.org/", "").strip("/")
        return self._get(f"/{clean}/works")

    def get_employments(self, orcid_id: str) -> dict[str, Any]:
        clean = orcid_id.replace("https://orcid.org/", "").strip("/")
        return self._get(f"/{clean}/employments")

    def extract_affiliations(self, record: dict) -> list[dict]:
        """Pull institution name, department, and dates from a full ORCID record."""
        affiliations = []
        activities = record.get("activities-summary", {})
        employments = (
            activities.get("employments", {})
            .get("affiliation-group", [])
        )
        for group in employments:
            summaries = group.get("summaries", [])
            for summary in summaries:
                emp = summary.get("employment-summary", {})
                org = emp.get("organization", {})
                dept = emp.get("department-name")
                role = emp.get("role-title")
                start = emp.get("start-date", {})
                end = emp.get("end-date", {})
                disambig = org.get("disambiguated-organization") or {}
                affiliations.append(
                    {
                        "org_name": org.get("name"),
                        "ror": (
                            disambig.get("disambiguated-organization-identifier")
                            if disambig.get("disambiguation-source") == "ROR"
                            else None
                        ),
                        "department": dept,
                        "role": role,
                        "start_year": start.get("year", {}).get("value") if start else None,
                        "end_year": end.get("year", {}).get("value") if end else None,
                    }
                )
        return affiliations
