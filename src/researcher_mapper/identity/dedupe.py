"""Deduplicate a candidate list by OpenAlex ID, ORCID, or fuzzy name+institution."""
from __future__ import annotations

from rapidfuzz.fuzz import token_sort_ratio


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    """
    Remove duplicates from a list of OpenAlex author dicts.
    Deduplication order:
      1. Same openalex `id`
      2. Same ORCID
      3. Name + institution fuzzy match (≥ 0.92)
    """
    seen_ids: set[str] = set()
    seen_orcids: set[str] = set()
    deduped: list[dict] = []

    for c in candidates:
        oa_id = (c.get("id") or "").split("/")[-1]
        orcid = (c.get("orcid") or "").replace("https://orcid.org/", "")

        if oa_id and oa_id in seen_ids:
            continue
        if orcid and orcid in seen_orcids:
            continue

        # Fuzzy name check only when IDs are absent
        if not oa_id and not orcid:
            name = (c.get("display_name") or "").lower()
            inst = (
                (c.get("last_known_institution") or {}).get("display_name") or ""
            ).lower()
            key = f"{name}|{inst}"
            is_dup = False
            for existing in deduped:
                ex_name = (existing.get("display_name") or "").lower()
                ex_inst = (
                    (existing.get("last_known_institution") or {}).get("display_name") or ""
                ).lower()
                ex_key = f"{ex_name}|{ex_inst}"
                if token_sort_ratio(key, ex_key) >= 92:
                    is_dup = True
                    break
            if is_dup:
                continue

        deduped.append(c)
        if oa_id:
            seen_ids.add(oa_id)
        if orcid:
            seen_orcids.add(orcid)

    return deduped
