"""
Retrieve candidate researchers from OpenAlex using complementary strategies,
then return two separate lists: a global pool and a country-specific pool.

The two pools are built and capped independently so that country-specific
candidates (e.g. Israel) always get dedicated slots and are never crowded out
by the larger global pool.

Strategies (global pool, in priority order)
-------------------------------------------
1. Citing authors - researchers who cited the target's top papers.
   Strongest specificity signal: they explicitly built on the target's work.
2. Reference neighbours - authors of works the target cited.
   Strong source for senior recommendation-letter candidates.
3. T-topic neighbours (primary) - recent works tagged with the target's
   publication-level T-prefixed topic IDs.  Finer-grained than concept search.
4. C-concept neighbours (fallback) - older x_concepts field; covers works with
   no topic tags.  Lighter: 3 concepts x 10 results.
5. Direct coauthors - included last (will be excluded from output by policy,
   but kept for graph building and coauthor-flag computation).

Country pool (separate, dedicated slots)
-----------------------------------------
• Strategy A: T-topic + affiliations.institution.country_code filter.
• Strategy B: Top-cited authors by country-wide affiliation filter.
• Strategy C: Per-institution fetch from each major research university in
  the target country — these authors tend to have the institution as their
  last_known_institution, maximising truly-current affiliation hit-rate.
"""
from __future__ import annotations

import logging

from researcher_mapper.api.openalex import OpenAlexClient
from researcher_mapper.features.topic_features import build_topic_vector
from researcher_mapper.identity.dedupe import dedupe_candidates
from researcher_mapper.models.schemas import Publication, ResearcherProfile

log = logging.getLogger(__name__)

# Dedicated slots for the country-specific pool within max_pool_size.
# These are subtracted from the global cap so the total never exceeds
# max_pool_size after dedup-merge.
_COUNTRY_SLOTS = 150

# Adaptive pool-size scaling: pool grows with the target's citation count
# so that all candidate strategies (including citation neighbours) always
# have room, even for highly-cited researchers.
# Formula: max(MIN, min(MAX, total_citations // DIVISOR))
_ADAPTIVE_POOL_MIN = 200
_ADAPTIVE_POOL_MAX = 800
_ADAPTIVE_POOL_DIVISOR = 20


def _get_top_topic_ids(publications: list[Publication], n: int = 5) -> list[str]:
    """Return the top-n T-prefixed topic IDs from the target's publication record."""
    topic_vec = build_topic_vector(publications)
    return [tid for tid, _ in sorted(topic_vec.items(), key=lambda x: x[1], reverse=True)[:n]]


def _get_top_concept_ids(profile: ResearcherProfile, n: int = 3) -> list[str]:
    """Return the top-n C-prefixed concept IDs from the target's author profile."""
    sorted_concepts = sorted(profile.top_concepts.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sorted_concepts[:n]]


def _extract_coauthor_ids(
    publications: list[Publication],
    target_openalex_id: str,
) -> list[str]:
    seen: dict[str, int] = {}
    for pub in publications:
        for aid in pub.author_ids:
            # aid is a full URL (https://openalex.org/A...) — compare normalized
            if aid and aid.split("/")[-1] != target_openalex_id.split("/")[-1]:
                seen[aid] = seen.get(aid, 0) + 1
    return [aid for aid, _ in sorted(seen.items(), key=lambda x: x[1], reverse=True)]


def _extract_cited_author_ids(
    publications: list[Publication],
    oa: OpenAlexClient,
    limit_works: int = 20,
    refs_per_work: int = 50,
) -> list[str]:
    """Authors of works referenced by the target's most-cited papers."""
    top_works = sorted(publications, key=lambda p: p.cited_by_count, reverse=True)[:limit_works]
    all_ref_ids: set[str] = set()
    for work in top_works:
        for ref_id in work.referenced_work_ids[:refs_per_work]:
            if ref_id:
                all_ref_ids.add(ref_id)
    if not all_ref_ids:
        return []
    short_ids = [rid.split("/")[-1] for rid in all_ref_ids if rid.split("/")[-1]]
    author_ids: set[str] = set()
    batch_size = 50
    for i in range(0, len(short_ids), batch_size):
        batch = short_ids[i : i + batch_size]
        try:
            data = oa._get(
                "/works",
                params={
                    "filter": "openalex_id:" + "|".join(batch),
                    "per-page": batch_size,
                    "select": "authorships",
                },
            )
            for work_data in data.get("results", []):
                for auth in work_data.get("authorships", []):
                    aid = auth.get("author", {}).get("id")
                    if aid:
                        author_ids.add(aid)
        except Exception:
            continue
    return list(author_ids)


def retrieve_candidate_pool(
    target_profile: ResearcherProfile,
    oa: OpenAlexClient | None = None,
    country_code: str | None = None,
    per_concept: int = 20,
    max_pool_size: int | None = None,
    include_citation_neighbors: bool = False,
) -> list[dict]:
    """
    Build a candidate pool split into global and country-specific parts.

    Returns a flat list of author dicts.  Country candidates always occupy
    ``_COUNTRY_SLOTS`` dedicated slots so they are never crowded out by the
    larger global pool.

    Parameters
    ----------
    target_profile:             The resolved target ResearcherProfile.
    oa:                         Reuse an existing client (or create one).
    country_code:               If set (e.g. "IL"), generate a country-specific
                                sub-pool with guaranteed slots.
    per_concept:                Authors to retrieve per topic/concept search.
    max_pool_size:              Total candidate cap (global + country combined).
                                When None (default), the cap is computed
                                adaptively from the target's total citation
                                count so that all strategies always have room.
    include_citation_neighbors: Also fetch authors from reference lists (slow).
    """
    oa = oa or OpenAlexClient()
    target_id = (target_profile.openalex_author_id or "").split("/")[-1]
    pubs = target_profile.publications

    if max_pool_size is None:
        total_citations = sum(p.cited_by_count for p in pubs)
        max_pool_size = max(
            _ADAPTIVE_POOL_MIN,
            min(_ADAPTIVE_POOL_MAX, total_citations // _ADAPTIVE_POOL_DIVISOR),
        )
        log.info("Adaptive pool size: %d (total citations: %d)", max_pool_size, total_citations)

    top_topic_ids = _get_top_topic_ids(pubs, n=8)
    log.info("Top topic IDs for candidate search: %s", top_topic_ids[:8])

    # ── Global pool ────────────────────────────────────────────────────────────
    global_pool: list[dict] = []

    # Strategy 0: direct coauthors — added FIRST so they are never crowded out
    # by the pool cap applied later.  Coauthorship is the strongest signal that
    # a researcher is in the same area; without this guarantee, prolific coauthors
    # can be displaced by less-relevant candidates from the topic searches.
    for aid in _extract_coauthor_ids(pubs, target_id)[:50]:
        global_pool.append({"id": aid, "_strategy": "coauthor"})
    log.info("After coauthor strategy: %d in global pool", len(global_pool))

    # Strategy 1: same-institution researchers (seeds dept_mentor bucket).
    # Uses work-level authorships.institutions (set at submission time) rather
    # than affiliations.institution.id (career history — returns honorary degree
    # holders and famous emigrés whose last_known_institution is elsewhere).
    if target_profile.current_institution and target_profile.current_institution.openalex_id:
        try:
            for a in oa.get_recent_institution_authors(
                target_profile.current_institution.openalex_id,
                from_year=2022,
                per_page_works=200,
            ):
                a["_strategy"] = "same_institution"
                global_pool.append(a)
        except Exception:
            pass
        log.info(
            "After institution strategy: %d in global pool", len(global_pool)
        )

    # Strategy 2: citing authors — researchers who cited the target's papers.
    # Strongest specificity signal; use up to 50 papers so the citation graph
    # neighbourhood is well-sampled.  Papers with no citations are skipped.
    top_cited = sorted(
        [p for p in pubs if p.cited_by_count > 0],
        key=lambda p: p.cited_by_count,
        reverse=True,
    )[:50]
    for work in top_cited:
        if not work.id:
            continue
        try:
            for a in oa.get_citing_authors(work.id, per_page=50):
                a["_strategy"] = "citing_author"
                global_pool.append(a)
        except Exception:
            pass
    log.info("After citing-authors strategy: %d in global pool", len(global_pool))

    # Strategy 3: reference neighbours (optional, slow). These are especially
    # useful for senior letter-writer candidates: people the target built on may
    # be more relevant than broad OpenAlex topic-neighbour leaders.
    if include_citation_neighbors:
        for aid in _extract_cited_author_ids(pubs, oa, limit_works=20, refs_per_work=50):
            global_pool.append({"id": aid, "_strategy": "cited_author"})
        log.info("After cited-author strategy: %d in global pool", len(global_pool))

    # Strategy 4: T-topic global search
    for topic_id in top_topic_ids:
        try:
            for a in oa.get_authors_by_topic(topic_id, per_page=per_concept):
                a["_strategy"] = "topic_global"
                global_pool.append(a)
        except Exception:
            pass

    # Strategy 5: C-concept global (supplementary — older works without topic tags)
    for concept_id in _get_top_concept_ids(target_profile, n=3):
        try:
            for a in oa.get_authors_by_concept(concept_id, per_page=10):
                a["_strategy"] = "concept_global"
                global_pool.append(a)
        except Exception:
            pass

    # Capture citing-author IDs BEFORE dedup so that candidates discovered via
    # multiple strategies don't lose their citing-author provenance when dedup
    # keeps the earlier (non-citing-author) copy.
    _citing_ids_pre_dedup: set[str] = {
        c.get("id", "").split("/")[-1]
        for c in global_pool
        if c.get("_strategy") == "citing_author" and c.get("id")
    }
    global_pool = dedupe_candidates(global_pool)
    # Re-annotate surviving entries that were also citing authors.
    for c in global_pool:
        if c.get("id", "").split("/")[-1] in _citing_ids_pre_dedup:
            c["_is_citing_author"] = True
    global_pool = [
        c for c in global_pool
        if c.get("id", "").split("/")[-1] not in (target_id, "")
    ]
    global_cap = max(max_pool_size - _COUNTRY_SLOTS, max_pool_size // 2)
    log.info(
        "Global pool after dedup: %d (capped at %d)", len(global_pool), global_cap
    )

    # ── Country pool (dedicated slots — not competing with global) ─────────────
    # Three strategies; C is built first because its candidates are the most
    # reliably current-IL and must not be truncated by the _COUNTRY_SLOTS cap.
    #
    #   C. Per top-institution (FIRST): fetch authors from each major research
    #      university in the target country.  Pre-filtered to researchers whose
    #      last_known_institution is currently in that country — the most
    #      reliable signal of active affiliation.
    #   A. Topic-specific: /authors by topics.id + affiliations.country_code
    #      (most topically relevant but includes past affiliates).
    #   B. Country-wide top-cited: high-citation researchers with any IL affiliation;
    #      covers broad fields but includes researchers who have since left.
    country_pool: list[dict] = []
    if country_code:
        # ── Strategy A (first) ─────────────────────────────────────────────────
        # Topic-filtered country researchers come FIRST so they survive the
        # _COUNTRY_SLOTS cap.  Country pools are much smaller than global, so
        # fetching top-200 per topic is cheap and ensures mid-ranked local
        # researchers (fewer citations than global leaders, but equally
        # relevant) are captured.  At per_page=50 a researcher ranked ~85 for
        # a given topic+country query would be missed; per_page=200 prevents this.
        country_per_page = max(per_concept * 4, 200)
        for topic_id in top_topic_ids:
            try:
                for a in oa.get_authors_by_topic_and_country(
                    topic_id, country_code, per_page=country_per_page
                ):
                    a["_strategy"] = f"topic_{country_code}"
                    country_pool.append(a)
            except Exception:
                pass
        log.info("Strategy A (topic+%s): %d candidates", country_code, len(country_pool))

        # ── Strategy B ─────────────────────────────────────────────────────────
        try:
            for a in oa.get_top_authors_by_country(country_code, per_page=100):
                a["_strategy"] = f"country_{country_code}"
                country_pool.append(a)
        except Exception:
            pass

        # ── Strategy C (last) ──────────────────────────────────────────────────
        # Recent IL paper authors: broad current-affiliation signal.  Placed
        # last because it lacks topical filtering and would otherwise saturate
        # the _COUNTRY_SLOTS cap with irrelevant researchers.
        strategy_c: list[dict] = []
        try:
            for a in oa.get_recent_country_authors(
                country_code, from_year=2022, per_page_works=200
            ):
                a["_strategy"] = f"recent_{country_code}"
                strategy_c.append(a)
        except Exception:
            pass
        log.info("Strategy C (recent works, %s): %d candidates", country_code, len(strategy_c))

        # Merge: A+B first so topically-relevant candidates take priority
        country_pool = country_pool + strategy_c
        log.info("Country pool before dedup: %d", len(country_pool))

        country_pool = dedupe_candidates(country_pool)
        country_pool = [
            c for c in country_pool
            if c.get("id", "").split("/")[-1] not in (target_id, "")
        ]
        log.info(
            "Country pool (%s) after dedup: %d (capped at %d)",
            country_code, len(country_pool), _COUNTRY_SLOTS,
        )

    # ── Merge with independent caps ────────────────────────────────────────────
    combined = dedupe_candidates(
        global_pool[:global_cap] + country_pool[:_COUNTRY_SLOTS]
    )
    log.info("Combined pool size: %d", len(combined))
    return combined
