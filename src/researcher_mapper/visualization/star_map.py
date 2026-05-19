"""Interactive Plotly star map of the researcher-mapping output.

Layout
------
- Target researcher at the origin (centre), rendered as a pulsing star.
- Israel candidates occupy the right half (x > 0); world on the left (x < 0).
- Within each half, 2D MDS on **citation-reference overlap** places
  researchers so that spatial distance ≈ intellectual distance (who cites
  the same papers).  Falls back to topic vectors when reference data is absent.

Constellations
--------------
- Candidates are clustered (K-means, k=5) on citation vectors.
- Within each cluster a **minimum spanning tree** (on spatial positions)
  defines the edges — producing a sparse constellation graph, not a mesh.

Visual polish
-------------
- Each star is drawn in three overlapping layers (halo / glow / core) for
  a real-starfield feel.
- Soft nebula patches in the background hint at galactic depth.
- CSS animations: twinkling background stars, pulsing target star,
  periodic shooting stars.
- A pinned info panel mirrors the currently-hovered researcher.

Interactivity
-------------
- Hover: star enlarges slightly and the info panel updates.
- Click a star: opens a Google Scholar search in a new tab.
- Click a legend entry: toggles that bucket on/off.
- Standard Plotly mode bar: zoom, pan, reset, save-as-PNG.
"""
from __future__ import annotations

import colorsys
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import plotly.graph_objects as go
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans

from researcher_mapper.models.schemas import CandidateScore, ResearcherProfile

log = logging.getLogger(__name__)


def _clamp_topic_axis_count(n: int | None) -> int:
    """Keep the star-map topic knob in a useful visual range."""
    try:
        value = int(n if n is not None else 5)
    except (TypeError, ValueError):
        value = 5
    return max(2, min(value, 10))


# Bucket → star colour (designed for dark background).
_BUCKET_COLORS: dict[str, str] = {
    "dept_mentors":                    "#7CFC9A",
    "area_mentors":                    "#D9BFFF",
    "recommendation_letter_writers":   "#FF9999",
    "in_area_collaborators":           "#F3D94D",
    "interdisciplinary_collaborators": "#8CD9FF",
}

_BUCKET_LABELS: dict[str, str] = {
    "dept_mentors":                    "Institutional Mentors",
    "area_mentors":                    "Area Mentors",
    "recommendation_letter_writers":   "Letter Writers",
    "in_area_collaborators":           "In-Area Collaborators",
    "interdisciplinary_collaborators": "Interdisciplinary Collaborators",
}

_TOPIC_ID_RE = re.compile(r"(?:https://openalex\.org/)?(T\d+)\b")


# ── Feature extraction ────────────────────────────────────────────────────────

def _collect_feature_dict(
    candidates: list[CandidateScore],
    summaries_by_id: dict,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    cit_vecs, top_vecs = [], []
    for cand in candidates:
        summary = summaries_by_id.get(cand.candidate_id)
        if summary is None:
            cit_vecs.append({}); top_vecs.append({}); continue
        fv = getattr(summary, "feature_vector", None)
        cit = dict(fv.citation_vector) if (fv and fv.citation_vector) else {}
        topic = (
            dict(fv.topic_vector) if (fv and fv.topic_vector)
            else dict(summary.top_topics or {})
        )
        cit_vecs.append(cit); top_vecs.append(topic)
    return cit_vecs, top_vecs


def _build_joint_matrix(
    target_cit: dict[str, float],
    target_top: dict[str, float],
    cand_cit_vecs: list[dict[str, float]],
    cand_top_vecs: list[dict[str, float]],
    citation_weight: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_row, candidate_matrix) in a shared L2-normalised feature space.

    Citation dimensions are weighted up so that when both are present, citations
    dominate; candidates with empty citation_vectors still have a non-zero row
    via their topic vector.
    """
    # Collect all keys from BOTH sources across target + candidates
    cit_keys: set[str] = set(target_cit.keys())
    for v in cand_cit_vecs:
        cit_keys.update(v.keys())
    top_keys: set[str] = set(target_top.keys())
    for v in cand_top_vecs:
        top_keys.update(v.keys())

    cit_list = sorted(cit_keys)
    top_list = sorted(top_keys)
    n_cit, n_top = len(cit_list), len(top_list)
    k = n_cit + n_top
    n = len(cand_cit_vecs)

    if k == 0:
        return np.zeros(0), np.zeros((n, 0))

    def _row(cit: dict, top: dict) -> np.ndarray:
        r = np.zeros(k)
        for i, key in enumerate(cit_list):
            r[i] = cit.get(key, 0.0) * citation_weight
        for i, key in enumerate(top_list):
            r[n_cit + i] = top.get(key, 0.0)
        nrm = np.linalg.norm(r)
        return r / nrm if nrm > 0 else r

    target_row = _row(target_cit, target_top)
    cand_mat = np.zeros((n, k))
    for idx, (cit, top) in enumerate(zip(cand_cit_vecs, cand_top_vecs)):
        cand_mat[idx] = _row(cit, top)

    return target_row, cand_mat


# ── Topic axes + nebulas ─────────────────────────────────────────────────────

def _find_top_topics(
    target_top: dict[str, float],
    cand_top_vecs: list[dict[str, float]],
    n: int = 5,
) -> list[tuple[str, float]]:
    """Return the target's top-weight topics, ranked primarily by target weight.

    The star map's axes should describe what the TARGET works on — not what
    distinguishes candidates from one another.  We therefore rank by target
    weight and only fall back on candidate-pool statistics as a tiebreaker
    when target weights are close.

    Topics below a small target-weight floor (1% of total) are rejected as
    OpenAlex-mislabelling noise — e.g. a single off-topic paper pulling an
    unrelated topic into an ML theorist's top-25.  The caller is expected
    to pass a small pool to the subfield-diversity step downstream, so a
    generous cap (3 × n) is fine here.
    """
    if not target_top:
        return []

    # 0.08 floor: a topic appearing in < 8% of the target's normalised top-25
    # is typically an OpenAlex mis-tag from one or two off-area collaboration
    # papers (e.g. "Sparse and Compressive Sensing" in an ML theorist's
    # record from one collaboration with a signal-processing coauthor).
    # These should not be displayed as an axis of the researcher's work.
    _MIN_TARGET_WEIGHT = 0.08

    n_cands = max(len(cand_top_vecs), 1) if cand_top_vecs else 1
    scores: list[tuple[str, float]] = []
    for tid, target_weight in target_top.items():
        if target_weight < _MIN_TARGET_WEIGHT:
            continue
        if cand_top_vecs:
            ws = np.array([v.get(tid, 0.0) for v in cand_top_vecs])
            coverage = float(np.count_nonzero(ws)) / n_cands
            std = float(ws.std())
            mean = float(ws.mean())
            saturation_penalty = 1.0 / (1.0 + 4.0 * mean)
            variance_bonus = std * np.sqrt(coverage) * saturation_penalty
        else:
            variance_bonus = 0.0
        # Target weight dominates; variance breaks ties among topics with
        # similar target weight.  0.05 coefficient ensures even a large
        # variance can't unseat a target topic with > 1.25× higher weight.
        score = target_weight + 0.05 * variance_bonus
        scores.append((tid, score))

    scores.sort(key=lambda x: -x[1])
    return scores[: 3 * n]


def _fetch_topic_metadata(
    topic_ids: list[str], timeout: float = 4.0,
) -> dict[str, dict]:
    """Fetch {topic_id → {name, subfield_id, field_id}} from OpenAlex.

    Subfield / field IDs are used downstream to deduplicate semantically
    overlapping topics (e.g. "Neural Networks" and "Machine Learning" both
    share the same subfield and should not both become topic axes).
    """
    def _fetch(tid: str) -> tuple[str, dict]:
        short = tid.split("/")[-1]
        info = {
            "name": short, "subfield_id": None, "subfield_name": None,
            "field_id": None, "field_name": None,
        }
        try:
            r = httpx.get(
                f"https://api.openalex.org/topics/{short}",
                headers={"User-Agent": "researcher-mapper"},
                timeout=timeout,
            )
            if r.status_code == 200:
                data = r.json()
                info["name"] = data.get("display_name") or short
                sf = data.get("subfield") or {}
                info["subfield_id"] = sf.get("id")
                info["subfield_name"] = sf.get("display_name")
                fld = data.get("field") or {}
                info["field_id"] = fld.get("id")
                info["field_name"] = fld.get("display_name")
        except Exception:
            pass
        return tid, info

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for tid, meta in pool.map(_fetch, topic_ids):
            out[tid] = meta
    return out


def _topic_color(idx: int, n: int) -> tuple[float, float, float]:
    """Return a pleasant (R,G,B) 0..1 for the i-th of n topics."""
    hue = (idx / max(n, 1) + 0.05) % 1.0     # slight offset from red
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return r, g, b


# ── Custom axes: CV research interests or target-title TF-IDF ────────────────

def _fetch_topic_names_bulk(
    topic_ids: list[str], timeout: float = 6.0,
) -> dict[str, str]:
    """Batched fetch of OpenAlex topic display names (up to 50 per call)."""
    out: dict[str, str] = {}
    short_ids = [tid.split("/")[-1] for tid in topic_ids if tid]
    batches = [short_ids[i:i + 50] for i in range(0, len(short_ids), 50)]

    def _fetch(batch: list[str]) -> list[tuple[str, str]]:
        try:
            r = httpx.get(
                "https://api.openalex.org/topics",
                # The /topics endpoint uses `ids.openalex` (NOT the works-style
                # `openalex_id`) for bulk ID filtering.
                params={"filter": "ids.openalex:" + "|".join(batch), "per-page": 50},
                headers={"User-Agent": "researcher-mapper"},
                timeout=timeout,
            )
            if r.status_code == 200:
                results = []
                for t in r.json().get("results", []):
                    full_id = t.get("id", "")
                    name = t.get("display_name") or full_id.split("/")[-1]
                    if full_id:
                        results.append((full_id, name))
                return results
        except Exception:
            pass
        return []

    if batches:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for batch_results in pool.map(_fetch, batches):
                for full_id, name in batch_results:
                    out[full_id] = name
    return out


def _collect_topic_ids_from_reasons(candidates: list[CandidateScore]) -> list[str]:
    """Return OpenAlex topic IDs mentioned in candidate reason strings."""
    found: set[str] = set()
    for cand in candidates:
        for reason in cand.reasons or []:
            for match in _TOPIC_ID_RE.finditer(reason):
                found.add(match.group(1))
    return sorted(found)


def _build_topic_name_lookup(
    topic_ids: list[str],
    known_names: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a short/full topic-ID lookup using known axis names plus OpenAlex."""
    lookup: dict[str, str] = {}
    for tid, name in (known_names or {}).items():
        short = tid.split("/")[-1]
        if short and name:
            lookup[short] = name
            lookup[f"https://openalex.org/{short}"] = name

    missing = [tid for tid in topic_ids if tid not in lookup]
    if missing:
        for full_id, name in _fetch_topic_names_bulk(
            [f"https://openalex.org/{tid}" for tid in missing]
        ).items():
            short = full_id.split("/")[-1]
            lookup[short] = name
            lookup[full_id] = name
    return lookup


def _replace_topic_ids_in_text(text: str, topic_names: dict[str, str]) -> str:
    """Replace T-prefixed OpenAlex topic IDs with display names in visible text."""
    def _replace(match: re.Match[str]) -> str:
        short = match.group(1)
        return topic_names.get(short, match.group(0))

    return _TOPIC_ID_RE.sub(_replace, text)


def _format_reason_text(
    reasons: list[str],
    topic_names: dict[str, str],
    limit: int = 3,
) -> str:
    cleaned = [_replace_topic_ids_in_text(reason, topic_names) for reason in reasons[:limit]]
    return " · ".join(cleaned)


def _candidate_hover_text(
    name: str,
    institution: str,
    bucket_label: str,
    side_label: str,
) -> str:
    """Small Plotly hover tooltip; detailed scores live in the pinned panel."""
    return (
        f"<b>{name}</b><br>"
        f"{institution}<br>"
        f"<i>{bucket_label}</i> ({side_label})"
        "<br><br><i>Click to open Google Scholar ↗</i>"
    )


def _dedup_similar_labels(labels: list[str], max_n: int) -> list[str]:
    """Drop labels whose token set substantially overlaps a label already kept.

    Handles cases like CV interests "Deep learning theory" vs "Theory of deep
    learning" — the second is near-duplicate and gets removed.
    """
    import re
    kept: list[str] = []
    kept_token_sets: list[set[str]] = []
    for label in labels:
        tokens = set(re.findall(r"[a-z][a-z]+", label.lower()))
        tokens -= {"and", "for", "the", "of", "in", "on", "with", "a", "an",
                   "to", "from", "by", "via"}
        if not tokens:
            continue
        duplicate = False
        for prior in kept_token_sets:
            overlap = len(tokens & prior) / max(len(tokens | prior), 1)
            if overlap >= 0.6:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(label)
        kept_token_sets.append(tokens)
        if len(kept) >= max_n:
            break
    return kept


def _corpus_tfidf_axes(
    corpus_docs: list[str], max_n: int = 12, min_df: int = 2,
) -> list[str]:
    """Extract representative multi-word phrases from a corpus (titles + CV).

    Each element of ``corpus_docs`` is treated as one document.  TF-IDF with
    bigram/trigram features ranks phrases by summed weight across the corpus.
    Returns a pool of up to ``max_n * 3`` candidate phrases so the caller's
    dedup step can still filter redundant overlaps.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [d for d in corpus_docs if d and len(d.split()) >= 3]
    if len(docs) < 5:
        return []
    # Soften min_df when the corpus is small so we still get matches.
    effective_min_df = min_df if len(docs) >= 15 else 1
    try:
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(2, 3),
            max_features=400,
            min_df=effective_min_df,
        )
        mat = vec.fit_transform(docs)
        feat_names = vec.get_feature_names_out()
        scores = np.asarray(mat.sum(axis=0)).flatten()
    except Exception as exc:
        log.warning("Corpus TF-IDF failed: %s", exc)
        return []

    order = np.argsort(-scores)
    return [feat_names[i] for i in order[:max_n * 3]]


# ── Token utilities (shared by axis-picking helpers) ─────────────────────────

_TOPIC_STOP = frozenset({
    "and", "or", "of", "the", "in", "on", "a", "an", "for", "to", "with",
    "by", "from", "using", "via", "applications", "techniques", "methods",
    "research", "advanced", "analysis",
})

# Generic tokens that are too broad to carry alignment signal on their own —
# a topic sharing ONLY "function" or "machine" with the target's vocabulary
# shouldn't qualify (e.g. "Neural dynamics and brain function" matching only
# via "function").  Still allowed to appear in multi-token matches alongside
# specific words.  Values are already stemmed.
_GENERIC_KEYWORDS = frozenset({
    "learning", "machine", "model", "deep", "algorithm", "classification",
    "method", "technique", "application", "system", "function", "dynamic",
    "process", "analysis", "data", "computation",
})


def _stem_word(w: str) -> str:
    """Tiny plural stripper so 'networks' → 'network', 'models' → 'model'."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def _topic_tokens(text: str) -> set[str]:
    """Content tokens of a topic/phrase: lowercased, stemmed, stopword-filtered."""
    import re
    raw = re.findall(r"[a-z]{3,}", (text or "").lower())
    return {_stem_word(t) for t in raw} - _TOPIC_STOP


def _topic_bigrams(text: str) -> set[str]:
    """Adjacent-word bigrams (stemmed, stopword-filtered).

    "Advanced Neural Network Applications" → {"neural network"}
    (advanced / applications are in _TOPIC_STOP and drop out, leaving the
    neural–network pair.)

    Bigram-level matching is much more specific than single-token matching:
    a topic only qualifies as an axis if it shares an actual phrase with
    the target's vocabulary, not just a common word like "function".
    """
    import re
    raw = re.findall(r"[a-z]{3,}", (text or "").lower())
    stemmed = [_stem_word(t) for t in raw]
    kept = [t for t in stemmed if t not in _TOPIC_STOP]
    return {f"{a} {b}" for a, b in zip(kept, kept[1:])}


def _build_openalex_axes(
    target_top: dict[str, float],
    cand_top_vecs: list[dict[str, float]],
    keyword_phrases: list[str],
    n_axes: int = 5,
) -> tuple[
    list[tuple[str, float]],          # top_topics_scored [(axis_id, score), …]
    dict[str, str],                    # axis_id → display name
    list[dict[str, float]],            # per-candidate axis weights
] | tuple[None, None, None]:
    """Pick OpenAlex topic IDs as axes, biased toward target vocabulary.

    Strategy
    --------
    1. Pool = target's top-weight OpenAlex topics (≥ 1.5% of weight, up to 30).
    2. Fetch display names + subfields via OpenAlex metadata API.
    3. **Keyword alignment** — token-overlap between the topic's display name
       and the target's CV+titles keyword phrases.  A topic whose name shares
       zero content words with the researcher's own vocabulary (e.g. "Sparse
       and Compressive Sensing" for a neural-net theorist) is demoted.
    4. **Score** = target_weight × (0.15 + 0.85 × alignment).  Ranks topics
       primarily by alignment, with weight as a tiebreaker.
    5. **Token-overlap dedup** removes near-duplicate axes ("Advanced Neural
       Network Applications" and "Neural Networks and Applications").
    6. For each picked axis, a **family** of related OpenAlex IDs is built
       (same token-overlap ≥ 0.35 against the axis's tokens).  Each
       candidate's axis-weight is the *sum* of their weights across the family
       — so a candidate tagged with any "neural-networks"-family topic
       contributes to the "Neural Networks" axis regardless of which exact
       OpenAlex ID OpenAlex used.

    Returns ``(None, None, None)`` when there's not enough signal
    (< 2 topics survive dedup).
    """
    if not target_top:
        return None, None, None

    pool = [(tid, w) for tid, w in
            sorted(target_top.items(), key=lambda x: -x[1])
            if w >= 0.015][:30]
    if not pool:
        return None, None, None
    pool_ids = [tid for tid, _ in pool]

    try:
        metadata = _fetch_topic_metadata(pool_ids)
    except Exception as exc:
        log.warning("Topic metadata fetch failed: %s", exc)
        metadata = {}

    # Two-level alignment:
    #   · bigrams (strong signal — shared phrase like "neural network")
    #   · specific tokens — matches on non-generic words (excluding
    #     "function" / "machine" / "learning" etc. that carry no distinctive
    #     meaning about the target)
    # A topic qualifies if it shares ≥ 1 bigram OR ≥ 2 specific tokens.
    keyword_tokens: set[str] = set()
    keyword_bigrams: set[str] = set()
    for phrase in keyword_phrases:
        keyword_tokens |= _topic_tokens(phrase)
        keyword_bigrams |= _topic_bigrams(phrase)
    specific_keywords = keyword_tokens - _GENERIC_KEYWORDS

    scored: list[tuple[str, float, int, str, set[str]]] = []
    for tid, weight in pool:
        name = metadata.get(tid, {}).get("name") or tid.split("/")[-1]
        name_toks = _topic_tokens(name)
        name_bigrams = _topic_bigrams(name)
        bigram_match = int(bool(name_bigrams & keyword_bigrams))
        specific_matches = len(name_toks & specific_keywords)
        qualifies = bigram_match >= 1 or specific_matches >= 2
        # Alignment is 1.0 for any qualified topic, 0 otherwise; ranking
        # among qualified topics falls back to raw target weight below.
        alignment = 1.0 if qualifies else 0.0
        score = weight * (0.05 + 0.95 * alignment)
        # Pack a qualifies flag into the tuple for filtering below.
        qual_flag = 1 if qualifies else 0
        scored.append((tid, score, qual_flag, name, name_toks))
    scored.sort(key=lambda x: -x[1])

    # Qualification filter
    qualified = [s for s in scored if s[2] >= 1]

    # Pick top-N with token-overlap dedup (looser threshold 0.6 keeps closely
    # related but distinct axes like "Neural Networks" vs "Model Reduction
    # and Neural Networks").
    picked: list[tuple[str, float, str, set[str]]] = []
    for tid, score, _, name, tokens in qualified:
        dup = False
        for _, _, _, prior_tokens in picked:
            if tokens and prior_tokens:
                overlap = len(tokens & prior_tokens) / max(len(tokens | prior_tokens), 1)
                if overlap >= 0.6:
                    dup = True
                    break
        if dup:
            continue
        picked.append((tid, score, name, tokens))
        if len(picked) >= n_axes:
            break

    # Need at least 2 axes to make an angular map meaningful.  If the
    # qualification filter was too strict, return None so the caller can
    # fall back.
    if len(picked) < 2:
        return None, None, None

    # Family expansion: for each axis, collect related IDs from the pool
    # so a candidate's weight across any neural-networks-family topic counts
    # for the "Neural Networks" axis.
    axis_families: dict[str, set[str]] = {}
    pool_tokens = {tid: _topic_tokens(metadata.get(tid, {}).get("name") or "")
                   for tid, _ in pool}
    for ax_tid, _, _, ax_tokens in picked:
        fam = {ax_tid}
        for other_tid, _ in pool:
            if other_tid == ax_tid:
                continue
            other_toks = pool_tokens.get(other_tid, set())
            if ax_tokens and other_toks:
                overlap = len(ax_tokens & other_toks) / max(len(ax_tokens | other_toks), 1)
                if overlap >= 0.35:
                    fam.add(other_tid)
        axis_families[ax_tid] = fam

    # Per-candidate axis weights (sum over family)
    axis_cand_top_vecs: list[dict[str, float]] = []
    for vec in cand_top_vecs:
        d: dict[str, float] = {}
        for ax_tid, fam in axis_families.items():
            s = sum(float(vec.get(t, 0.0)) for t in fam)
            if s > 0:
                d[ax_tid] = s
        axis_cand_top_vecs.append(d)

    top_topics_scored = [(tid, s) for tid, s, _, _ in picked]
    topic_names = {tid: name for tid, _, name, _ in picked}
    return top_topics_scored, topic_names, axis_cand_top_vecs


def _build_neighborhood_openalex_axes(
    target_top: dict[str, float],
    cand_top_vecs: list[dict[str, float]],
    keyword_phrases: list[str],
    n_axes: int = 5,
) -> tuple[
    list[tuple[str, float]],
    dict[str, str],
    list[dict[str, float]],
] | tuple[None, None, None]:
    """Pick star-map axes from target core topics plus nearby candidate topics."""
    n_axes = _clamp_topic_axis_count(n_axes)
    if not target_top and not cand_top_vecs:
        return None, None, None

    n_cands = max(len(cand_top_vecs), 1)
    cand_sum: dict[str, float] = {}
    cand_count: dict[str, int] = {}
    for vec in cand_top_vecs:
        seen: set[str] = set()
        for tid, weight in vec.items():
            w = float(weight)
            if w <= 0:
                continue
            cand_sum[tid] = cand_sum.get(tid, 0.0) + w
            if tid not in seen:
                cand_count[tid] = cand_count.get(tid, 0) + 1
                seen.add(tid)

    cand_stats: dict[str, tuple[float, float, float]] = {}
    for tid, total in cand_sum.items():
        mean = total / n_cands
        coverage = cand_count.get(tid, 0) / n_cands
        signal = mean * np.sqrt(max(coverage, 0.0))
        cand_stats[tid] = (mean, coverage, signal)

    target_pool = [
        tid for tid, weight in sorted(target_top.items(), key=lambda x: -x[1])
        if weight >= 0.015
    ][:max(n_axes * 8, 40)]

    coverage_floor = min(0.20, max(1.0 / n_cands, 0.06))
    neighbor_pool = [
        tid for tid, (_, coverage, signal) in
        sorted(cand_stats.items(), key=lambda x: -x[1][2])
        if coverage >= coverage_floor and signal >= 0.004
    ][:max(n_axes * 10, 50)]

    pool_ids = list(dict.fromkeys(target_pool + neighbor_pool))
    if not pool_ids:
        return None, None, None

    try:
        metadata = _fetch_topic_metadata(pool_ids)
    except Exception as exc:
        log.warning("Topic metadata fetch failed: %s", exc)
        metadata = {}

    anchor_names = [
        metadata.get(tid, {}).get("name") or tid.split("/")[-1]
        for tid, weight in target_top.items()
        if weight >= 0.04 and tid in metadata
    ]
    keyword_tokens: set[str] = set()
    keyword_bigrams: set[str] = set()
    for phrase in list(keyword_phrases) + anchor_names:
        keyword_tokens |= _topic_tokens(phrase)
        keyword_bigrams |= _topic_bigrams(phrase)
    specific_keywords = keyword_tokens - _GENERIC_KEYWORDS

    target_fields = {
        metadata.get(tid, {}).get("field_id")
        for tid, weight in target_top.items()
        if weight >= 0.04
    } - {None}
    target_subfields = {
        metadata.get(tid, {}).get("subfield_id")
        for tid, weight in target_top.items()
        if weight >= 0.04
    } - {None}

    scored: list[tuple[str, float, str, str, set[str], str | None]] = []
    for tid in pool_ids:
        target_weight = float(target_top.get(tid, 0.0))
        _mean, coverage, neighbor_signal = cand_stats.get(tid, (0.0, 0.0, 0.0))
        meta = metadata.get(tid, {})
        name = meta.get("name") or tid.split("/")[-1]
        tokens = _topic_tokens(name)
        bigrams = _topic_bigrams(name)
        lexical = bool(bigrams & keyword_bigrams) or len(tokens & specific_keywords) >= 2
        generic_only = bool(tokens) and not (tokens - _GENERIC_KEYWORDS)
        same_subfield = meta.get("subfield_id") in target_subfields
        same_field = meta.get("field_id") in target_fields

        core = target_weight >= 0.05 or (target_weight >= 0.025 and lexical)
        neighborhood = (
            neighbor_signal >= 0.004
            and coverage >= coverage_floor
            and not generic_only
            and (lexical or same_subfield or same_field or not target_top)
        )
        if not core and not neighborhood:
            continue

        if lexical:
            alignment = 1.0
        elif same_subfield:
            alignment = 0.60
        elif same_field:
            alignment = 0.45
        else:
            alignment = 0.25

        score = 0.60 * target_weight + 2.50 * neighbor_signal + 0.05 * alignment
        if generic_only:
            score *= 0.20
        role = "core" if core else "neighborhood"
        scored.append((tid, score, role, name, tokens, meta.get("subfield_id")))

    if not scored:
        return None, None, None
    scored.sort(key=lambda x: -x[1])

    picked: list[tuple[str, float, str, set[str], str | None, str]] = []

    def _is_duplicate(tokens: set[str], subfield_id: str | None) -> bool:
        for _, _, _, prior_tokens, prior_subfield, _ in picked:
            if tokens and prior_tokens:
                overlap = len(tokens & prior_tokens) / max(len(tokens | prior_tokens), 1)
                if overlap >= 0.6:
                    return True
            if subfield_id and prior_subfield == subfield_id:
                return True
        return False

    def _try_add(item: tuple[str, float, str, str, set[str], str | None]) -> bool:
        tid, score, role, name, tokens, subfield_id = item
        if _is_duplicate(tokens, subfield_id):
            return False
        picked.append((tid, score, name, tokens, subfield_id, role))
        return True

    core_quota = max(1, int(round(n_axes * 0.60)))
    for item in [s for s in scored if s[2] == "core"]:
        if sum(1 for p in picked if p[5] == "core") >= core_quota:
            break
        _try_add(item)
    for item in [s for s in scored if s[2] == "neighborhood"]:
        if len(picked) >= n_axes:
            break
        _try_add(item)
    for item in scored:
        if len(picked) >= n_axes:
            break
        _try_add(item)

    if len(picked) < 2:
        return None, None, None

    axis_families: dict[str, set[str]] = {}
    pool_tokens = {
        tid: _topic_tokens(metadata.get(tid, {}).get("name") or "")
        for tid in pool_ids
    }
    for ax_tid, _, _, ax_tokens, _, _ in picked:
        fam = {ax_tid}
        for other_tid in pool_ids:
            if other_tid == ax_tid:
                continue
            other_toks = pool_tokens.get(other_tid, set())
            if ax_tokens and other_toks:
                overlap = len(ax_tokens & other_toks) / max(len(ax_tokens | other_toks), 1)
                if overlap >= 0.35:
                    fam.add(other_tid)
        axis_families[ax_tid] = fam

    axis_cand_top_vecs: list[dict[str, float]] = []
    for vec in cand_top_vecs:
        d: dict[str, float] = {}
        for ax_tid, fam in axis_families.items():
            value = sum(float(vec.get(t, 0.0)) for t in fam)
            if value > 0:
                d[ax_tid] = value
        axis_cand_top_vecs.append(d)

    top_topics_scored = [(tid, score) for tid, score, _, _, _, _ in picked]
    topic_names = {tid: name for tid, _, name, _, _, _ in picked}
    return top_topics_scored, topic_names, axis_cand_top_vecs


def _custom_axes_from_labels(
    axis_labels: list[str],
    all_cands: list[CandidateScore],
    summaries_by_id: dict,
    n_axes: int = 5,
) -> tuple[
    list[tuple[str, float]],          # top_topics_scored (label, pseudo-score)
    dict[str, str],                    # topic_names (label → label)
    list[dict[str, float]],            # per-candidate weights keyed by label
] | tuple[None, None, None]:
    """Score candidates against a list of free-text axis labels via TF-IDF cosine.

    1. Fetch OpenAlex display names for every candidate's top_topics.
    2. Build one "topic-name bag" per candidate, each topic name repeated in
       proportion to its weight so TF-IDF captures the weighting.
    3. Fit a single TF-IDF vectoriser over axis labels + candidate bags so
       they share vocabulary, then return cosine similarities (candidates × axes).

    Returns (None, None, None) if there's not enough signal to proceed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    labels = _dedup_similar_labels(axis_labels, max_n=n_axes)
    if not labels:
        return None, None, None

    # Fetch display names for all topic IDs any candidate touches
    all_topic_ids: set[str] = set()
    for cand in all_cands:
        s = summaries_by_id.get(cand.candidate_id)
        if s and s.top_topics:
            all_topic_ids.update(s.top_topics.keys())
    if not all_topic_ids:
        return None, None, None
    try:
        topic_name_map = _fetch_topic_names_bulk(list(all_topic_ids))
    except Exception as exc:
        log.warning("Topic-name bulk fetch failed: %s", exc)
        topic_name_map = {}

    # Build weighted topic-name bag per candidate
    cand_docs: list[str] = []
    for cand in all_cands:
        s = summaries_by_id.get(cand.candidate_id)
        top = s.top_topics if (s and s.top_topics) else {}
        parts: list[str] = []
        for tid, w in top.items():
            name = topic_name_map.get(tid, "")
            if not name:
                continue
            rep = max(int(round(w * 40)), 1)       # weight → repetition count
            parts.extend([name] * rep)
        cand_docs.append(" ".join(parts) if parts else "")

    # If nearly every candidate doc is empty, there's nothing to score against.
    non_empty = sum(1 for d in cand_docs if d)
    if non_empty < max(len(all_cands) * 0.4, 3):
        return None, None, None

    all_docs = list(labels) + cand_docs
    try:
        vec = TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=600,
        )
        mat = vec.fit_transform(all_docs)
    except Exception as exc:
        log.warning("Custom-axes TF-IDF failed: %s", exc)
        return None, None, None

    n_labels = len(labels)
    axis_mat = mat[:n_labels]
    cand_mat = mat[n_labels:]
    axis_norm = normalize(axis_mat).toarray()
    cand_norm = normalize(cand_mat).toarray()
    sim = cand_norm @ axis_norm.T         # (N, A)

    # If the whole similarity matrix is essentially zero, the labels share no
    # vocabulary with the candidate topic names — bail and let the caller fall
    # back to the OpenAlex path.
    if float(sim.max()) < 0.05:
        return None, None, None

    # Assemble outputs
    top_topics_scored = [(lbl, float(sim[:, i].mean())) for i, lbl in enumerate(labels)]
    topic_names = {lbl: lbl for lbl in labels}
    new_cand_top_vecs: list[dict[str, float]] = []
    for i in range(cand_mat.shape[0]):
        d: dict[str, float] = {}
        for a, lbl in enumerate(labels):
            v = float(sim[i, a])
            if v > 0:
                d[lbl] = v
        new_cand_top_vecs.append(d)

    return top_topics_scored, topic_names, new_cand_top_vecs


# ── Directional polar layout (topic-based angles, citation-based radii) ─────

def _directional_layout(
    target_row: np.ndarray,
    cand_mat: np.ndarray,
    cand_top_vecs: list[dict[str, float]],
    top_topics: list[tuple[str, float]],
    r_range: tuple[float, float] = (0.8, 3.3),
    seed: int = 7,
) -> tuple[np.ndarray, dict[str, float]]:
    """Place candidates at topic-weighted angles + citation-distance radii.

    Returns (coords, topic_angle_map).

    - Each of the N top topics is assigned an evenly-spaced angle around 2π.
    - Each candidate's angle is the weighted circular mean of the top topics
      (weights = candidate's own top_topics entries for those IDs).  A
      candidate with a dominant topic sits near that topic's direction;
      mixed-topic candidates sit between.
    - Radius = citation+topic distance from target, mapped linearly to r_range.
      A small bit of angular and radial jitter avoids perfect overlaps.
    """
    n = len(cand_top_vecs)
    if n == 0:
        return np.zeros((0, 2)), {}

    # Topic angles around the circle
    topic_ids = [tid for tid, _ in top_topics]
    N_t = len(topic_ids)
    topic_angle_map: dict[str, float] = {}
    if N_t > 0:
        for i, tid in enumerate(topic_ids):
            topic_angle_map[tid] = 2 * np.pi * i / N_t

    # Per-topic mean across candidates — subtracting this from each candidate's
    # weight turns the signal into "deviation from the group average".  A
    # topic that every candidate shares equally contributes zero direction,
    # so angles end up driven by what differentiates researchers.
    topic_means: dict[str, float] = {}
    if N_t > 0:
        for tid in topic_ids:
            vals = [float(v.get(tid, 0.0)) for v in cand_top_vecs]
            topic_means[tid] = float(np.mean(vals)) if vals else 0.0

    rng = np.random.default_rng(seed)
    angles = np.zeros(n)
    for i, vec in enumerate(cand_top_vecs):
        if N_t == 0:
            angles[i] = rng.uniform(0, 2 * np.pi)
            continue
        # Weighted circular mean over ABOVE-AVERAGE topic deviations
        wx, wy, total = 0.0, 0.0, 0.0
        for tid in topic_ids:
            dev = float(vec.get(tid, 0.0)) - topic_means[tid]
            if dev <= 0:             # below or at mean → not a defining topic
                continue
            th = topic_angle_map[tid]
            wx += dev * np.cos(th)
            wy += dev * np.sin(th)
            total += dev
        if total > 1e-6 and (abs(wx) > 1e-6 or abs(wy) > 1e-6):
            angles[i] = float(np.arctan2(wy, wx))
        else:
            # Candidate is average on every top topic → assign a spread-out
            # angle based on their index to fill gaps rather than random bunch.
            angles[i] = (2 * np.pi) * (i / max(n, 1))

    # Small angular jitter so identical-profile candidates don't stack
    angles = angles + rng.uniform(-0.15, 0.15, n)

    # Radial: citation/topic distance from target
    if cand_mat.shape[1] > 0 and target_row.size > 0:
        sim = cand_mat @ target_row
        dist = np.clip(1.0 - sim, 0.0, 1.0)
    else:
        dist = np.linspace(0.4, 0.9, n)

    r_min, r_max = r_range
    d_min, d_max = float(dist.min()), float(dist.max())
    span = d_max - d_min
    if span < 0.25:
        # Distances are all bunched together — stretch via absolute distance
        # so a distance of 0.9 still lands near the outside of the band.
        radii = r_min + dist * (r_max - r_min)
    else:
        norm = (dist - d_min) / span
        radii = r_min + norm * (r_max - r_min)

    # Small radial jitter to avoid perfect circles
    radii = radii + rng.uniform(-0.08, 0.08, n)
    radii = np.clip(radii, r_min * 0.85, r_max)

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y]), topic_angle_map


def _separate_overlapping(
    coords: np.ndarray,
    min_dist: float,
    r_max: float,
    iters: int = 60,
) -> np.ndarray:
    """Iteratively push too-close points apart until none overlap.

    Pairs closer than `min_dist` are repelled along their separating vector
    by half the overshoot each. Points are clamped to `r_max` after each
    sweep so stars stay within the design envelope. Origin is protected
    by a soft floor of `min_dist / 2` on the radius.
    """
    n = coords.shape[0]
    if n < 2:
        return coords
    pts = coords.astype(float).copy()
    r_floor = min_dist * 0.5
    for _ in range(iters):
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        overlap = np.clip(min_dist - dist, 0.0, None)
        if overlap.max() < 1e-4:
            break
        safe = np.where(dist > 1e-6, dist, 1.0)
        unit = diff / safe[..., None]
        pts = pts + (unit * (overlap * 0.5)[..., None]).sum(axis=1)
        # Keep stars away from the origin (target star sits there).
        r = np.linalg.norm(pts, axis=1)
        too_inner = r < r_floor
        if too_inner.any():
            safe_r = np.where(r > 1e-6, r, 1.0)
            pts[too_inner] = pts[too_inner] / safe_r[too_inner, None] * r_floor
        # Clamp to outer envelope.
        r = np.linalg.norm(pts, axis=1)
        too_outer = r > r_max
        if too_outer.any():
            pts[too_outer] = pts[too_outer] / r[too_outer, None] * r_max
    return pts


def _build_feature_matrix(
    candidates: list[CandidateScore],
    summaries_by_id: dict,
) -> np.ndarray:
    """Legacy helper used by the clustering step (doesn't need target)."""
    cit_vecs, top_vecs = _collect_feature_dict(candidates, summaries_by_id)
    # Build without target — we just want per-candidate vectors for clustering
    _target_row, cand_mat = _build_joint_matrix({}, {}, cit_vecs, top_vecs)
    return cand_mat


def _cluster_rows(feat_mat: np.ndarray, n_clusters: int) -> np.ndarray:
    n = feat_mat.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n <= n_clusters or feat_mat.shape[1] == 0:
        return np.arange(n) % max(n_clusters, 1)
    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return km.fit_predict(feat_mat)
    except Exception:
        return np.zeros(n, dtype=int)


def _mst_edges(positions: np.ndarray) -> list[tuple[int, int]]:
    n = positions.shape[0]
    if n < 2:
        return []
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mst = minimum_spanning_tree(dist).toarray()
    return [(int(i), int(j)) for i, j in zip(*np.where(mst > 0))]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── CSS + JavaScript for animations and interactivity ────────────────────────

_CUSTOM_CSS = """
<style>
html, body {
    margin: 0; padding: 0;
    width: 100%; height: 100%;
    overflow: hidden;
    background: #080A14;
}
/* The Plotly-rendered wrapper div — fill the whole viewport and respond
   to window resizes via the `responsive: true` config option. */
.plotly-graph-div,
.js-plotly-plot,
.js-plotly-plot > div {
    width: 100vw !important;
    height: 100vh !important;
}

/* Radial gradient giving the canvas a gentle galactic halo */
.plotly-graph-div {
    background: radial-gradient(ellipse at center,
                #10142A 0%, #0B0D18 50%, #060710 100%) !important;
}

/* The white target layers provide plenty of visual interest statically,
   no per-trace CSS animation needed (nth-child selectors are fragile against
   changing topic counts). */

/* Twinkle for the (static) background twinkle stars.
   We animate opacity of the dot group periodically. */
@keyframes twinkle {
    0%, 100% { opacity: 0.10; }
    50%      { opacity: 0.28; }
}
.js-plotly-plot .scatterlayer .trace:nth-child(1) .points path {
    animation: twinkle 4s ease-in-out infinite;
}

/* Shooting-star container: absolute-positioned, randomly launched by JS */
#rm-meteors { position: absolute; inset: 0; pointer-events: none;
              overflow: hidden; z-index: 5; }
.rm-meteor {
    position: absolute; width: 140px; height: 1.5px;
    background: linear-gradient(90deg, rgba(255,255,255,0.0) 0%,
                                       rgba(200,220,255,0.9) 80%,
                                       rgba(255,255,255,1.0) 100%);
    border-radius: 1px;
    filter: drop-shadow(0 0 6px rgba(180,210,255,0.9));
    transform-origin: right center;
}

/* Info panel (pinned, updated on hover) */
#rm-info {
    position: absolute; right: 18px; top: 78px;
    width: 280px; max-width: 30vw;
    padding: 14px 16px;
    background: rgba(20, 26, 44, 0.82);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    color: #EDEFF8;
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
    font-size: 12.5px;
    line-height: 1.45;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    opacity: 0; transform: translateY(-6px);
    transition: opacity 0.25s ease, transform 0.25s ease;
    pointer-events: none;
    z-index: 10;
}
#rm-info.rm-show { opacity: 1; transform: translateY(0); }
#rm-info .rm-name { font-size: 15px; font-weight: 700;
                    margin-bottom: 4px; color: #FFFFFF; }
#rm-info .rm-inst { font-size: 11.5px; color: #B8C4DA;
                    margin-bottom: 8px; font-style: italic; }
#rm-info .rm-bucket { display: inline-block;
    padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
    background: rgba(255,255,255,0.08);
    margin-bottom: 8px;
}
#rm-info .rm-stats { color: #C8D2E8; font-size: 11.5px;
                     margin-bottom: 6px; }
#rm-info .rm-hint { font-size: 10.5px; color: #8B95B0;
                    margin-top: 8px; font-style: italic; }

/* Default cursor = grab (pan), pointer on star hover (set by JS) */
.js-plotly-plot .nsewdrag { cursor: grab !important; }
.js-plotly-plot .nsewdrag:active { cursor: grabbing !important; }
</style>
"""


def _post_script(
    researcher_data_json: str,
    zoom_limit_x: float, zoom_limit_y: float,
    max_zoom_out_x: float, max_zoom_out_y: float,
) -> str:
    """JS that wires up meteors, info panel, zoom limit, and Scholar clicks."""
    return f"""
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var infoData = {researcher_data_json};
    var ZOOM_LIMIT_X = {zoom_limit_x:.2f};
    var ZOOM_LIMIT_Y = {zoom_limit_y:.2f};
    var MAX_ZOOM_OUT_X = {max_zoom_out_x:.2f};
    var MAX_ZOOM_OUT_Y = {max_zoom_out_y:.2f};

    // ── Double-click reset guard ───────────────────────────────────
    // When the user double-clicks, Plotly default autoranges to fit ALL data
    // — including the background stars 3× outside the initial view.  We
    // intercept the autorange event and snap back to the initial clean view.
    // (Regular zoom in/out via scroll or drag-select is NOT touched, so small
    //  zoom-ins always take effect.)
    var isAdjusting = false;

    gd.on('plotly_relayout', function(evt) {{
        if (isAdjusting || !evt) return;
        var isAutorange = (
            evt['xaxis.autorange'] === true ||
            evt['yaxis.autorange'] === true
        );
        if (!isAutorange) return;

        try {{
            isAdjusting = true;
            Plotly.relayout(gd, {{
                'xaxis.range': [-ZOOM_LIMIT_X, ZOOM_LIMIT_X],
                'yaxis.range': [-ZOOM_LIMIT_Y, ZOOM_LIMIT_Y]
            }}).then(function() {{
                setTimeout(function() {{ isAdjusting = false; }}, 150);
            }}).catch(function() {{ isAdjusting = false; }});
        }} catch(e) {{ isAdjusting = false; }}
    }});

    // ── Info panel ─────────────────────────────────────────────────
    var panel = document.createElement('div');
    panel.id = 'rm-info';
    panel.innerHTML = '<div class="rm-hint">Hover a star for details · ' +
                      'Click to open Google Scholar</div>';
    document.body.appendChild(panel);

    function showPanel(name) {{
        var d = infoData[name];
        if (!d) return;
        var html = '';
        html += '<div class="rm-name">' + d.name + '</div>';
        if (d.institution) html += '<div class="rm-inst">' + d.institution + '</div>';
        if (d.bucket) {{
            var colorSwatch = '<span style="background:' + d.color + ';' +
                              'width:8px;height:8px;border-radius:50%;' +
                              'display:inline-block;margin-right:6px;' +
                              'box-shadow:0 0 6px ' + d.color + ';"></span>';
            html += '<div class="rm-bucket">' + colorSwatch + d.bucket + ' · ' +
                    d.side + '</div>';
        }}
        if (d.stats) html += '<div class="rm-stats">' + d.stats + '</div>';
        if (d.reasons) html += '<div class="rm-stats">' + d.reasons + '</div>';
        html += '<div class="rm-hint">Click to open in Google Scholar ↗</div>';
        panel.innerHTML = html;
        panel.classList.add('rm-show');
    }}

    // ── Shooting stars ─────────────────────────────────────────────
    var meteorBox = document.createElement('div');
    meteorBox.id = 'rm-meteors';
    gd.appendChild(meteorBox);

    function launchMeteor() {{
        var m = document.createElement('div');
        m.className = 'rm-meteor';
        var startX = Math.random() * gd.clientWidth * 0.3 + gd.clientWidth * 0.05;
        var startY = Math.random() * gd.clientHeight * 0.4;
        var angle = 18 + Math.random() * 12;
        var distance = 700 + Math.random() * 400;
        var duration = 1.3 + Math.random() * 0.9;
        m.style.left = startX + 'px';
        m.style.top = startY + 'px';
        m.style.transform = 'rotate(' + angle + 'deg)';
        m.style.opacity = '0';
        meteorBox.appendChild(m);
        // Force a reflow so the browser paints the initial state BEFORE we
        // change transform/opacity — without this, some meteors would skip
        // the transition and appear/disappear in place.
        void m.offsetWidth;
        // Double-rAF belt-and-braces for browsers that batch style mutations.
        requestAnimationFrame(function() {{
            requestAnimationFrame(function() {{
                m.style.transition = 'transform ' + duration + 's linear, ' +
                                     'opacity ' + (duration * 0.35) + 's ease-out';
                m.style.transform = 'rotate(' + angle + 'deg) translateX(' +
                                    distance + 'px)';
                m.style.opacity = '1';
            }});
        }});
        setTimeout(function() {{ m.style.opacity = '0'; }}, duration * 750);
        setTimeout(function() {{ m.remove(); }}, duration * 1200);
    }}
    // Launch a meteor every 3-7s
    setInterval(function() {{
        if (Math.random() < 0.7) launchMeteor();
    }}, 2500);
    setTimeout(launchMeteor, 900);

    // ── Plotly event wiring ────────────────────────────────────────
    if (gd.on) {{
        gd.on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                var name = data.points[0].customdata;
                if (name) {{
                    var url = 'https://scholar.google.com/scholar?q=' +
                              encodeURIComponent('"' + name + '"');
                    window.open(url, '_blank');
                }}
            }}
        }});

        gd.on('plotly_hover', function(data) {{
            if (data.points && data.points[0]) {{
                var name = data.points[0].customdata;
                if (name) {{
                    var dragLayer = gd.querySelector('.nsewdrag');
                    if (dragLayer) dragLayer.style.cursor = 'pointer';
                    showPanel(name);
                }}
            }}
        }});
        gd.on('plotly_unhover', function() {{
            var dragLayer = gd.querySelector('.nsewdrag');
            if (dragLayer) dragLayer.style.cursor = '';
        }});
    }}
    """


# ── Public entry point ────────────────────────────────────────────────────────

def generate_star_map(
    target_profile: ResearcherProfile,
    israel_list: list[CandidateScore],
    world_list: list[CandidateScore],
    summaries_by_id: dict,
    output_path: Path,
    target_fv=None,            # FeatureVector — target's own feature vector
    n_clusters: int = 5,
    n_topic_axes: int = 5,
) -> None:
    """Render an interactive star-map HTML page for the mapping result."""
    all_cands = list(israel_list) + list(world_list)
    if not all_cands:
        log.warning("Star map: no candidates to plot; skipping")
        return
    n_topic_axes = _clamp_topic_axis_count(n_topic_axes)

    # ── Target feature vectors ───────────────────────────────────────────────
    target_cit: dict[str, float] = {}
    target_top: dict[str, float] = {}
    if target_fv is not None:
        target_cit = dict(getattr(target_fv, "citation_vector", {}) or {})
        target_top = dict(getattr(target_fv, "topic_vector", {}) or {})

    # ── Topic axes — OpenAlex IDs with keyword-alignment, else fallbacks ────
    # Primary strategy: pick axis IDs from OpenAlex's standardised topic
    # taxonomy (so candidates can be scored via direct ID match, no
    # vocabulary mismatch) — but rank them using the target's own
    # CV + paper-title vocabulary, so noisy auto-assigned topics like
    # "Sparse and Compressive Sensing" (for an ML theorist) get demoted.
    all_cit, all_top = _collect_feature_dict(all_cands, summaries_by_id)

    top_topics_scored = None
    topic_names = None
    axis_cand_top_vecs = None
    chosen_source = "openalex"

    cv_interests = list(target_profile.cv_research_interests or [])
    target_titles = [p.title for p in target_profile.publications if p.title]
    corpus = cv_interests + target_titles
    keyword_phrases = _corpus_tfidf_axes(corpus, max_n=25) if corpus else []

    # Primary path: OpenAlex topic IDs ranked by keyword alignment
    result = _build_neighborhood_openalex_axes(
        target_top, all_top, keyword_phrases, n_axes=n_topic_axes,
    )
    if result[0] is not None:
        top_topics_scored, topic_names, axis_cand_top_vecs = result
        chosen_source = (
            "openalex+cv+titles" if cv_interests
            else "openalex+titles" if target_titles
            else "openalex"
        )

    # Fallback 1: free-text axes (when OpenAlex target_top is empty / matches fail)
    if top_topics_scored is None and keyword_phrases:
        r2 = _custom_axes_from_labels(
            keyword_phrases, all_cands, summaries_by_id, n_axes=n_topic_axes,
        )
        if r2[0] is not None:
            top_topics_scored, topic_names, axis_cand_top_vecs = r2
            chosen_source = "cv+titles"

    # Fallback 2: legacy OpenAlex-only selection (no keyword guidance)
    if top_topics_scored is None:
        pool_scored = _find_top_topics(target_top, all_top, n=max(n_topic_axes, 5))
        pool_ids = [tid for tid, _ in pool_scored]
        try:
            topic_metadata = _fetch_topic_metadata(pool_ids) if pool_ids else {}
        except Exception as exc:
            log.warning("Could not fetch topic metadata: %s", exc)
            topic_metadata = {}
        top_topics_scored = pool_scored[:n_topic_axes]
        top_topic_ids_tmp = [tid for tid, _ in top_topics_scored]
        topic_names = {
            tid: topic_metadata.get(tid, {}).get("name") or tid.split("/")[-1]
            for tid in top_topic_ids_tmp
        }
        axis_cand_top_vecs = all_top
        chosen_source = "openalex-only"

    top_topic_ids = [tid for tid, _ in top_topics_scored]
    log.info(
        "Star map axes (source=%s): %s",
        chosen_source,
        [topic_names.get(t, t) for t in top_topic_ids],
    )
    reason_topic_names = _build_topic_name_lookup(
        _collect_topic_ids_from_reasons(all_cands),
        topic_names,
    )

    # ── Feature matrices + directional layout ───────────────────────────────
    all_tgt_row, all_cand_mat = _build_joint_matrix(
        target_cit, target_top, all_cit, all_top
    )
    coords, topic_angle_map = _directional_layout(
        all_tgt_row, all_cand_mat, axis_cand_top_vecs, top_topics_scored,
        r_range=(0.9, 3.4),
    )

    # Safety: clamp any star that ended up outside the design envelope.
    TARGET_MAX_R = 3.45
    if coords.shape[0] > 0:
        radii = np.linalg.norm(coords, axis=1)
        max_r = radii.max() if radii.size > 0 else 0.0
        if max_r > TARGET_MAX_R:
            coords = coords * (TARGET_MAX_R / max_r)

    # Push visually colliding stars apart while preserving layout intent.
    # min_dist is chosen so the size-26 glows don't overlap their cores.
    coords = _separate_overlapping(coords, min_dist=0.38, r_max=TARGET_MAX_R)

    # Room for the topic nebulas on the outer ring.
    nebula_radius = TARGET_MAX_R + 0.95

    # Dynamic plot bounds: large enough to include nebulas + labels.
    x_extent = nebula_radius + 1.1
    y_extent = nebula_radius + 1.1

    # Zoom-out cap — stays well within the background star coverage.
    zoom_limit_x = x_extent * 1.25
    zoom_limit_y = y_extent * 1.25
    max_zoom_out_x = x_extent * 1.35
    max_zoom_out_y = y_extent * 1.35

    # Background stars extend well past the initial view.
    bg_extent_x = x_extent * 3.0
    bg_extent_y = y_extent * 3.0

    # ── Disjoint constellation edges ─────────────────────────────────────────
    # More clusters → smaller groups; then we drop MST edges that are
    # unusually long so the result looks like disconnected constellations
    # rather than one connected tree per cluster.
    full_mat = _build_feature_matrix(all_cands, summaries_by_id)
    labels = _cluster_rows(full_mat, n_clusters)

    edge_xs: list[float | None] = []
    edge_ys: list[float | None] = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        edges = _mst_edges(coords[idx])
        if not edges:
            continue
        # Compute edge lengths; keep only the shorter half + any edge ≤ 1.3
        lengths = [float(np.linalg.norm(coords[idx[i]] - coords[idx[j]]))
                   for (i, j) in edges]
        if lengths:
            median_len = float(np.median(lengths))
            cutoff = min(max(median_len * 1.15, 0.6), 1.4)
        else:
            cutoff = 1.2
        for (i, j), L in zip(edges, lengths):
            if L > cutoff:
                continue
            a, b = idx[i], idx[j]
            edge_xs.extend([coords[a, 0], coords[b, 0], None])
            edge_ys.extend([coords[a, 1], coords[b, 1], None])

    fig = go.Figure()

    # ── Background twinkle stars — cover full zoom-out area ──────────────────
    rng = np.random.default_rng(42)
    n_bg = 700
    bg_x = rng.uniform(-bg_extent_x, bg_extent_x, n_bg)
    bg_y = rng.uniform(-bg_extent_y, bg_extent_y, n_bg)
    bg_size = 1.5 + 5 * rng.random(n_bg) ** 2
    fig.add_trace(go.Scatter(
        x=bg_x, y=bg_y, mode="markers",
        marker=dict(size=bg_size, color="white", opacity=0.16),
        hoverinfo="skip", showlegend=False, name="background",
    ))

    # ── Topic nebulas (one per top topic, labeled) ──────────────────────────
    # Placed at the angle of each topic, just outside the candidate ring.
    # The nebula colour matches the topic sector so the reader can tell at a
    # glance which direction is which topic.
    n_topics_used = len(top_topic_ids)
    for i, tid in enumerate(top_topic_ids):
        th = topic_angle_map[tid]
        nx = nebula_radius * np.cos(th)
        ny = nebula_radius * np.sin(th)
        r, g, b = _topic_color(i, n_topics_used)

        # Soft cloud: three overlapping blobs for a nebula feel
        for size, alpha in [(320, 0.12), (200, 0.18), (110, 0.22)]:
            fig.add_trace(go.Scatter(
                x=[nx], y=[ny], mode="markers",
                marker=dict(
                    size=size,
                    color=f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})",
                    line=dict(width=0),
                ),
                hoverinfo="skip", showlegend=False,
            ))

        # Label slightly further out so it doesn't overlap the blob core
        label_r = nebula_radius + 0.45
        lx = label_r * np.cos(th)
        ly = label_r * np.sin(th)
        name = topic_names.get(tid, tid.split("/")[-1])
        # Wrap long labels: insert <br> after ~18 chars at the next space
        wrapped = name
        if len(name) > 22:
            # break at the midpoint's nearest space
            mid = len(name) // 2
            left_space = name.rfind(" ", 0, mid)
            right_space = name.find(" ", mid)
            brk = left_space if (left_space != -1 and (right_space == -1 or mid - left_space <= right_space - mid)) else right_space
            if brk != -1:
                wrapped = name[:brk] + "<br>" + name[brk + 1:]
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="text",
            text=[f"<b>{wrapped}</b>"],
            textfont=dict(
                color=f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.95)",
                size=12, family="Arial",
            ),
            hoverinfo="skip", showlegend=False,
        ))

    # Constellation edges — two-layer (wide faint glow + narrow bright core)
    # so the lines read clearly against the dark nebulas.
    if edge_xs:
        fig.add_trace(go.Scatter(      # outer glow
            x=edge_xs, y=edge_ys, mode="lines",
            line=dict(color="rgba(180, 210, 255, 0.18)", width=3.0),
            hoverinfo="skip", showlegend=False, name="constellations-glow",
        ))
        fig.add_trace(go.Scatter(      # bright inner line
            x=edge_xs, y=edge_ys, mode="lines",
            line=dict(color="rgba(220, 235, 255, 0.75)", width=1.3),
            hoverinfo="skip", showlegend=False, name="constellations",
        ))

    # ── Central target star ──────────────────────────────────────────────────
    target_inst = (
        target_profile.current_institution.name
        if target_profile.current_institution else ""
    )
    target_hover = (
        f"<b>{target_profile.canonical_name}</b><br>"
        f"<i>Target researcher</i><br>"
        f"{target_inst or '—'}"
    )
    # ── White target star ────────────────────────────────────────────────
    # Concentric star layers, drawn largest-first. Because star shapes have
    # pointed tips, the outer layers peek out between the points of the
    # inner layers — producing a layered white glow.
    # Final overlay: a tiny white dot for unmistakable visual centring.
    white_layers = [
        ("#FFFFFF", 18),
        ("#FFFFFF", 23),
        ("#FFFFFF", 29),
        ("#FFFFFF", 35),
        ("#FFFFFF", 42),
        ("#FFFFFF", 50),
    ]
    # Soft warm halo first (matches the radiating glow)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(symbol="circle", size=80,
                    color="rgba(255,180,255,0.12)", line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(symbol="circle", size=54,
                    color="rgba(255,220,255,0.28)", line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    for color, size in white_layers:
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(
                symbol="star", size=size, color=color,
                line=dict(width=0), opacity=0.96,
            ),
            hoverinfo="skip", showlegend=False,
        ))
    # Core: hover + click handler at the origin, drawn on top of the layers.
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(symbol="circle", size=6, color="#FFFFFF",
                    line=dict(color="rgba(0,0,0,0.6)", width=0.8)),
        hovertext=[target_hover], hoverinfo="text",
        customdata=[target_profile.canonical_name],
        name=f"⭐ {target_profile.canonical_name}",
        showlegend=True,
    ))
    # Name rendered as a separate text trace placed below the outer
    # star layer (size 50) so the label isn't swallowed by the star body.
    fig.add_trace(go.Scatter(
        x=[0], y=[-0.6], mode="text",
        text=[f"<b>{target_profile.canonical_name}</b>"],
        textposition="bottom center",
        textfont=dict(color="#FFFFFF", size=14, family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ))

    # ── Per-bucket stars ─────────────────────────────────────────────────
    # Draw order: per-bucket halos → per-bucket glows → Israeli white rings
    # → per-bucket core stars. The white ring sits at the star's radius so
    # its rim peeks out between the star's tips.
    israel_ids = {c.candidate_id for c in israel_list}
    info_data: dict[str, dict] = {}
    STAR_BORDER = "rgba(25,30,50,0.9)"
    STAR_BORDER_WIDTH = 1.0
    il_dot_xs: list[float] = []
    il_dot_ys: list[float] = []

    bucket_data: list[tuple[str, str, list[float], list[float],
                            list[str], list[str], list[str]]] = []
    for bucket_key, bucket_label in _BUCKET_LABELS.items():
        xs, ys, labels_txt, hovers, customs = [], [], [], [], []
        for i, cand in enumerate(all_cands):
            if cand.assigned_bucket != bucket_key:
                continue
            is_il = cand.candidate_id in israel_ids
            xs.append(coords[i, 0])
            ys.append(coords[i, 1])
            labels_txt.append(cand.name)
            customs.append(cand.name)
            side_label = "🇮🇱 Israel" if is_il else "🌍 World"
            inst = cand.institution or "—"
            reasons_text = _format_reason_text(cand.reasons or [], reason_topic_names)
            hover = _candidate_hover_text(cand.name, inst, bucket_label, side_label)
            hovers.append(hover)
            info_data[cand.name] = {
                "name": cand.name,
                "institution": inst if inst != "—" else "",
                "bucket": bucket_label,
                "side": side_label,
                "color": _BUCKET_COLORS[bucket_key],
                "stats": (
                    f"same-area <b>{cand.same_area_score:.2f}</b> · "
                    f"seniority <b>{cand.seniority_score:.2f}</b> · "
                    f"reputation <b>{cand.reputation_score:.2f}</b>"
                ),
                "reasons": reasons_text,
            }
            if is_il:
                il_dot_xs.append(coords[i, 0])
                il_dot_ys.append(coords[i, 1])
        if xs:
            bucket_data.append(
                (bucket_key, bucket_label, xs, ys,
                 labels_txt, hovers, customs)
            )

    for bucket_key, _lbl, xs, ys, *_ in bucket_data:
        color = _BUCKET_COLORS[bucket_key]
        fig.add_trace(go.Scatter(    # outer halo
            x=xs, y=ys, mode="markers",
            marker=dict(size=42, color=_hex_to_rgba(color, 0.10),
                        line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))

    for bucket_key, _lbl, xs, ys, *_ in bucket_data:
        color = _BUCKET_COLORS[bucket_key]
        fig.add_trace(go.Scatter(    # mid glow
            x=xs, y=ys, mode="markers",
            marker=dict(size=26, color=_hex_to_rgba(color, 0.28),
                        line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))

    # ── White ring marking Israeli researchers ─────────────────────────────
    # Matches the core-star radius (size 17) so the circle's rim peeks out
    # between the star's pointed tips, reading as a clean white halo.
    if il_dot_xs:
        fig.add_trace(go.Scatter(
            x=il_dot_xs, y=il_dot_ys, mode="markers",
            marker=dict(symbol="circle", size=17, color="#FFFFFF"),
            hoverinfo="skip", showlegend=False, name="il-ring",
        ))

    for bucket_key, bucket_label, xs, ys, labels_txt, hovers, customs in bucket_data:
        color = _BUCKET_COLORS[bucket_key]
        fig.add_trace(go.Scatter(    # core star (interactive)
            x=xs, y=ys, mode="markers+text",
            marker=dict(
                symbol="star", size=17, color=color,
                line=dict(color=STAR_BORDER, width=STAR_BORDER_WIDTH),
                opacity=0.98,
            ),
            text=labels_txt, textposition="top center",
            textfont=dict(color="rgba(235,245,255,0.92)", size=9.5),
            hovertext=hovers, hoverinfo="text",
            customdata=customs,
            name=bucket_label,
        ))

    # ── Legend-only "Israeli researcher" explainer ──────────────────────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=12, color="#FFFFFF"),
        name="🇮🇱 Israeli (white ring)",
        hoverinfo="skip", showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=(f"<span style='color:#F5F5FA;'>Researcher Star Map — </span>"
                  f"<span style='color:#FFD700;'>"
                  f"{target_profile.canonical_name}</span>"),
            x=0.5, xanchor="center",
            font=dict(size=22, family="Arial"),
        ),
        paper_bgcolor="#080A14",
        plot_bgcolor="#080A14",
        xaxis=dict(visible=False, range=[-x_extent, x_extent]),
        yaxis=dict(
            visible=False,
            range=[-y_extent, y_extent],
            scaleanchor="x", scaleratio=1,
        ),
        showlegend=True,
        legend=dict(
            x=0.5, xanchor="center", y=-0.02, yanchor="top",
            orientation="h",
            font=dict(color="#F5F5FA", size=12),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        hoverlabel=dict(
            bgcolor="#1C2030",
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="#F5F5FA", size=12),
            align="left",
        ),
        margin=dict(l=20, r=20, t=70, b=90),
        # No fixed pixel size — the plot fills the viewport thanks to
        # autosize + CSS (see _CUSTOM_CSS); it adapts when the window resizes.
        autosize=True,
        # Default drag = pan (not zoom).  Scroll-wheel zooms in/out in place.
        dragmode="pan",
    )

    config = {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "displaylogo": False,
        # Natural cursor interactions: drag to pan, wheel to zoom.
        "scrollZoom": True,
        # Automatically re-lay out when the window/container resizes.
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"star_map_{target_profile.canonical_name.replace(' ', '_')}",
            "width": 1500, "height": 950,
        },
    }

    researcher_json = json.dumps(info_data)

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        config=config,
        post_script=_post_script(
            researcher_json,
            zoom_limit_x, zoom_limit_y,
            max_zoom_out_x, max_zoom_out_y,
        ),
        default_width="100vw",
        default_height="100vh",
    )

    # Inject CSS by post-processing the HTML (plotly.write_html has no CSS hook).
    html_text = output_path.read_text(encoding="utf-8")
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", _CUSTOM_CSS + "</head>", 1)
    else:
        html_text = _CUSTOM_CSS + html_text
    output_path.write_text(html_text, encoding="utf-8")

    log.info("Star map written to %s", output_path)
