# Researcher Mapper — Specifications & Design Decisions

This document records all agreed-upon specifications, design decisions, and
calibration examples established across conversations about this project.
**When modifying the pipeline, verify every relevant specification below is
still satisfied before considering a change complete.**

---

## 1. Output format

- Two independent ranked lists per run: **Israel** (IL-affiliated) and **World**.
- Each list contains up to **20 researchers**, assigned across five buckets.
  The internal key `dept_mentors` is displayed to users as **Institutional Mentors**
  because OpenAlex rarely exposes reliable department-level affiliation data.
- Produced as CSV files: `output/<run_id>/israel_top20.csv` and `world_top20.csv`.

---

## 2. Buckets

### 2.1 Priority order (fills in this sequence; each researcher appears once)

```
dept_mentors → area_mentors → recommendation_letter_writers
  → in_area_collaborators → interdisciplinary_collaborators
```

There is no separate overflow pass. Extra capacity is absorbed by the larger
`in_area_collaborators` cap, and every selected researcher must pass the hard
filter for their assigned bucket.

### 2.2 Caps (`config/bucket_caps.yaml`)

| Bucket                         | Cap |
|-------------------------------|-----|
| `dept_mentors` (Institutional Mentors) | 2 |
| `area_mentors`                | 3   |
| `recommendation_letter_writers` | 3 |
| `in_area_collaborators`       | 20  |
| `interdisciplinary_collaborators` | 5 |

### 2.3 Hard eligibility filters per bucket (from `assignment.py`)

| Bucket | Constraints |
|--------|-------------|
| `dept_mentors` | same institution; seniority ≥ 0.65; same_area ≥ **0.30**; no explicit dept mismatch; displayed as Institutional Mentors |
| `area_mentors` | same_area ≥ **0.30**; seniority ≥ 0.65; NOT same dept; NOT industry; reputation ≥ target's reputation |
| `recommendation_letter_writers` | same_area ≥ **0.30**; seniority ≥ 0.65; reputation ≥ **0.75**; NOT recent coauthor; NOT same dept; NOT industry |
| `in_area_collaborators` | same_area ≥ 0.18; seniority ≥ 0.45 unless same_area ≥ 0.55; if same_area < 0.45, requires `candidate_cites_target`, familiarity_proxy ≥ 0.30, or the country-list senior-local exception |
| `interdisciplinary_collaborators` | 0.07 ≤ same_area < 0.18; complementary_topic ≥ 0.15; seniority ≥ 0.45 |

**Global gates (every bucket):**
- Last publication within 10 years (`max_inactivity_years: 10`)
- Seniority ≥ **0.40** (`min_seniority_for_all`) — raised from 0.20 to filter
  PhD students whose seniority falls below faculty range
- Works count ≥ **20** (`min_works_for_all`) — raised from 10; PhD students
  typically have 5–15 indexed works, faculty have 20+
- Collaborator seniority ≥ **0.45** (`min_seniority_for_collaborators`) — a
  conservative "already has PhD" proxy, since OpenAlex has no PhD-status field.
  Clear same-area junior faculty can pass when same_area ≥ **0.55**.
- Borderline country-list collaborators can pass without an explicit citation
  bridge only when same_area ≥ **0.35**, seniority ≥ **0.50**, and reputation ≥
  **0.55**. This preserves strong local same-area researchers whose OpenAlex
  topic vector is diluted, while still filtering topic-distant local matches.
- When a requested country list is still under-filled, a supplemental local
  in-area pass can add Israeli-affiliated academics with same_area ≥ **0.40**
  and seniority ≥ **0.48**; broader local matches require **0.25 ≤ same_area
  < 0.35**, seniority ≥ **0.50**, and base_similarity ≥ **0.18**. The broad
  local fallback does not impose a reputation floor because its role is to fill
  collaborator slots, not mentor/letter-writer slots.
- Co-authors of the target are excluded from ALL buckets except `dept_mentors`

**`area_mentors` relative seniority rule:**
A mentor must be more established than the target. Enforced via
`reputation_score >= target_reputation_score`, where `target_reputation_score`
is computed per run as `estimate_reputation(target.cited_by_count, target.works_count)`
and injected into the policy dict in `run_target.py` before calling
`assign_israel_and_world`. Do NOT use `seniority_score` for this comparison —
OpenAlex `counts_by_year` misattributions cause career-start years to be wrong
for many candidates, making seniority scores unreliable for relative comparison.

---

## 3. Scoring formulas

### 3.1 `base_similarity` (weights from `config/weights.yaml`)

```
base_similarity = 0.35 * topic_sim
                + 0.25 * citation_sim
                + 0.20 * collab_sim
                + 0.10 * venue_sim
                + 0.10 * career_sim
```

### 3.2 `same_area_score` (`similarity.py`)

```
same_area_score = 0.65 * topic_sim + 0.35 * citation_sim
```

**NOTE:** Candidates typically have an empty `citation_vector`, so
`citation_sim ≈ 0` for most candidates. The topic component dominates.
Do NOT change this formula to boost a specific researcher — it scales
all candidates equally and does not change relative ordering.

### 3.3 `_best_topic_sim` — namespace matching

OpenAlex uses two distinct topic namespaces:
- `T-prefixed`: topics (newer, from publication `topic_ids` and `author.topics`)
- `C-prefixed`: concepts (older, from `author.x_concepts`)

Comparing T-IDs against C-IDs always gives Jaccard = 0 (disjoint sets).
`_best_topic_sim` returns `max(T-vs-T jaccard, C-vs-C jaccard)`.

### 3.4 Bucket-specific scores (`config/weights.yaml`)

```yaml
in_area_collaborators:
  same_area: 0.45 | collaboration: 0.20 | venue: 0.15
  institution_distance_bonus: 0.10 | recency_overlap: 0.10

area_mentors:
  same_area: 0.35 | seniority: 0.25 | reputation: 0.15
  distance_diversity_bonus: 0.15 | mentoring_signal: 0.10

recommendation_letter_writers:
  same_area: 0.30 | seniority: 0.25 | reputation: 0.20
  no_conflict: 0.15 | familiarity_proxy: 0.10

dept_mentors:
  department_match: 0.35 | same_area: 0.25 | seniority: 0.20
  institutional_role: 0.10 | mentoring_signal: 0.10

interdisciplinary_collaborators:
  complementary_topic: 0.35 | method_complementarity: 0.20
  citation_bridge: 0.15 | collaboration_potential: 0.15
  institutional_proximity: 0.15
```

---

## 4. Co-author exclusion policy

```yaml
exclude_all_coauthors: true           # global — all historical co-authors excluded
dept_mentors_allows_coauthors: true   # exception: same-institution co-authorship is
                                      # a positive working-relationship signal
```

**Co-authors must NOT appear in any output bucket except `dept_mentors`.**
Adding `_allows_coauthors: true` to `in_area_collaborators` or
`interdisciplinary_collaborators` violates this specification.

The `any_coauthor` flag is true if they co-authored ANY paper with the target
in OpenAlex (all time, not just recent years).

---

## 5. Israel affiliation detection (`candidate_ingest.py:_is_israel_affiliated`)

Four checks in order of reliability. Returns `True` on first match:

1. `last_known_institution.country_code == "IL"` (current, singular) — **corroborated**
2. Any entry in `last_known_institutions[].country_code == "IL"` (current, plural) — **corroborated**
3. Any `affiliations[]` entry where `country_code == "IL"`, `type != "company"`,
   `len(years) >= 2`, and `max(years) >= 2023`
4. Any `affiliations[]` entry where `country_code == "IL"`, `type != "company"`,
   `len(years) >= 2`, and `max(years) >= 2021` — catches OpenAlex lag (2–3 yr)

**Corroboration requirement for Checks 1 and 2:** OpenAlex's `last_known_institution` /
`last_known_institutions` can be stale or wrong (e.g. a researcher whose record was
updated based on a single visiting-year paper). If the `affiliations[]` list is
non-empty, Checks 1 and 2 only fire when at least one non-company IL entry in
`affiliations[]` has `len(years) >= 2`. If `affiliations[]` is completely empty
(new researcher with no publication history), Checks 1/2 trust the current-institution
field without corroboration.
- **Counter-example**: Ashia Wilson (MIT) had `last_known_institutions = [Hebrew University]`
  but only 1 year of Hebrew University affiliation in her publication record → Checks 1/2
  are skipped; Checks 3/4 also fail (< 2 years) → correctly classified as non-Israeli.

**Company affiliations (Meta Israel, Apple Israel, etc.) are excluded from all checks.**
Working at a tech company's Israel office does not constitute academic IL affiliation.
This prevents Léon Bottou, Michael Rabbat, Vitaly Feldman, Xiaohan Wei from being
incorrectly classified as Israeli.

**Israel vs. World list assignment** (`assignment.py:_currently_in_israel`):
- `israel_affiliated = True` AND `is_academic_institution = True` → Israel list,
  even when the current institution country is non-IL
- `country_code == "IL"` AND `israel_affiliated = True` AND `is_academic_institution = True` → Israel list
- `country_code == "IL"` AND `israel_affiliated = False` → World list (wrong/stale OpenAlex institution)
- `country_code == "IL"` AND `is_academic_institution = False` (company) → World list
- `country_code` is a known non-IL code and `israel_affiliated = False` → World list
- `country_code` is empty/unknown → falls back to `israel_affiliated`

---

## 6. Candidate pool generation (`candidate_generation.py`)

### 6.1 Pool sizes

```
_COUNTRY_SLOTS = 150   # dedicated slots for country-specific candidates
global_cap = max(max_pool_size - _COUNTRY_SLOTS, max_pool_size // 2)
max_pool_size = 200    # default argument
```

### 6.2 Global pool strategy order

| # | Strategy | Notes |
|---|----------|-------|
| 0 | Coauthors (FIRST) | Up to 50; placed first so never crowded out by pool cap |
| 1 | Same institution | Seeds `dept_mentor` bucket |
| 2 | Citing authors | Strongest specificity signal; up to 50 top-cited works |
| 3 | Reference neighbours | Optional (`--include-citation-neighbors`); up to 20 target works x 50 references; prioritized for senior letter writers |
| 4 | T-topic global | Top 8 topic IDs x `per_concept` results |
| 5 | C-concept global | Top 3 concept IDs x 10 results (fallback for old records) |

### 6.3 Country pool strategy order (IL example)

| # | Strategy | Notes |
|---|----------|-------|
| A | Topic + IL filter (FIRST) | `per_page = max(per_concept * 4, 200)` — fetches top-200 per topic to capture mid-ranked local researchers |
| B | Country-wide top-cited | High-citation IL researchers; broad field coverage |
| C | Recent IL papers (LAST) | Unfiltered; placed last to avoid saturating country slots with off-topic researchers |

**Ordering rationale:** Strategy C alone generates ~500 authors and would
saturate the 150-slot country cap before A and B contribute. Strategy A
must come first to guarantee topically-relevant local researchers are included.
Nadav Cohen (rank ~85 for T11612+IL) is a concrete example; he is missed at
per_page=50 and must be fetched with per_page=200.

---

## 7. Target feature vector (`run_target.py:_build_target_feature_vector`)

The target's topic vector is built from their publication record (not from
`profile.top_topics`), then truncated to the top-25 topics by weight.

```python
raw = build_topic_vector(pubs)
if len(raw) > 25:
    top_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:25]
    total = sum(v for _, v in top_items)
    topic_vec = {k: v / total for k, v in top_items} if total else {}
else:
    topic_vec = raw
```

---

## 8. Industry filter

```yaml
exclude_industry_from_mentors: true
exclude_industry_from_letter_writers: true
```

Detection: `is_academic_institution` is False when
`current_institution.institution_type == "company"`.

Two-layer override in `candidate_ingest.py`:
1. **`last_known_institution` present**: if a researcher has a company affiliation
   within the last 3 years in `affiliations[]`, override to `"company"`.
2. **`last_known_institution` absent (None)**: if any company affiliation exists
   within the last 5 years AND is more recent than the latest `education`-type
   affiliation, create a placeholder institution with `institution_type="company"`.
   Guard: a researcher with a 2021 visiting-Google entry but 2025 TAU entry is
   treated as academic (university year ≥ company year — not flagged).

Industry detection uses both `institution_type == "company"` AND a name keyword
list (`_INDUSTRY_NAME_KEYWORDS`) because OpenAlex sometimes leaves `institution_type`
unset for known tech companies (e.g. Meta AI Research).

**Tier-1 non-research institution filter** (`run_target.py:_is_non_research_institution`):
A second filter catches disease-advocacy nonprofits and honour societies that OpenAlex
sometimes lists as researcher affiliations (e.g. "Alpha Omega Alpha Medical Honor Society",
"Alzheimer's Association of Israel"). These set `is_non_research_institution = True` on
`CandidateScore` and are excluded from **all** buckets including collaborator buckets,
unlike the industry gate which only blocks mentor/letter-writer roles.

The check is **name-only** (no institution_type filter) because the company-override in
`candidate_ingest.py` can mutate `institution_type → "company"` while leaving the display
name unchanged. The `_NON_RESEARCH_INST_RE` patterns are narrow enough to avoid false
positives against research hospitals, medical schools, or legitimate research foundations
(Simons Foundation, NIH, etc.).

**CV-advisor exception**: candidates with `cv_relationship ∈ {"advisor", "postdoc_host"}`
bypass the co-author exclusion gate AND the `same_area` minimum for
`recommendation_letter_writers`. This ensures PhD advisors/postdoc hosts extracted
from a provided CV always appear as letter writers regardless of topic-vector overlap.

**OpenAlex record contamination guard** (`candidate_ingest.py:_parse_candidate_summary`):
OpenAlex sometimes merges a living researcher's record with historical publications from
an unrelated person with the same name. The misattributed publications corrupt the
`author.topics` aggregation (topic weights are computed over ALL attributed works), causing
the researcher's `top_topics` to include foreign topics with large weights and inflating
`same_area` scores against the target.

Two detection signals — both clear `top_topics` / `top_concepts`:
1. **Career span > 55 years** (`first_pub_year_raw < current_year − 55`):
   impossible for any living researcher. Example: Dan Feldman (first_pub=1958).
2. **Major leading gap** (≥ 10 years between the earliest `counts_by_year`
   entry and the next): a sole publication 10+ years before the main career
   is almost always a merge artifact. Example: Gil Shamai had a single 2003
   paper 12 years before his main 2015–2025 career, inflating his career_age
   in the seniority score.

After contamination detection (but independent of it), the same gap-cleanup
algorithm also trims leading outliers ≥ 5 years apart from the next
publication year, then recomputes `first_pub_year` from the cleaned series.
This prevents a single outlier paper from inflating `career_age_norm` in
the seniority score even when the gap is only 5–9 years (below the
topic-contamination threshold).

**Industry-override requires ≥ 2 years of industry presence**
(`candidate_ingest.py`): OpenAlex sometimes merges a single-year company
entry from a different person into a researcher's record. The industry
override that replaces `current_institution` with a company affiliation
only fires when the industry affiliation has `len(years) >= 2`. This mirrors
SPECS §5 Checks 3/4. Example: Nadav Cohen's record is contaminated with a
single-year "South Australian Water Corporation" (2023) entry; without this
rule he is classified as industry and routed to the world list instead of
in_area_collaborators on the Israel list.

**Fallback to most-recent academic affiliation**: when `last_known_institution`
is absent AND no industry override fires, `current_institution` is set to
the most recent academic affiliation (≤ 5 years). Gives researchers like
Nadav Cohen a proper country_code / institution for routing and display.

**Citing-author strategy ordering** (`candidate_generation.py`, `openalex.py`):
Citing papers are fetched sorted by `cited_by_count:desc` (not by recency), so the most
influential citing papers (authored by prominent researchers) are captured first. Without
this sort, the default ordering by date biases toward recent grad-student citations and
misses established researchers who cited the target years ago. `per_page=50` per target
paper gives sufficient coverage.

**Same-area floor for citing authors** (`similarity.py:compute_same_area_score`):
If `candidate_cites_target=True`, `same_area_score` is floored at `0.25`. This
ensures researchers with an explicit citation bridge clear the basic
`in_area_collaborators` threshold of 0.18, even when their author-level topic
vector is diluted across many sub-topics. Borderline same-area scores below
0.45 still require this citation bridge, familiarity_proxy ≥ 0.30, or the
country-list senior-local exception, which prevents broad OpenAlex topics such
as AI/ML from promoting adjacent but topic-distant researchers into the in-area
bucket.

---

## 9. Calibration examples (Daniel Soudry target)

The following concrete researchers serve as calibration tests. When running
against Daniel Soudry (Technion EE/ML, OpenAlex A5025123281), the Israel
list should satisfy:

### Must appear
| Researcher | Expected bucket | Reason |
|------------|----------------|--------|
| Ron Meir | `dept_mentors` / Institutional Mentors | Technion EE, same institution, same_area ≈ 0.40 |
| Ran El-Yaniv | `dept_mentors` / Institutional Mentors | Technion CS, same institution, same_area ≈ 0.38; not displayed as a department mentor |
| Shai Shalev-Shwartz | `area_mentors` | Hebrew University ML; reputation 0.88 > Soudry's 0.81; seniority 0.69; same_area ≈ 0.42 |

### Must NOT appear
| Researcher | Reason |
|------------|--------|
| Gal Vardi | Co-author of Soudry |
| Elad Hoffer | Co-author of Soudry |
| Itay Hubara | Co-author of Soudry |
| Brian Chmiel | Co-author of Soudry |
| Edward Moroshko | Co-author of Soudry |
| Alon Brutzkus | Co-author of Soudry |
| Léon Bottou | Not Israeli; Meta Israel is a company office, not an academic affiliation |
| Xiaohan Wei | Not Israeli; Meta Israel company affiliation only |
| Preetum Nakkiran | Not Israeli; Apple Israel company office |
| Yair Goldberg (Technion, statistician) | same_area ≈ 0.14 < 0.30 threshold for Institutional Mentors |
| Evgenii Zheltonozhskii | PhD student / below collaborator career-stage proxy (`seniority_score` ≈ 0.42 < 0.45) |
| Vadim Indelman | Robotics/SLAM is adjacent but too far for in-area ML theory; same_area ≈ 0.37 is below the 0.45 unbridged in-area floor and lacks an explicit citation/familiarity bridge |
| Huanyu Zhang | Topic match is too broad/remote for an in-area ML-theory collaborator; same_area ≈ 0.44 is below the 0.45 unbridged in-area floor and lacks an explicit citation/familiarity bridge |
| Nadav Cohen (as `area_mentors`) | Less senior than target; reputation 0.56 < Soudry's 0.81 |
| Dan Feldman | Research area (computational geometry / coresets) is too far from ML theory; OpenAlex record is contaminated (first_pub_year=1958, career span > 55 years) → topic vector cleared by contamination guard → same_area ≈ 0; excluded from all buckets |
| Gil Shamai (Technion, PhD student) | OpenAlex record contaminated by a single 2003 paper merged in from a different person (main career starts 2015). The ≥10-year leading gap fires the contamination guard → top_topics cleared → same_area ≈ 0; excluded |
| Avigdor Gal (Technion) | same_area ≈ 0.27 < 0.30 threshold; data integration / knowledge representation is too far from ML theory for any senior role (dept_mentor, letter_writer) |
| Ashia Wilson (MIT) — **Israel list only** | Not Israeli; OpenAlex incorrectly lists her at Hebrew University based on a single visiting-year paper (2026 only, < 2 years) — corroboration check correctly excludes her from the Israel list. She may legitimately appear in the World list as an MIT ML theorist. |
| Andrew Tulloch | Industry (Meta AI Research); `last_known_institution = None` but Meta affiliations within 5 years with no more-recent academic affiliation → company override fires |

### Nadav Cohen's correct placement
Nadav Cohen (Hebrew University ML theorist) IS a relevant researcher and
SHOULD appear in `in_area_collaborators` — NOT in `area_mentors`.
He is NOT a co-author of Soudry.

His `israel_affiliated = True` via Check 4 of `_is_israel_affiliated`:
TAU affiliation years `[..., 2020, 2021]` (academic, len ≥ 2, max ≥ 2021).

**Why he might be missing**: His author-level `topics` in OpenAlex span many
sub-topics (deep learning, tensor decompositions, quantum, etc.), diluting the
Jaccard with Soudry's concentrated publication topics. When he is captured as a
citing author, `compute_same_area_score` applies a floor of 0.25 and provides
the explicit bridge required for borderline in-area candidates. When OpenAlex
misses that citation-neighbour provenance, the country-list senior-local
exception keeps him eligible: same_area ≈ 0.36, seniority ≈ 0.64, reputation ≈
0.56.

### Known OpenAlex data issues
- **Nadav Cohen (A5104108669)**: `last_known_institution: None`; IL affiliation via TAU/Technion/HU history through 2022. His record is also contaminated with a single-year "South Australian Water Corporation" (2023) entry from a different person — this triggered the industry-override and mis-classified him as AU-industry before the ≥2-year rule was added. After the fix, the single-year AU-company entry is ignored and the most-recent academic affiliation (Hebrew University of Jerusalem, 2022) is used as `current_institution`.
- **Léon Bottou, Michael Rabbat, Vitaly Feldman, Xiaohan Wei**: all have Meta Israel or Apple Israel affiliations that used to trigger Check 4, but the company filter now correctly excludes them.
- **Dan Feldman**: `first_pub_year=1958` (career span 68 years) — clear OpenAlex record merge with a historical person. The contamination guard in `candidate_ingest.py` detects career span > 55 years and clears `top_topics`/`top_concepts`, dropping his same_area to ≈ 0 and excluding him from all buckets.
- **Eider Moore**: Meta (Israel), correctly excluded from mentors/letter-writers by industry filter; also correctly excluded from Israel list (company-IL → World list).

---

## 10. Pipeline execution

```bash
# Standard run (from project root):
python -m researcher_mapper.pipelines.run_target --name "Daniel Soudry"

# Output is written to: output/daniel_soudry_<timestamp>/
# Key files:
#   israel_top20.csv
#   world_top20.csv
#   results.json
```

**Verification checklist after every run:**
1. Scores differ from the previous run (confirms the fix/change took effect).
2. All "must appear" researchers from §9 are present in their expected buckets.
3. None of the "must NOT appear" researchers appear in any bucket.
4. No company-Israel-office researchers (Meta Israel, Apple Israel) are in the Israel list.
5. No co-authors of the target are in any bucket other than `dept_mentors`.
6. Bucket counts do not exceed caps defined in `bucket_caps.yaml`.
7. No researcher appears in more than one bucket.
