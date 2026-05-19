from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InstitutionRef(BaseModel):
    openalex_id: Optional[str] = None
    ror: Optional[str] = None
    name: str
    country_code: Optional[str] = None
    department: Optional[str] = None
    raw_department: Optional[str] = None
    institution_type: Optional[str] = None  # OpenAlex type: "education", "company", "nonprofit", etc.


class Publication(BaseModel):
    id: str
    doi: Optional[str] = None
    title: str
    year: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    author_ids: list[str] = Field(default_factory=list)
    institution_ids: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    referenced_work_ids: list[str] = Field(default_factory=list)
    cited_by_count: int = 0


class ResearcherProfile(BaseModel):
    canonical_name: str
    openalex_author_id: Optional[str] = None
    semantic_scholar_author_id: Optional[str] = None
    orcid: Optional[str] = None

    institutions: list[InstitutionRef] = Field(default_factory=list)
    current_institution: Optional[InstitutionRef] = None
    current_department: Optional[str] = None

    publications: list[Publication] = Field(default_factory=list)

    works_count: int = 0
    cited_by_count: int = 0
    h_index: Optional[int] = None

    first_pub_year: Optional[int] = None
    last_pub_year: Optional[int] = None
    seniority_years: Optional[float] = None
    estimated_rank: Optional[str] = None  # "assistant" | "associate" | "full" | "unknown"

    # Author-level topic/concept summaries from OpenAlex
    # top_topics: T-prefixed IDs (newer field) — count-normalised, same repr as CandidateSummary
    # top_concepts: C-prefixed IDs (older x_concepts field) — score-normalised fallback
    top_topics: dict[str, float] = Field(default_factory=dict)
    top_concepts: dict[str, float] = Field(default_factory=dict)

    israel_affiliated: bool = False

    # CV-derived enrichment (populated when a CV file is provided)
    cv_research_interests: list[str] = Field(default_factory=list)
    known_cv_advisors: list[str] = Field(default_factory=list)


class CVData(BaseModel):
    """Structured data extracted from a researcher's CV (PDF or plain-text)."""
    # Identity
    name: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None

    # Affiliation
    institution: Optional[str] = None
    department: Optional[str] = None
    position_rank: Optional[str] = None   # "assistant" | "associate" | "full" | "postdoc"

    # Career timeline
    career_start_year: Optional[int] = None   # earliest year from PhD / first position
    phd_year: Optional[int] = None

    # Research
    research_interests: list[str] = Field(default_factory=list)  # free-text topics

    # Publications extracted directly from CV
    publication_titles: list[str] = Field(default_factory=list)
    publication_dois: list[str] = Field(default_factory=list)

    # Advisor / postdoc-host names extracted from Education and Positions sections
    known_advisors: list[str] = Field(default_factory=list)

    # Raw text (used for downstream department extraction)
    raw_text: str = ""


class FeatureVector(BaseModel):
    researcher_id: str
    topic_vector: dict[str, float] = Field(default_factory=dict)
    concept_vector: dict[str, float] = Field(default_factory=dict)
    coauthor_vector: dict[str, float] = Field(default_factory=dict)
    citation_vector: dict[str, float] = Field(default_factory=dict)
    venue_vector: dict[str, float] = Field(default_factory=dict)
    career_features: dict[str, float] = Field(default_factory=dict)


class CandidateSummary(BaseModel):
    """Lightweight profile built from an OpenAlex author summary record."""
    openalex_id: str
    name: str
    orcid: Optional[str] = None
    works_count: int = 0
    cited_by_count: int = 0
    current_institution: Optional[InstitutionRef] = None
    israel_affiliated: bool = False
    # Topics (T-prefixed IDs, newer OpenAlex field) — matches publication topic_ids
    top_topics: dict[str, float] = Field(default_factory=dict)
    # Concepts (C-prefixed IDs, older OpenAlex field) — fallback
    top_concepts: dict[str, float] = Field(default_factory=dict)
    first_pub_year: Optional[int] = None
    last_pub_year: Optional[int] = None
    # Populated after full work ingestion
    feature_vector: Optional[FeatureVector] = None


class BucketScores(BaseModel):
    in_area_collaborators: float = 0.0
    interdisciplinary_collaborators: float = 0.0
    recommendation_letter_writers: float = 0.0
    dept_mentors: float = 0.0
    area_mentors: float = 0.0


class CandidateScore(BaseModel):
    candidate_id: str
    name: str
    institution: Optional[str] = None
    country_code: Optional[str] = None
    department: Optional[str] = None
    israel_affiliated: bool = False

    last_pub_year: Optional[int] = None   # used to gate out inactive/deceased researchers
    works_count: int = 0                  # used to gate out students / preprint-only researchers

    base_similarity: float = 0.0
    same_area_score: float = 0.0
    complementary_topic_score: float = 0.0
    seniority_score: float = 0.0
    reputation_score: float = 0.0
    familiarity_proxy: float = 0.0
    no_conflict_score: float = 0.0
    same_department: bool = False
    same_institution: bool = False
    candidate_cites_target: bool = False  # True when candidate authored a paper citing the target.
    recent_coauthor: bool = False
    any_coauthor: bool = False          # True if coauthor on ANY publication (all years)

    # True when both target and candidate departments are known and they differ.
    # Used to hard-exclude cross-department candidates from dept_mentors even
    # when dept_mentor_requires_same_department is False in policy.
    dept_explicitly_different: bool = False

    # False when OpenAlex reports the candidate's institution as a company.
    # Used to exclude industry researchers from mentor / letter-writer buckets.
    is_academic_institution: bool = True

    # True when the institution is a non-research organisation (disease-advocacy
    # nonprofit, honour society, etc.) detected by the Tier-1 name-pattern check.
    # These candidates are excluded from ALL buckets, not just mentor/letter roles.
    # Unlike is_academic_institution, this is NOT set for plain industry companies —
    # a researcher at Meta AI Research can still be a valid collaborator.
    is_non_research_institution: bool = False

    # CV-derived relationship label — e.g. "advisor" or "postdoc_host"
    cv_relationship: Optional[str] = None

    bucket_scores: BucketScores = Field(default_factory=BucketScores)
    assigned_bucket: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
