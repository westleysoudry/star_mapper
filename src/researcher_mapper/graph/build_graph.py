"""
Build a NetworkX multi-layer graph connecting the target researcher to
their candidate pool.  The graph is used for visualisation and can be
exported to JSON (node-link format) or GraphML.

Node attributes: name, institution, country_code, israel_affiliated,
                 seniority_score, base_similarity, assigned_bucket.
Edge attributes:  relation (coauthor | topic | citation | assignment),
                  weight (similarity or collaboration frequency).
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from researcher_mapper.models.schemas import CandidateScore, ResearcherProfile


def build_researcher_graph(
    target: ResearcherProfile,
    all_scores: list[CandidateScore],
    israel_list: list[CandidateScore],
    world_list: list[CandidateScore],
) -> nx.Graph:
    """
    Construct a graph where:
    - Nodes are the target + all scored candidates.
    - Edges connect candidates to the target, weighted by base_similarity.
    - Assigned candidates carry extra edge/node metadata.
    """
    G = nx.Graph()

    # ── Target node ────────────────────────────────────────────────────────────
    target_id = (target.openalex_author_id or "target").split("/")[-1]
    G.add_node(
        target_id,
        label=target.canonical_name,
        node_type="target",
        institution=target.current_institution.name if target.current_institution else "",
        country_code=target.current_institution.country_code if target.current_institution else "",
        israel_affiliated=target.israel_affiliated,
    )

    assigned_israel = {c.candidate_id for c in israel_list}
    assigned_world = {c.candidate_id for c in world_list}

    for score in all_scores:
        cid = score.candidate_id.split("/")[-1]

        lists_in = []
        if score.candidate_id in assigned_israel:
            lists_in.append("israel")
        if score.candidate_id in assigned_world:
            lists_in.append("world")

        G.add_node(
            cid,
            label=score.name,
            node_type="candidate",
            institution=score.institution or "",
            country_code=score.country_code or "",
            israel_affiliated=score.israel_affiliated,
            assigned_bucket=score.assigned_bucket or "",
            base_similarity=round(score.base_similarity, 4),
            same_area_score=round(score.same_area_score, 4),
            seniority_score=round(score.seniority_score, 4),
            lists=",".join(lists_in),
        )

        G.add_edge(
            target_id,
            cid,
            weight=round(score.base_similarity, 4),
            relation="similarity",
            assigned_bucket=score.assigned_bucket or "",
        )

    return G


def graph_to_node_link(G: nx.Graph) -> dict:
    """Serialize to JSON-compatible node-link format."""
    return nx.node_link_data(G)


def save_graph_json(G: nx.Graph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = graph_to_node_link(G)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def save_graph_graphml(G: nx.Graph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(path))
