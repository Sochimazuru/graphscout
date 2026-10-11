"""Plumbing tests. The papers and LLM replies below are FAKE fixtures written by hand
to exercise the code paths. They are not results and say nothing about model quality."""
from __future__ import annotations

import json

import pytest

from graphscout.build import build_graph
from graphscout.config import Config
from graphscout.extract import THIS_PAPER, extract_all, validate
from graphscout.fetch import parse_feed
from graphscout.neo4j_export import collect, to_cypher
from graphscout.qa import answer_question, trace_to_markdown
from graphscout.queries import QUERIES, run
from graphscout.resolve import Resolver
from graphscout.schema import align_relation, align_type
from graphscout.store import GraphStore

PAPERS = [
    {"id": "0000.00001", "title": "Fake paper A", "abstract": "A fixture.", "published": "2025-01-01",
     "authors": ["Test Author"], "url": "https://arxiv.org/abs/0000.00001"},
    {"id": "0000.00002", "title": "Fake paper B", "abstract": "Another fixture.", "published": "2025-02-01",
     "authors": ["Test Author"], "url": "https://arxiv.org/abs/0000.00002"},
]

EXTRACTIONS = [
    {"paper_id": "0000.00001", "entities": [
        {"name": "MethodX", "type": "Method"},
        {"name": "Retrieval-Augmented Generation (RAG)", "type": "Method"},
        {"name": "Knowledge Graphs", "type": "Concept"},
        {"name": "HotpotQA", "type": "Dataset"}],
     "triples": [
        {"subject": THIS_PAPER, "relation": "proposes", "object": "MethodX"},
        {"subject": "MethodX", "relation": "leverages", "object": "Knowledge Graphs"},
        {"subject": "MethodX", "relation": "outperforms", "object": "RAG"},
        {"subject": THIS_PAPER, "relation": "evaluated on", "object": "HotpotQA"}]},
    {"paper_id": "0000.00002", "entities": [
        {"name": "MethodY", "type": "framework"},
        {"name": "knowledge graph", "type": "Concept"},
        {"name": "HotpotQA", "type": "benchmark"}],
     "triples": [
        {"subject": THIS_PAPER, "relation": "introduces", "object": "MethodY"},
        {"subject": "MethodY", "relation": "is built on", "object": "MethodX"},
        {"subject": "MethodY", "relation": "uses", "object": "knowledge graph"},
        {"subject": "MethodY", "relation": "is inspired by", "object": "HotpotQA"},
        {"subject": THIS_PAPER, "relation": "tested on", "object": "HotpotQA"}]},
]


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.data_dir, c.out_dir = tmp_path / "data", tmp_path / "out"
    c.ensure_dirs()
    return c


@pytest.fixture
def graph():
    g, stats = build_graph(PAPERS, EXTRACTIONS)
    return g, stats


def test_parse_arxiv_feed():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2401.12345v2</id><title>A  Title</title>
      <summary> Some
      abstract </summary><published>2024-01-20T00:00:00Z</published>
      <author><name>Ada</name></author></entry></feed>"""
    [p] = parse_feed(xml)
    assert p["id"] == "2401.12345" and p["title"] == "A Title" and p["abstract"] == "Some abstract"
    assert p["published"] == "2024-01-20" and p["authors"] == ["Ada"]


@pytest.mark.parametrize("raw,expected", [
    ("proposes", "proposes"), ("leverages", "uses"), ("evaluated on", "evaluatedOn"),
    ("evaluatedOn", "evaluatedOn"), ("outperforms", "outperforms"), ("is built on", "extends"),
    ("is inspired by", "relatedTo"), ("used for", "addresses"),
])
def test_align_relation(raw, expected):
    assert align_relation(raw) == expected


def test_align_type():
    assert align_type("benchmark") == "Dataset" and align_type("framework") == "Method"
    assert align_type("weird") == "Concept"


def test_validate_drops_bad_items_and_adds_missing_entities():
    out = validate({"entities": [{"name": "A", "type": "Method"}, {"x": 1}],
                    "triples": [{"subject": "A", "relation": "uses", "object": "B"},
                                {"subject": "A", "relation": "", "object": "C"},
                                {"subject": "A", "relation": "is", "object": "A"}]}, 10)
    assert {e["name"] for e in out["entities"]} == {"A", "B"}
    assert len(out["triples"]) == 1


def test_resolver_merges_acronyms_and_plurals():
    r = Resolver()
    for name in ["Retrieval-Augmented Generation (RAG)", "RAG", "retrieval augmented generation",
                 "Knowledge Graphs", "knowledge graph"]:
        r.add(name, "Method")
    r.build()
    assert r.lookup("RAG")["id"] == r.lookup("retrieval augmented generation")["id"]
    assert r.lookup("Knowledge Graphs")["id"] == r.lookup("knowledge graph")["id"]
    assert len(r.clusters) == 2


def test_resolver_does_not_merge_ambiguous_acronym():
    r = Resolver()
    for name in ["graph neural network", "generative news network", "GNN"]:
        r.add(name)
    r.build()
    assert len(r.clusters) == 3


def test_build_graph_stats(graph):
    _, stats = graph
    assert stats["papers"] == 2
    # MethodX, RAG, knowledge graph, HotpotQA, MethodY
    assert stats["entities_after_resolution"] == 5
    assert stats["relations_by_property"]["relatedTo"] == 1
    assert stats["triples_skipped"] == 0
    assert stats["entity_types"]["Dataset"] == 1


def test_canned_sparql_queries_run(graph):
    g, _ = graph
    for name in QUERIES:
        run(g, name)
    _, _, rows = run(g, "shared_datasets")
    assert rows == [("HotpotQA", "2")]
    _, _, rows = run(g, "outperforms")
    assert ("MethodX", "Retrieval-Augmented Generation") in rows


def test_store_find_and_neighbours_with_provenance(graph):
    g, _ = graph
    store = GraphStore(g)
    [uri] = store.find("Method X", k=1)
    assert store.label(uri) == "MethodX"
    edges = store.neighbours(uri)
    preds = {e.pred for e in edges}
    assert {"proposes", "uses", "outperforms", "extends"} <= preds
    ext = next(e for e in edges if e.pred == "extends")
    assert ext.sources == ("0000.00002",) and ext.original == ("is built on",)


def test_neo4j_cypher(graph):
    g, _ = graph
    nodes, edges = collect(g)
    cypher = to_cypher(nodes, edges)
    assert "MERGE (n:Node" in cypher and ":EXTENDS]" in cypher and ":MENTIONS]" in cypher


class ScriptedLLM:
    """Returns pre-written JSON by matching the system prompt. Fixture only."""

    def __init__(self):
        self.calls = []

    def json(self, system, user):
        self.calls.append(system)
        if "plan how" in system:
            return {"entities": ["MethodY"], "plan": "find what MethodY builds on and where it is tested"}
        if "explore" in system:
            lines = [l for l in user.splitlines() if l.startswith("[")]
            wanted = [i for i, l in enumerate(lines) if "--extends-->" in l or "--evaluatedOn-->" in l]
            return {"selected": wanted[:2], "done": len(self.calls) > 2, "reason": "fixture"}
        return {"answer": "MethodY extends MethodX [E1].", "cited": ["E1"], "confidence": "high"}


def test_multi_agent_qa_end_to_end(graph, cfg):
    g, _ = graph
    store = GraphStore(g)
    trace = answer_question(ScriptedLLM(), store, cfg, "What does MethodY build on?")
    assert trace.seeds == ["MethodY"]
    assert trace.hops and trace.evidence
    assert any("extends" in ev["fact"] for ev in trace.evidence)
    md = trace_to_markdown(trace, store)
    assert "MethodY extends MethodX" in md and "arxiv.org/abs/0000.00002" in md


def test_extract_all_uses_cache(cfg):
    class OneShotLLM:
        n = 0

        def json(self, system, user):
            OneShotLLM.n += 1
            return {"entities": [{"name": "A", "type": "Method"}], "triples": []}

    llm = OneShotLLM()
    extract_all(llm, PAPERS[:1], cfg)
    extract_all(llm, PAPERS[:1], cfg)
    assert OneShotLLM.n == 1
    saved = json.loads(next(cfg.extraction_dir.glob("*.json")).read_text())
    assert saved["paper_id"] == "0000.00001"
