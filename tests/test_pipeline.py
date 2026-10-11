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


def test_extraction_stops_when_model_missing(cfg):
    from graphscout.llm import ModelNotFound

    class MissingModel:
        calls = 0

        def json(self, system, user):
            MissingModel.calls += 1
            raise ModelNotFound("not installed")

    with pytest.raises(ModelNotFound):
        extract_all(MissingModel(), PAPERS, cfg)
    assert MissingModel.calls == 1


def test_model_check_is_strict(monkeypatch):
    from graphscout import llm as llm_mod

    class Resp:
        def json(self):
            return {"models": [{"name": "qwen2.5:7b"}]}

    monkeypatch.setattr(llm_mod.requests, "get", lambda *a, **k: Resp())
    c = Config()
    c.model = "qwen2.5:3b"
    with pytest.raises(llm_mod.LLMError, match="Installed: qwen2.5:7b"):
        llm_mod.OllamaLLM(c).check()
    c.model = "qwen2.5:7b"
    llm_mod.OllamaLLM(c).check()


def test_repair_truncated_json_keeps_complete_items():
    from graphscout.llm import _parse_json

    cut = ('{"entities": [{"name": "GraphRAG", "type": "Method"}, {"name": "HotpotQA", "type": "Dataset"}],\n'
           ' "triples": [{"subject": "THIS_PAPER", "relation": "proposes", "object": "GraphRAG"},\n'
           '   {"subject": "GraphRAG", "relation": "evaluated on", "object": "Hotp')
    out = _parse_json(cut)
    assert [e["name"] for e in out["entities"]] == ["GraphRAG", "HotpotQA"]
    assert out["triples"] == [{"subject": "THIS_PAPER", "relation": "proposes", "object": "GraphRAG"}]


def test_repair_handles_cut_inside_entities_list():
    from graphscout.llm import _parse_json

    out = _parse_json('{"entities": [{"name": "A", "type": "Method"}, {"name": "B", "ty')
    assert out == {"entities": [{"name": "A", "type": "Method"}]}
    assert validate(out, 12)["entities"] == [{"name": "A", "type": "Method"}]


def test_bad_json_is_not_retried(monkeypatch):
    from graphscout import llm as llm_mod

    calls = []

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "not json at all"}}

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: calls.append(1) or Resp())
    with pytest.raises(llm_mod.LLMError):
        llm_mod.OllamaLLM(Config()).json("s", "u")
    assert len(calls) == 1


def test_clean_name_drops_quantities_and_list_strings():
    from graphscout.extract import clean_name

    assert clean_name("['hallucinations']") == "hallucinations"
    assert clean_name(["GraphRAG"]) == "GraphRAG"
    for junk in ["100 patients", "76.4%", "1,500-query benchmark", "15 clinical categories"]:
        assert clean_name(junk) == ""
    assert clean_name("GPT-4") == "GPT-4" and clean_name("3D scene graph") == "3D scene graph"


def test_type_majority_vote():
    r = Resolver()
    for t in ["Concept", "Concept", "Dataset"]:
        r.add("knowledge graph", t)
    r.add("HotpotQA", "Dataset")
    r.add("HotpotQA", "Concept")
    r.build()
    assert r.lookup("knowledge graph")["type"] == "Concept"
    assert r.lookup("HotpotQA")["type"] == "Dataset"  # tie goes to the specific type


def test_duplicate_triples_counted_once():
    ex = [{"paper_id": "0000.00001", "entities": [{"name": "Alpha", "type": "Method"}, {"name": "Beta", "type": "Method"}],
           "triples": [{"subject": "Alpha", "relation": "extends", "object": "Beta"}] * 3}]
    _, stats = build_graph(PAPERS[:1], ex)
    assert stats["relations_by_property"] == {"extends": 1}
    assert stats["duplicate_triples_removed"] == 2


@pytest.mark.parametrize("raw,expected", [
    # wordings seen in a real run with qwen2.5:1.5b
    ("improves", "improves"), ("compared with", "comparedWith"), ("demonstrates", "reports"),
    ("enables", "addresses"), ("constructs", "proposes"), ("achieves", "improves"),
    ("deployed on", "uses"), ("on", "relatedTo"), ("preferred", "relatedTo"),
])
def test_align_relation_real_wordings(raw, expected):
    assert align_relation(raw) == expected


def test_latex_and_acronym_with_suffix():
    from graphscout.extract import clean_name
    from graphscout.resolve import split_acronym

    assert clean_name("over $10\text{%}$ in overall quality") == "over 10% in overall quality"
    r = Resolver()
    r.add("Document-level Retrieval Augmented Knowledge Graph Construction (RAKG) framework", "Method")
    r.add("Document-level Retrieval Augmented Knowledge Graph Construction", "Method")
    r.add("RAKG", "Method")
    r.build()
    assert len(r.clusters) == 1
    assert split_acronym("Thing (X) with extra words here")[1] is None


def test_type_word_seeds_and_grounding(graph, cfg):
    g, _ = graph
    store = GraphStore(g)

    class LLM:
        def json(self, system, user):
            if "plan how" in system:
                return {"entities": ["datasets"], "plan": "p"}
            if "explore" in system:
                return {"selected": [0], "done": True}
            return {"answer": "HotpotQA is used [E1]. Also [E9].", "cited": ["E1", "E7"]}

    trace = answer_question(LLM(), store, cfg, "Which datasets are used?")
    assert trace.seeds == ["HotpotQA"]
    assert trace.cited == ["E1"]  # E7 and E9 do not exist and are dropped
    # routing ("datasets" -> evaluatedOn) adds the HotpotQA facts before the explorer runs
    assert trace.confidence.startswith("cites 1 of 3")
    assert "0 of 2 sentences have no citation" in trace.confidence


def test_ollama_5xx_is_retried(monkeypatch):
    from graphscout import llm as llm_mod

    class Resp:
        def __init__(self, code):
            self.status_code, self.text = code, "oops"

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": '{"ok": 1}'}}

    replies = [Resp(500), Resp(200)]
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: replies.pop(0))
    assert llm_mod.OllamaLLM(Config()).json("s", "u") == {"ok": 1}


def test_relation_routing_and_uncited_sentences(graph, cfg):
    from graphscout.qa import routed_properties

    assert routed_properties("What does X improve, and what is it compared with?") == ["improves", "comparedWith"]
    g, _ = graph
    store = GraphStore(g)

    class LLM:  # an explorer that picks nothing: routing alone must still find the fact
        def json(self, system, user):
            if "plan how" in system:
                return {"entities": ["MethodX"], "plan": "p"}
            if "explore" in system:
                return {"selected": [], "done": True}
            return {"answer": "MethodX beats RAG [E1]. It is great.", "cited": ["E1"]}

    trace = answer_question(LLM(), store, cfg, "Which approach does MethodX outperform?")
    assert any("--outperforms-->" in ev["fact"] for ev in trace.evidence)
    assert trace.uncited_sentences == 1
    assert "1 of 2 sentences have no citation" in trace.confidence


def test_type_word_ignored_when_specific_entity_named(graph, cfg):
    g, _ = graph
    store = GraphStore(g)

    class LLM:
        def json(self, system, user):
            if "plan how" in system:
                return {"entities": ["HotpotQA", "methods"], "plan": "p"}
            if "explore" in system:
                return {"selected": [], "done": True}
            return {"answer": "x [E1].", "cited": ["E1"]}

    trace = answer_question(LLM(), store, cfg, "Which methods are evaluated on HotpotQA?")
    assert trace.seeds == ["HotpotQA"]
