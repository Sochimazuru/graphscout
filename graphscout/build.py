"""Step 3: merge all extractions into one RDF knowledge graph with provenance.

Each extracted fact is stored twice:
- as a plain triple  (kg:GraphRAG kg:uses kg:Knowledge_Graph), for easy querying;
- as a reified rdf:Statement that records which paper it came from (prov:wasDerivedFrom)
  and the exact relation text the LLM produced (kg:originalRelation).
That way every answer can point back to its source paper.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import PROV, RDF, RDFS, SKOS, XSD

from .config import Config
from .extract import THIS_PAPER, validate
from .resolve import Resolver
from .schema import KG, SDO, align_relation, ontology_triples


def paper_uri(paper_id: str) -> URIRef:
    return KG[f"paper/{paper_id.replace('/', '_')}"]


def entity_uri(cid: str) -> URIRef:
    return KG[f"entity/{cid}"]


def new_graph() -> Graph:
    g = Graph()
    g.bind("kg", KG)
    g.bind("schema", SDO)
    g.bind("prov", PROV)
    g.bind("skos", SKOS)
    return g


def build_graph(papers: list[dict], extractions: list[dict]) -> tuple[Graph, dict]:
    g = new_graph()
    for t in ontology_triples():
        g.add(t)

    by_id = {p["id"]: p for p in papers}
    for p in papers:
        pu = paper_uri(p["id"])
        g.add((pu, RDF.type, KG.Paper))
        g.add((pu, RDFS.label, Literal(p["title"])))
        g.add((pu, SDO.name, Literal(p["title"])))
        g.add((pu, SDO.url, URIRef(p["url"])))
        if p.get("published"):
            g.add((pu, SDO.datePublished, Literal(p["published"], datatype=XSD.date)))
        for a in p.get("authors", [])[:10]:
            g.add((pu, SDO.author, Literal(a)))

    # Re-clean cached extractions, so build-time fixes apply without re-running the LLM.
    extractions = [
        {**validate(ex, max_triples=10**6), "paper_id": ex["paper_id"]} for ex in extractions
    ]

    resolver = Resolver()
    for ex in extractions:
        for e in ex["entities"]:
            resolver.add(e["name"], e["type"])
    resolver.build()

    for cid, c in resolver.clusters.items():
        eu = entity_uri(cid)
        g.add((eu, RDF.type, KG[c["type"]]))
        g.add((eu, RDFS.label, Literal(c["label"])))
        for alias in c["aliases"]:
            g.add((eu, SKOS.altLabel, Literal(alias)))

    relation_counts: Counter = Counter()
    skipped = duplicates = 0
    seen = set()
    for ex in extractions:
        pid = ex["paper_id"]
        if pid not in by_id:
            continue
        pu = paper_uri(pid)
        for e in ex["entities"]:
            c = resolver.lookup(e["name"])
            if c:
                g.add((pu, KG.mentions, entity_uri(c["id"])))
        for t in ex["triples"]:
            s = pu if t["subject"] == THIS_PAPER else _node(resolver, t["subject"])
            o = pu if t["object"] == THIS_PAPER else _node(resolver, t["object"])
            if s is None or o is None or s == o:
                skipped += 1
                continue
            prop = align_relation(t["relation"])
            p_uri = KG[prop]
            key = (s, p_uri, o, pid, t["relation"])
            if key in seen:  # the model repeated itself; count each fact once
                duplicates += 1
                continue
            seen.add(key)
            relation_counts[prop] += 1
            g.add((s, p_uri, o))
            st = KG["stmt/" + hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16]]
            g.add((st, RDF.type, RDF.Statement))
            g.add((st, RDF.subject, s))
            g.add((st, RDF.predicate, p_uri))
            g.add((st, RDF.object, o))
            g.add((st, PROV.wasDerivedFrom, pu))
            g.add((st, KG.originalRelation, Literal(t["relation"])))

    mentions = sum(len(ex["entities"]) for ex in extractions)
    total_rel = sum(relation_counts.values())
    stats = {
        "papers": len(papers),
        "papers_with_extractions": len({ex["paper_id"] for ex in extractions}),
        "entity_mentions": mentions,
        "entities_after_resolution": len(resolver.clusters),
        "entity_types": dict(Counter(c["type"] for c in resolver.clusters.values())),
        "relation_facts": total_rel,
        "relations_by_property": dict(relation_counts.most_common()),
        "relations_mapped_to_schema": total_rel - relation_counts.get("relatedTo", 0),
        "triples_skipped": skipped,
        "duplicate_triples_removed": duplicates,
        "rdf_triples_total": len(g),
    }
    return g, stats


def _node(resolver: Resolver, name: str):
    c = resolver.lookup(name)
    return entity_uri(c["id"]) if c else None


def save(g: Graph, stats: dict, cfg: Config) -> None:
    cfg.ensure_dirs()
    g.serialize(destination=str(cfg.graph_file), format="turtle")
    (cfg.out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_graph(cfg: Config) -> Graph:
    if not cfg.graph_file.exists():
        raise FileNotFoundError(f"{cfg.graph_file} not found. Run `python -m graphscout build` first.")
    g = new_graph()
    g.parse(str(cfg.graph_file), format="turtle")
    return g
