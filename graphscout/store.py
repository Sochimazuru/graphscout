"""Read-only access to the RDF graph: entity lookup and SPARQL neighbour queries."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from rdflib import Graph, URIRef
from rdflib.namespace import PROV, RDF, RDFS, SKOS

from .resolve import norm_key, split_acronym
from .schema import KG, RELATION_DESCRIPTIONS

FACT_PROPS = list(RELATION_DESCRIPTIONS) + ["mentions"]

NEIGHBOURS_SPARQL = """
PREFIX kg: <http://example.org/graphscout/>
SELECT ?dir ?p ?other WHERE {
  { ?node ?p ?other . BIND("out" AS ?dir) }
  UNION
  { ?other ?p ?node . BIND("in" AS ?dir) }
  FILTER(?p IN (%s))
}
""" % ", ".join(f"kg:{p}" for p in FACT_PROPS)

PROVENANCE_SPARQL = """
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX kg: <http://example.org/graphscout/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?paper ?orig WHERE {
  ?st rdf:subject ?s ; rdf:predicate ?p ; rdf:object ?o ; prov:wasDerivedFrom ?paper .
  OPTIONAL { ?st kg:originalRelation ?orig }
}
"""


@dataclass(frozen=True)
class Edge:
    subj: str
    pred: str
    obj: str
    subj_label: str
    obj_label: str
    sources: tuple = field(default=())
    original: tuple = field(default=())

    def text(self) -> str:
        orig = f' (paper wording: "{self.original[0]}")' if self.original else ""
        return f"{self.subj_label} --{self.pred}--> {self.obj_label}{orig}"


class GraphStore:
    def __init__(self, g: Graph):
        self.g = g
        self.labels: dict[str, str] = {}
        self.types: dict[str, str] = {}
        self.index: list[tuple[str, str]] = []  # (normalised name, uri)
        for node, label in g.subject_objects(RDFS.label):
            t = g.value(node, RDF.type)
            if t is None or str(t).startswith("http://www.w3.org/2002/07/owl#"):
                continue
            uri = str(node)
            self.labels[uri] = str(label)
            self.types[uri] = str(t).replace(str(KG), "")
            self.index.append((norm_key(split_acronym(str(label))[0]), uri))
            for alt in g.objects(node, SKOS.altLabel):
                self.index.append((norm_key(str(alt)), uri))

    def label(self, uri: str) -> str:
        return self.labels.get(uri, uri.rsplit("/", 1)[-1])

    def find(self, mention: str, k: int = 2, threshold: float = 0.88) -> list[str]:
        """Fuzzy match a phrase from the question to graph nodes (papers excluded)."""
        q = norm_key(split_acronym(mention)[0])
        if not q:
            return []
        q_tokens = set(q.split())
        scored: dict[str, float] = {}
        for name, uri in self.index:
            if self.types.get(uri) == "Paper" or not name:
                continue
            ratio = difflib.SequenceMatcher(None, q, name).ratio()
            tokens = set(name.split())
            jacc = len(q_tokens & tokens) / len(q_tokens | tokens)
            score = 1.0 if q == name else max(ratio, jacc)
            if score >= threshold and score > scored.get(uri, 0):
                scored[uri] = score
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        if ranked and ranked[0][1] == 1.0:  # an exact match beats near misses
            return [ranked[0][0]]
        return [u for u, _ in ranked[:k]]

    def neighbours(self, uri: str) -> list[Edge]:
        rows = self.g.query(NEIGHBOURS_SPARQL, initBindings={"node": URIRef(uri)})
        edges = []
        for direction, pred, other in rows:
            p = str(pred).replace(str(KG), "")
            s, o = (uri, str(other)) if str(direction) == "out" else (str(other), uri)
            sources, original = self.provenance(s, p, o)
            edges.append(Edge(s, p, o, self.label(s), self.label(o), sources, original))
        return edges

    def provenance(self, s: str, p: str, o: str) -> tuple[tuple, tuple]:
        if p == "mentions":
            return (s.rsplit("/", 1)[-1],), ()
        rows = self.g.query(
            PROVENANCE_SPARQL,
            initBindings={"s": URIRef(s), "p": KG[p], "o": URIRef(o)},
        )
        papers, orig = set(), set()
        for paper, o_text in rows:
            papers.add(str(paper).rsplit("/", 1)[-1])
            if o_text is not None:
                orig.add(str(o_text))
        return tuple(sorted(papers)), tuple(sorted(orig))

    def paper_title(self, paper_id: str) -> str:
        return self.labels.get(str(KG[f"paper/{paper_id}"]), paper_id)

    def statement_count(self) -> int:
        return len(set(self.g.subjects(RDF.type, RDF.Statement)))

    def sources_of(self, st) -> list[str]:
        return [str(x) for x in self.g.objects(st, PROV.wasDerivedFrom)]
