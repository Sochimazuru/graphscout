"""Manual quality check of extracted facts.

`audit` writes a random sample of facts to outputs/audit_sample.csv with the
abstract sentence context. You mark each row correct (1) or wrong (0) in the
`correct` column, then `audit-score` reports precision from your labels.
This is the only accuracy number the project reports, and it comes from you.
"""
from __future__ import annotations

import csv
import random

from rdflib.namespace import PROV, RDF

from .config import Config
from .schema import KG
from .store import GraphStore


def write_sample(g, papers: list[dict], cfg: Config, n: int = 40, seed: int = 0) -> str:
    store = GraphStore(g)
    abstracts = {p["id"].replace("/", "_"): p for p in papers}
    stmts = sorted(set(g.subjects(RDF.type, RDF.Statement)), key=str)
    random.Random(seed).shuffle(stmts)
    path = cfg.out_dir / "audit_sample.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["correct", "subject", "property", "object", "original_relation", "paper_id", "paper_title", "abstract"])
        for st in stmts[:n]:
            s, p, o = g.value(st, RDF.subject), g.value(st, RDF.predicate), g.value(st, RDF.object)
            src = str(g.value(st, PROV.wasDerivedFrom)).rsplit("/", 1)[-1]
            paper = abstracts.get(src, {})
            w.writerow([
                "",
                store.label(str(s)),
                str(p).rsplit("/", 1)[-1],
                store.label(str(o)),
                str(g.value(st, KG.originalRelation) or ""),
                src,
                paper.get("title", ""),
                paper.get("abstract", ""),
            ])
    return str(path)


def score(cfg: Config) -> dict:
    path = cfg.out_dir / "audit_sample.csv"
    labels = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("correct") or "").strip()
            if v in {"0", "1"}:
                labels.append(int(v))
    if not labels:
        raise ValueError(f"No rows labelled yet. Put 1 or 0 in the `correct` column of {path}.")
    return {"labelled": len(labels), "correct": sum(labels), "precision": round(sum(labels) / len(labels), 3)}
