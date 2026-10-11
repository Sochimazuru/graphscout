"""The small ontology that open extractions are aligned to.

Open information extraction gives free-text relations ("leverages", "is tested on").
Here they are mapped onto a fixed set of properties so the graph stays consistent,
and the classes are anchored to Schema.org and PROV-O, two existing vocabularies.
"""
from __future__ import annotations

import re

from rdflib import Namespace
from rdflib.namespace import OWL, PROV, RDF, RDFS

KG = Namespace("http://example.org/graphscout/")
SDO = Namespace("https://schema.org/")

# Entity classes the extractor may use, with a short definition for the prompt.
ENTITY_TYPES: dict[str, str] = {
    "Method": "a technique, algorithm, framework or system (e.g. GraphRAG, beam search)",
    "Task": "a problem being solved (e.g. multi-hop question answering)",
    "Dataset": "a named dataset or benchmark (e.g. HotpotQA)",
    "Metric": "an evaluation measure (e.g. F1, exact match)",
    "Model": "a specific named language model (e.g. Llama 3, GPT-4)",
    "Concept": "any other important idea (e.g. knowledge graph, hallucination)",
}

# Link each class to an existing, widely used vocabulary.
SCHEMA_ALIGNMENT = {
    "Paper": SDO.ScholarlyArticle,
    "Dataset": SDO.Dataset,
    "Model": SDO.SoftwareApplication,
    "Method": SDO.DefinedTerm,
    "Task": SDO.DefinedTerm,
    "Metric": SDO.DefinedTerm,
    "Concept": SDO.DefinedTerm,
}

# Canonical properties, checked in order. The first pattern that matches wins.
RELATION_RULES: list[tuple[str, str]] = [
    ("outperforms", r"outperform|surpass|beat|better than|gains? over|exceed"),
    ("comparedWith", r"compar|versus|\bvs\b|against|benchmark(s|ed)? against"),
    ("improves", r"improv|enhanc|boost|increas|achiev|accelerat|speed"),
    ("extends", r"extend|buil(d|ds|t) (up)?on|based upon|generaliz|variant of|augment"),
    ("proposes", r"propos|introduc|present|develop|design|contribut|construct|build|creat|establish"),
    ("evaluatedOn", r"evaluat|test(ed)? on|benchmark|experiment|validat|assess|implemented on"),
    ("measuredBy", r"measur|metric|scor"),
    ("addresses", r"address|tackl|solv|target|aim|focus|for |applied to|mitigat|reduc|enabl|support|standardiz"),
    ("partOf", r"part of|component|consist|compris|contain|includ|compos"),
    ("uses", r"use|employ|leverag|utiliz|rel(y|ies) on|based on|adopt|integrat|incorporat|combin|apply|via|deploy|run on"),
    ("reports", r"demonstrat|show|find|highlight|explain|analy|suggest|reveal|report|observ|conclud|argu"),
]
FALLBACK_RELATION = "relatedTo"

RELATION_DESCRIPTIONS = {
    "proposes": "paper or method introduces something new",
    "addresses": "method addresses a task or problem",
    "uses": "method uses another method, model or concept",
    "evaluatedOn": "method or paper is evaluated on a dataset",
    "measuredBy": "evaluation reports a metric",
    "outperforms": "method beats another method",
    "comparedWith": "method or paper is compared with another method",
    "improves": "method improves a task, metric or quality",
    "reports": "paper or method reports a finding (object is a short claim)",
    "extends": "method builds on or improves another",
    "partOf": "component belongs to a larger system",
    "relatedTo": "open relation with no schema match (original wording kept)",
}


def align_relation(raw: str) -> str:
    # camelCase -> words, so "evaluatedOn" matches too
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw.strip()).lower().replace("_", " ") + " "
    for prop, pattern in RELATION_RULES:
        if re.search(pattern, text):
            return prop
    return FALLBACK_RELATION


def align_type(raw: str) -> str:
    t = (raw or "").strip().lower()
    for name in ENTITY_TYPES:
        if t == name.lower():
            return name
    synonyms = {
        "benchmark": "Dataset", "corpus": "Dataset", "data": "Dataset",
        "llm": "Model", "language model": "Model",
        "algorithm": "Method", "framework": "Method", "system": "Method", "technique": "Method",
        "approach": "Method", "architecture": "Method", "tool": "Method",
        "problem": "Task", "application": "Task",
        "measure": "Metric", "evaluation metric": "Metric",
    }
    return synonyms.get(t, "Concept")


def ontology_triples():
    """Class and property declarations written into every graph."""
    yield KG.Paper, RDF.type, OWL.Class
    yield KG.Paper, RDFS.subClassOf, SCHEMA_ALIGNMENT["Paper"]
    yield KG.Entity, RDF.type, OWL.Class
    for cls, desc in ENTITY_TYPES.items():
        yield KG[cls], RDF.type, OWL.Class
        yield KG[cls], RDFS.subClassOf, KG.Entity
        yield KG[cls], RDFS.subClassOf, SCHEMA_ALIGNMENT[cls]
        yield KG[cls], RDFS.comment, _lit(desc)
    for prop, desc in RELATION_DESCRIPTIONS.items():
        yield KG[prop], RDF.type, OWL.ObjectProperty
        yield KG[prop], RDFS.comment, _lit(desc)
    yield KG.mentions, RDF.type, OWL.ObjectProperty
    yield KG.originalRelation, RDF.type, OWL.DatatypeProperty
    yield KG.originalRelation, RDFS.comment, _lit("relation text exactly as the LLM extracted it")
    yield PROV.wasDerivedFrom, RDFS.comment, _lit("links an extracted statement to its source paper")


def _lit(text: str):
    from rdflib import Literal

    return Literal(text, lang="en")
