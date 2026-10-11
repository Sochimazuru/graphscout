"""Step 2: LLM-based open information extraction, one paper at a time.

Results are cached per paper in data/extractions/<id>.json, so a crash or a
re-run never repeats work that already finished.
"""
from __future__ import annotations

import json
import re
import sys
import time

from .config import Config
from .llm import LLMError, ModelNotFound
from .schema import ENTITY_TYPES

THIS_PAPER = "THIS_PAPER"

SYSTEM = (
    "You extract a knowledge graph from scientific abstracts. "
    "Only state facts that are explicitly written in the text. Never guess. "
    "Answer with JSON only."
)


def build_prompt(paper: dict, max_triples: int) -> str:
    types = "\n".join(f"- {k}: {v}" for k, v in ENTITY_TYPES.items())
    return f"""Read this paper abstract and extract entities and relations.

Entity types:
{types}

Rules:
- Use short, canonical names (e.g. "GraphRAG", not "our GraphRAG-based approach").
- If an entity has an acronym, write it as "Full Name (ACRONYM)".
- Use "{THIS_PAPER}" as the subject for what this paper itself proposes, evaluates or addresses.
- A relation is a short verb phrase, e.g. "proposes", "uses", "evaluated on", "outperforms".
- At most {max_triples} entities and at most {max_triples} triples. Prefer the most important facts.
- Write compact JSON. Do not repeat items.

Return JSON with exactly this shape:
{{"entities": [{{"name": "...", "type": "Method"}}],
  "triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Title: {paper["title"]}
Abstract: {paper["abstract"]}"""


_QUANTITY = re.compile(r"^\s*\d[\d.,]*(%|\s|-|$)")


def clean_name(name) -> str:
    """Tidy a name from the model. Returns "" for things that are not entities."""
    if isinstance(name, list):
        name = name[0] if name else ""
    name = str(name).strip()
    # the model sometimes writes a list as a string: "['hallucinations']"
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1].strip()
    name = name.strip("'\" ").strip()
    # LaTeX from abstracts: "\text" can arrive as a tab followed by "ext"
    name = name.replace("\t", "\\t")
    name = re.sub(r"\\text\{([^}]*)\}", r"\1", name).replace("$", "")
    name = re.sub(r"\s+", " ", name).strip()
    if _QUANTITY.match(name):  # "100 patients", "76.4%", "1,500-query benchmark"
        return ""
    return name


def validate(raw: dict, max_triples: int) -> dict:
    """Keep only well-formed items; add any entity a triple mentions but the list forgot."""
    entities: dict[str, str] = {}
    for e in raw.get("entities", []) or []:
        if isinstance(e, dict):
            name = clean_name(e.get("name", ""))
            if name:
                entities.setdefault(name, str(e.get("type", "Concept")))
    triples = []
    for t in (raw.get("triples", []) or [])[:max_triples]:
        if not isinstance(t, dict):
            continue
        s, o = clean_name(t.get("subject", "")), clean_name(t.get("object", ""))
        r = str(t.get("relation", "")).strip()
        if not (s and r and o) or s == o:
            continue
        for name in (s, o):
            if name != THIS_PAPER:
                entities.setdefault(name, "Concept")
        triples.append({"subject": s, "relation": r, "object": o})
    entities.pop(THIS_PAPER, None)
    return {
        "entities": [{"name": n, "type": t} for n, t in entities.items()],
        "triples": triples,
    }


def extract_paper(llm, paper: dict, cfg: Config) -> dict:
    raw = llm.json(SYSTEM, build_prompt(paper, cfg.max_triples_per_paper))
    result = validate(raw, cfg.max_triples_per_paper)
    result["paper_id"] = paper["id"]
    result["model"] = cfg.model
    return result


def extract_all(llm, papers: list[dict], cfg: Config, force: bool = False) -> list[dict]:
    cfg.ensure_dirs()
    results = []
    failures_in_a_row = 0
    for i, paper in enumerate(papers, 1):
        cache = cfg.extraction_dir / f"{paper['id'].replace('/', '_')}.json"
        if cache.exists() and not force:
            results.append(json.loads(cache.read_text(encoding="utf-8")))
            print(f"[{i}/{len(papers)}] cached   {paper['title'][:70]}")
            continue
        print(f"[{i}/{len(papers)}] extract  {paper['title'][:70]}", flush=True)
        started = time.time()
        try:
            result = extract_paper(llm, paper, cfg)
        except ModelNotFound:
            raise  # retrying other papers will not help
        except Exception as exc:  # keep going; one bad paper should not kill the run
            print(f"    skipped: {exc}", file=sys.stderr)
            failures_in_a_row += 1
            if failures_in_a_row >= 3:
                raise LLMError("3 papers in a row failed. Stopping; check the errors above.") from exc
            continue
        failures_in_a_row = 0
        cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    {len(result['entities'])} entities, {len(result['triples'])} triples "
              f"({time.time() - started:.0f}s)")
        results.append(result)
    return results


def load_extractions(cfg: Config) -> list[dict]:
    files = sorted(cfg.extraction_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError("No extractions found. Run `python -m graphscout extract` first.")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]
