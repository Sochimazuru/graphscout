"""Step 4: graph-based question answering with three small cooperating agents.

Planner   reads the question and names the entities to start from.
Explorer  walks the graph hop by hop. At each hop it sees the edges around the
          current nodes (fetched with SPARQL) and picks the ones worth following.
Answerer  writes the answer using only the collected evidence, citing each fact.

The full trace (seeds, every hop, every chosen edge) is kept, so you can show
exactly how an answer was reached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .store import Edge, GraphStore

PLANNER_SYSTEM = (
    "You plan how to answer a question using a knowledge graph about research papers. "
    "Answer with JSON only."
)
EXPLORER_SYSTEM = (
    "You explore a knowledge graph to collect evidence for a question. "
    "Pick only edges that help answer it. Answer with JSON only."
)
ANSWERER_SYSTEM = (
    "You answer questions using only the numbered evidence given. "
    "If the evidence is not enough, say so plainly. Answer with JSON only."
)


@dataclass
class Trace:
    question: str
    plan: str = ""
    mentions: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    hops: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    answer: str = ""
    cited: list = field(default_factory=list)
    confidence: str = ""


class Planner:
    def __init__(self, llm, store: GraphStore):
        self.llm, self.store = llm, store

    def run(self, question: str, trace: Trace) -> list[str]:
        prompt = f"""Question: {question}

List the specific things (methods, tasks, datasets, models, concepts) named or implied
in the question that we should look up in the graph, and a one-sentence plan.
Return JSON: {{"entities": ["..."], "plan": "..."}}"""
        out = self.llm.json(PLANNER_SYSTEM, prompt)
        mentions = [m for m in out.get("entities", []) if isinstance(m, str)][:5]
        trace.plan = str(out.get("plan", ""))
        trace.mentions = mentions
        seeds: list[str] = []
        for m in mentions:
            for uri in self.store.find(m):
                if uri not in seeds:
                    seeds.append(uri)
        if not seeds:  # fall back to matching words of the question itself
            words = question.replace("?", " ").split()
            for n in (3, 2, 1):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    if len(phrase) < 3:
                        continue
                    for uri in self.store.find(phrase, k=1, threshold=0.85):
                        if uri not in seeds:
                            seeds.append(uri)
        trace.seeds = [self.store.label(s) for s in seeds[:6]]
        return seeds[:6]


class Explorer:
    def __init__(self, llm, store: GraphStore, cfg: Config):
        self.llm, self.store, self.cfg = llm, store, cfg

    def run(self, question: str, seeds: list[str], trace: Trace) -> list[Edge]:
        evidence: list[Edge] = []
        frontier, visited = list(seeds), set(seeds)
        for hop in range(1, self.cfg.max_hops + 1):
            candidates: list[Edge] = []
            for node in frontier:
                for e in self.store.neighbours(node):
                    if e not in evidence and e not in candidates:
                        candidates.append(e)
            # facts before generic "paper mentions X" links, and keep the prompt small
            candidates.sort(key=lambda e: e.pred == "mentions")
            candidates = candidates[:30]
            if not candidates:
                break
            listing = "\n".join(f"[{i}] {e.text()}" for i, e in enumerate(candidates))
            known = "\n".join(f"- {e.text()}" for e in evidence) or "(none yet)"
            prompt = f"""Question: {question}

Evidence collected so far:
{known}

Edges you can follow now (hop {hop} of {self.cfg.max_hops}):
{listing}

Pick at most {self.cfg.beam_width} edge numbers that help answer the question.
Set "done" to true if the evidence is already enough.
Return JSON: {{"selected": [0, 3], "done": false, "reason": "..."}}"""
            out = self.llm.json(EXPLORER_SYSTEM, prompt)
            picked = []
            for i in out.get("selected", []) or []:
                try:
                    idx = int(i)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(candidates) and candidates[idx] not in picked:
                    picked.append(candidates[idx])
            picked = picked[: self.cfg.beam_width]
            evidence.extend(picked)
            trace.hops.append(
                {
                    "hop": hop,
                    "frontier": [self.store.label(n) for n in frontier],
                    "candidates": len(candidates),
                    "chosen": [e.text() for e in picked],
                    "reason": str(out.get("reason", "")),
                }
            )
            next_frontier = []
            for e in picked:
                for n in (e.subj, e.obj):
                    if n not in visited:
                        visited.add(n)
                        next_frontier.append(n)
            frontier = next_frontier
            if out.get("done") or not frontier:
                break
        return evidence


class Answerer:
    def __init__(self, llm, store: GraphStore):
        self.llm, self.store = llm, store

    def run(self, question: str, evidence: list[Edge], trace: Trace) -> None:
        trace.evidence = [
            {"id": f"E{i}", "fact": e.text(), "sources": list(e.sources)}
            for i, e in enumerate(evidence, 1)
        ]
        if not evidence:
            trace.answer = "The graph has no evidence for this question."
            trace.confidence = "low"
            return
        listing = "\n".join(
            f"[E{i}] {e.text()}  (source: {', '.join(e.sources) or 'n/a'})"
            for i, e in enumerate(evidence, 1)
        )
        prompt = f"""Question: {question}

Evidence:
{listing}

Write a short answer (2 to 5 sentences) using only this evidence, citing facts like [E1].
Return JSON: {{"answer": "...", "cited": ["E1"], "confidence": "high|medium|low"}}"""
        out = self.llm.json(ANSWERER_SYSTEM, prompt)
        trace.answer = str(out.get("answer", "")).strip() or "No answer produced."
        trace.cited = [str(c) for c in out.get("cited", []) or []]
        trace.confidence = str(out.get("confidence", ""))


def answer_question(llm, store: GraphStore, cfg: Config, question: str) -> Trace:
    trace = Trace(question=question)
    seeds = Planner(llm, store).run(question, trace)
    evidence = Explorer(llm, store, cfg).run(question, seeds, trace) if seeds else []
    Answerer(llm, store).run(question, evidence, trace)
    return trace


def trace_to_markdown(trace: Trace, store: GraphStore) -> str:
    lines = [f"### Q: {trace.question}", "", f"**Answer** ({trace.confidence or 'n/a'} confidence): {trace.answer}", ""]
    lines.append(f"- Plan: {trace.plan or 'n/a'}")
    lines.append(f"- Mentions found in question: {', '.join(trace.mentions) or 'none'}")
    lines.append(f"- Start nodes in graph: {', '.join(trace.seeds) or 'none'}")
    for h in trace.hops:
        chosen = "; ".join(h["chosen"]) or "nothing"
        lines.append(f"- Hop {h['hop']}: from {', '.join(h['frontier'])} ({h['candidates']} edges seen), followed: {chosen}")
    if trace.evidence:
        lines += ["", "| Evidence | Fact | Source papers |", "|---|---|---|"]
        for ev in trace.evidence:
            src = ", ".join(f"[{s}](https://arxiv.org/abs/{s})" for s in ev["sources"]) or "n/a"
            lines.append(f"| {ev['id']} | {ev['fact'].replace('|', '/')} | {src} |")
    return "\n".join(lines) + "\n"
