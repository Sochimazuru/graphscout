# GraphScout

GraphScout reads a small set of research papers, uses a local LLM to turn them into a knowledge graph, and answers multi-hop questions by letting a few small agents walk that graph. Every answer comes with the path of facts behind it and the papers those facts came from.

I built it as a compact, end-to-end version of a research pipeline for LLM-based knowledge graph construction and graph-enhanced retrieval augmented generation (GraphRAG). The corpus is itself about GraphRAG and knowledge graphs, so the graph describes the field it belongs to.

Everything runs locally on a laptop CPU with [Ollama](https://ollama.com). No API keys, no GPU.

An extended abstract (a proof of concept prepared for a PhD application) describing the method, the maths behind each step and the results is in [`paper/graphscout.pdf`](paper/graphscout.pdf).

## How it works

```
arXiv API ──► 1. Fetch ──► 2. Extract (LLM, open IE) ──► 3. Align + resolve ──► RDF graph (Turtle)
                                                                                 │        │
                                                                     SPARQL ◄────┘        └──► Neo4j (Cypher)
                                                                        │
              question ──► Planner ──► Explorer (hop by hop, SPARQL) ──► Answerer ──► answer + evidence path
```

1. **Fetch.** Pulls 20 papers (title, abstract, authors, date) from the public arXiv API.
2. **Extract.** A local LLM reads each abstract and returns entities and `(subject, relation, object)` triples as JSON. Relations are open, free text: whatever verb the paper implies. Results are cached per paper.
3. **Align to a schema.** Open relations are mapped onto a small ontology (`proposes`, `uses`, `addresses`, `evaluatedOn`, `measuredBy`, `improves`, `outperforms`, `comparedWith`, `extends`, `partOf`, `reports`). The mapping rules were refined by reading the relation wordings the model actually produced on this corpus. Anything that does not fit is kept as `relatedTo` with the original wording, so no information is thrown away. Entity classes (`Method`, `Task`, `Dataset`, `Metric`, `Model`, `Concept`) are linked to [Schema.org](https://schema.org) classes, and provenance uses [PROV-O](https://www.w3.org/TR/prov-o/).
4. **Resolve entities.** "Retrieval-Augmented Generation (RAG)", "RAG" and "retrieval augmented generation" become one node. The rules are deterministic and explainable: normalised names, explicit acronyms, and initials matching only when there is exactly one candidate.
5. **Store.** The graph is written as RDF (`outputs/graph.ttl`). Each fact is also stored as a reified statement pointing to its source paper, so any answer can be traced back. The same graph is exported to Neo4j as Cypher.
6. **Answer questions.** Three agents cooperate:
   - **Planner** picks the entities in the question and finds them in the graph.
   - **Routing** comes first: relation words in the question ("improve", "compared", "evaluated", ...) pull in edges with the matching schema property straight from the start nodes. This keeps a small model from overlooking the obvious facts.
   - **Explorer** looks at the edges around the current nodes (a SPARQL query), chooses which ones to follow, and repeats for up to 3 hops.
   - **Answerer** writes the answer from the collected evidence only, citing each fact. Each answer reports two grounding signals: the evidence IDs the model says it cited (self-reported), and the number of answer sentences with no inline citation (measured from the answer text). Only the second is independent of the model.
   - Counting questions ("which datasets appear in more than one paper?") are answered better by SPARQL than by the agents, so `python -m graphscout sparql` covers those.

## Run it

You need Python 3.10+ and Ollama.

```bash
git clone https://github.com/<your-username>/graphscout.git
cd graphscout
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5:3b          # or any model you have; my results use qwen2.5:1.5b
python -m pytest -q             # plumbing tests, no LLM needed

python -m graphscout all                    # fetch, extract, build, SPARQL report, visualisation
python -m graphscout ask --file questions.txt
# with another model: python -m graphscout --model qwen2.5:1.5b all
```

Extraction is the slow step on a CPU. It caches each paper, so you can stop and restart at any time.

Single steps:

| Command | What it does |
|---|---|
| `python -m graphscout fetch` | download papers into `data/papers.json` |
| `python -m graphscout extract` | LLM extraction into `data/extractions/` (add `--force` to redo) |
| `python -m graphscout build` | build `outputs/graph.ttl`, `outputs/graph.cypher`, `outputs/stats.json` |
| `python -m graphscout sparql` | run the ready-made queries, save `outputs/sparql_results.md` |
| `python -m graphscout ask "..."` | answer one question with the agents |
| `python -m graphscout viz` | interactive graph at `outputs/graph.html` |
| `python -m graphscout audit` | sample 40 facts for manual checking |
| `python -m graphscout audit-score` | precision from your labels |

Settings (environment variables): `GRAPHSCOUT_MODEL` (default `qwen2.5:3b`), `GRAPHSCOUT_QUERY` (arXiv search), `GRAPHSCOUT_N_PAPERS` (default 20), `OLLAMA_URL`. You can also pass `--model` to any command.

### Neo4j (optional)

`build` always writes `outputs/graph.cypher`. To load it into Neo4j:

```bash
docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/graphscout neo4j:5
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=graphscout
python -m graphscout build      # now also pushes into Neo4j
```

Then open http://localhost:7474 and try `MATCH (m:Method)-[r]->(n) RETURN m, r, n LIMIT 100`.

## Measuring quality

`audit` writes a random sample of 40 extracted facts, each next to the abstract it came from. A fact counts as correct only if the abstract states it, with the right direction and relation. Labels were drafted by Claude (an LLM) using that rubric and reviewed by me. Every fact marked incorrect has a written reason in `outputs/audit_sample.csv`. `audit-score` computes precision from the labels.

## Results

All numbers come from files in `outputs/`, from one run with `qwen2.5:1.5b` on a laptop CPU (no GPU).

**Graph** (`outputs/stats.json`)

| | |
|---|---|
| Papers | 20 |
| Entity mentions extracted | 252 |
| Entities after resolution | 193 |
| Relation facts (after removing 13 repeated triples) | 99 |
| Facts mapped to a schema property | 92 of 99 |

Of the 7 facts left open (`relatedTo`), 6 are cases where the model wrote just "on" as the relation.

**Extraction precision** (`outputs/audit_score.json`): **0.625 (25 of 40)** on the random sample.

The 15 errors fall into a few groups:
- **Wrong relation or object** (7): for example, "outperforms token costs" where the paper reduces token costs, or a dataset recorded as something a method is compared with.
- **Garbled relation "on"** (3): the model wrote "on" as the relation.
- **Wrong subject** (3): for example, a method credited with proposing something its paper proposed, or a method "proposing" itself.
- **Reversed comparison** (1): "Vector RAG outperforms GraphRAG", where the abstract says the opposite.
- **Hallucinated entity** (1): HotpotQA, which does not appear in the source abstract.

**Question answering** (`outputs/answers.md`): five questions, each with the answer, the routing step and hops taken, and the evidence with source papers. Judged by hand against the source abstracts, 2 answers are correct (Inference-Scaled GraphRAG, BEACON-SP), 1 is partly correct (TagRAG) and 2 are wrong (HotpotQA, GraphRAG comparison). Both wrong answers trace to extraction errors or to the answerer going beyond its evidence. None of the 6 answer sentences carries an inline citation, while the model's own cited-ID list named every evidence fact for 3 of the 5 answers, so the self-reported list is not a reliable grounding signal.

Also in `outputs/`: `sparql_results.md` (ready-made queries), `graph.html` (interactive graph), `graph.ttl` (RDF), `graph.cypher` (Neo4j).

## Limitations and next steps

- **Small model.** A 1.5B model fits a laptop CPU but makes the errors listed above. The pipeline is model-agnostic, so a larger model is a one-flag change (`--model`), and the audit can be re-run to compare.
- **Grounding filter.** Dropping any entity whose name does not appear in its source abstract would have caught the HotpotQA hallucination automatically.
- **Text only, abstracts only.** A research version would read full papers, including figures and tables, with a multimodal model.
- **Rule-based schema alignment.** The mapping rules were refined by reading this run's real relation wordings. A next step is LLM- or embedding-based alignment to larger existing ontologies (Wikidata, the Open Research Knowledge Graph), with the rules as a fallback.
- **Simple entity resolution.** Deterministic rules are easy to explain but miss paraphrases. Embedding similarity plus an LLM check would catch more.
- **Small evaluation.** One precision check on 40 facts. A fuller study would add recall, compare plain chunk-based RAG with graph exploration on the same questions, and use a multi-hop QA benchmark.

## Project layout

```
graphscout/
  fetch.py         arXiv download
  llm.py           Ollama client (JSON mode)
  extract.py       open information extraction prompt + validation + cache
  schema.py        ontology, Schema.org alignment, relation mapping
  resolve.py       entity resolution
  build.py         RDF graph with provenance
  neo4j_export.py  Cypher export and optional push
  store.py         SPARQL neighbour and provenance queries
  qa.py            Planner, Explorer, Answerer agents
  queries.py       ready-made SPARQL queries
  visualize.py     interactive HTML graph
  audit.py         manual precision check
tests/             plumbing tests with hand-written fixtures (not results)
```
