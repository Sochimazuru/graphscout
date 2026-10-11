# GraphScout

GraphScout reads a small set of research papers, uses a local LLM to turn them into a knowledge graph, and answers multi-hop questions by letting a few small agents walk that graph. Every answer comes with the path of facts behind it and the papers those facts came from.

I built it as a compact, end-to-end version of a research pipeline for LLM-based knowledge graph construction and graph-enhanced retrieval augmented generation (GraphRAG). The corpus is itself about GraphRAG and knowledge graphs, so the graph describes the field it belongs to.

Everything runs locally on a laptop CPU with [Ollama](https://ollama.com). No API keys, no GPU.

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
3. **Align to a schema.** Open relations are mapped onto a small ontology (`proposes`, `uses`, `addresses`, `evaluatedOn`, `measuredBy`, `outperforms`, `extends`, `partOf`). Anything that does not fit is kept as `relatedTo` with the original wording, so no information is thrown away. Entity classes (`Method`, `Task`, `Dataset`, `Metric`, `Model`, `Concept`) are linked to [Schema.org](https://schema.org) classes, and provenance uses [PROV-O](https://www.w3.org/TR/prov-o/).
4. **Resolve entities.** "Retrieval-Augmented Generation (RAG)", "RAG" and "retrieval augmented generation" become one node. The rules are deterministic and explainable: normalised names, explicit acronyms, and initials matching only when there is exactly one candidate.
5. **Store.** The graph is written as RDF (`outputs/graph.ttl`). Each fact is also stored as a reified statement pointing to its source paper, so any answer can be traced back. The same graph is exported to Neo4j as Cypher.
6. **Answer questions.** Three agents cooperate:
   - **Planner** picks the entities in the question and finds them in the graph.
   - **Explorer** looks at the edges around the current nodes (a SPARQL query), chooses which ones to follow, and repeats for up to 3 hops.
   - **Answerer** writes the answer from the collected evidence only, citing each fact.

## Run it

You need Python 3.10+ and Ollama.

```bash
git clone https://github.com/<your-username>/graphscout.git
cd graphscout
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5:3b          # small enough for a laptop CPU
python -m pytest -q             # plumbing tests, no LLM needed

python -m graphscout all        # fetch, extract, build, SPARQL report, visualisation
python -m graphscout ask --file questions.txt
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

The project reports one accuracy number, and it comes from a human check, not from the model grading itself. `audit` writes a random sample of extracted facts next to the abstract they came from. I mark each one as correct or not, and `audit-score` computes precision.

## Results

_Filled in from my own run. All numbers below come from files in `outputs/`._

- Model: `qwen2.5:3b` on a laptop CPU
- Graph size (`outputs/stats.json`): papers, entities after resolution, facts, share of facts mapped to the schema
- Extraction precision on a manual sample (`outputs/audit_score.json`):
- Example questions with answers and evidence paths: `outputs/answers.md`
- SPARQL results: `outputs/sparql_results.md`
- Graph visualisation: `outputs/graph.html`

## Limitations and next steps

- **Text only, abstracts only.** The research version would read full papers, including figures and tables, with a multimodal model. Abstracts keep this demo fast on a CPU.
- **Small model.** A 3B model misses facts and sometimes mislabels types. The pipeline is model-agnostic; a bigger model is a one-flag change.
- **Rule-based schema alignment.** Relation mapping uses keyword rules. A natural next step is LLM- or embedding-based alignment to larger existing ontologies (Wikidata, the Open Research Knowledge Graph), with the rules as a fallback.
- **Simple entity resolution.** Deterministic rules are easy to explain but miss paraphrases. Embedding similarity plus an LLM check would catch more.
- **Small evaluation.** One manual precision check on a sample. A fuller study would add recall, compare plain chunk-based RAG against graph exploration on the same questions, and use a multi-hop QA benchmark.

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
