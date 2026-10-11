"""GraphScout command line.

    python -m graphscout fetch            # download papers from arXiv
    python -m graphscout extract          # LLM extraction (cached per paper)
    python -m graphscout build            # merge into RDF, write Turtle + Cypher
    python -m graphscout sparql [name]    # canned SPARQL queries (no LLM)
    python -m graphscout ask "question"   # multi-agent graph QA
    python -m graphscout ask --file questions.txt
    python -m graphscout viz              # interactive HTML graph
    python -m graphscout audit            # sample facts for manual checking
    python -m graphscout audit-score      # precision from your labels
    python -m graphscout all              # fetch, extract, build, sparql, viz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def cmd_fetch(cfg, args):
    from .fetch import fetch_papers

    papers = fetch_papers(cfg)
    print(f"Fetched {len(papers)} papers into {cfg.papers_file}")
    for p in papers:
        print(f"  {p['id']}  {p['title'][:80]}")


def cmd_extract(cfg, args):
    from .extract import extract_all
    from .fetch import load_papers
    from .llm import OllamaLLM

    llm = OllamaLLM(cfg)
    llm.check()
    results = extract_all(llm, load_papers(cfg), cfg, force=args.force)
    print(f"Extractions available for {len(results)} papers in {cfg.extraction_dir}")


def cmd_build(cfg, args):
    from .build import build_graph, save
    from .extract import load_extractions
    from .fetch import load_papers
    from .neo4j_export import export

    g, stats = build_graph(load_papers(cfg), load_extractions(cfg))
    save(g, stats, cfg)
    print(json.dumps(stats, indent=2))
    print(f"Wrote {cfg.graph_file}")
    print(export(g, cfg))


def cmd_sparql(cfg, args):
    from .build import load_graph
    from .queries import QUERIES, run, run_custom, to_markdown

    g = load_graph(cfg)
    if args.query_file:
        headers, rows = run_custom(g, Path(args.query_file).read_text(encoding="utf-8"))
        print(to_markdown(args.query_file, headers, rows))
        return
    names = [args.name] if args.name else list(QUERIES)
    report = ["# SPARQL results", ""]
    for name in names:
        title, headers, rows = run(g, name)
        md = to_markdown(title, headers, rows)
        print(md)
        report.append(md)
    if not args.name:
        (cfg.out_dir / "sparql_results.md").write_text("\n".join(report), encoding="utf-8")
        print(f"Saved to {cfg.out_dir / 'sparql_results.md'}")


def cmd_ask(cfg, args):
    from dataclasses import asdict

    from .build import load_graph
    from .llm import LLMError, OllamaLLM
    from .qa import answer_question, trace_to_markdown
    from .store import GraphStore

    llm = OllamaLLM(cfg)
    llm.check()
    store = GraphStore(load_graph(cfg))
    if args.file:
        questions = [q.strip() for q in Path(args.file).read_text(encoding="utf-8").splitlines()
                     if q.strip() and not q.startswith("#")]
    elif args.question:
        questions = [args.question]
    else:
        sys.exit("Give a question, or --file questions.txt")
    md = [f"# Question answering results (model: {cfg.model})", ""]
    traces = []
    for q in questions:
        print(f"\n? {q}", flush=True)
        try:
            trace = answer_question(llm, store, cfg, q)
        except LLMError as exc:
            print(f"   skipped: {exc}")
            md.append(f"### Q: {q}\n\nSkipped: {exc}\n")
            continue
        block = trace_to_markdown(trace, store)
        print(block)
        md.append(block)
        traces.append(asdict(trace))
    if args.file:
        (cfg.out_dir / "answers.md").write_text("\n".join(md), encoding="utf-8")
        (cfg.out_dir / "answers.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {cfg.out_dir / 'answers.md'} and answers.json")


def cmd_viz(cfg, args):
    from .build import load_graph
    from .visualize import render

    print(f"Wrote {render(load_graph(cfg), cfg, include_papers=not args.no_papers)}")


def cmd_audit(cfg, args):
    from .audit import write_sample
    from .build import load_graph
    from .fetch import load_papers

    path = write_sample(load_graph(cfg), load_papers(cfg), cfg, n=args.n)
    print(f"Wrote {path}. Fill the `correct` column with 1 or 0, then run audit-score.")


def cmd_audit_score(cfg, args):
    from .audit import score

    result = score(cfg)
    (cfg.out_dir / "audit_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def cmd_all(cfg, args):
    if not cfg.papers_file.exists():
        cmd_fetch(cfg, args)
    cmd_extract(cfg, args)
    cmd_build(cfg, args)
    args.name, args.query_file = None, None
    cmd_sparql(cfg, args)
    args.no_papers = False
    cmd_viz(cfg, args)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="graphscout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Ollama model name (default: env GRAPHSCOUT_MODEL or qwen2.5:3b)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p = sub.add_parser("extract")
    p.add_argument("--force", action="store_true", help="ignore the cache and re-extract")
    sub.add_parser("build")
    p = sub.add_parser("sparql")
    p.add_argument("name", nargs="?", help="one of the canned query names")
    p.add_argument("--query-file", help="run your own .rq/.sparql file")
    p = sub.add_parser("ask")
    p.add_argument("question", nargs="?")
    p.add_argument("--file")
    p = sub.add_parser("viz")
    p.add_argument("--no-papers", action="store_true", help="hide paper nodes")
    p = sub.add_parser("audit")
    p.add_argument("-n", type=int, default=40)
    sub.add_parser("audit-score")
    p = sub.add_parser("all")
    p.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config()
    if args.model:
        cfg.model = args.model
    cfg.ensure_dirs()
    handlers = {
        "fetch": cmd_fetch, "extract": cmd_extract, "build": cmd_build, "sparql": cmd_sparql,
        "ask": cmd_ask, "viz": cmd_viz, "audit": cmd_audit, "audit-score": cmd_audit_score, "all": cmd_all,
    }
    try:
        handlers[args.cmd](cfg, args)
    except Exception as exc:
        from .llm import LLMError

        if isinstance(exc, (LLMError, FileNotFoundError)):
            sys.exit(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
