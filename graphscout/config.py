"""Central settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_QUERY = (
    'abs:"GraphRAG" OR (abs:"knowledge graph" AND abs:"retrieval-augmented generation")'
)


@dataclass
class Config:
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    # A small model, so it runs on a laptop CPU. Swap for a bigger one if you have a GPU.
    model: str = field(default_factory=lambda: os.getenv("GRAPHSCOUT_MODEL", "qwen2.5:3b"))
    llm_timeout: int = field(default_factory=lambda: int(os.getenv("GRAPHSCOUT_TIMEOUT", "300")))
    # Hard cap on output length. Small models in JSON mode can otherwise keep generating
    # whitespace until the timeout, which looks like a hang.
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("GRAPHSCOUT_MAX_TOKENS", "1024")))
    arxiv_query: str = field(default_factory=lambda: os.getenv("GRAPHSCOUT_QUERY", DEFAULT_QUERY))
    n_papers: int = field(default_factory=lambda: int(os.getenv("GRAPHSCOUT_N_PAPERS", "20")))
    max_triples_per_paper: int = 12
    max_hops: int = 3
    beam_width: int = 4
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("GRAPHSCOUT_DATA", "data")))
    out_dir: Path = field(default_factory=lambda: Path(os.getenv("GRAPHSCOUT_OUT", "outputs")))

    @property
    def papers_file(self) -> Path:
        return self.data_dir / "papers.json"

    @property
    def extraction_dir(self) -> Path:
        return self.data_dir / "extractions"

    @property
    def graph_file(self) -> Path:
        return self.out_dir / "graph.ttl"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
