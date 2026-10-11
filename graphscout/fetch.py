"""Step 1: download paper metadata and abstracts from the public arXiv API."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import requests

from .config import Config

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=ATOM)
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        papers.append(
            {
                "id": arxiv_id,
                "title": _clean(entry.findtext("a:title", default="", namespaces=ATOM)),
                "abstract": _clean(entry.findtext("a:summary", default="", namespaces=ATOM)),
                "published": entry.findtext("a:published", default="", namespaces=ATOM)[:10],
                "authors": [
                    _clean(a.findtext("a:name", default="", namespaces=ATOM))
                    for a in entry.findall("a:author", ATOM)
                ],
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return papers


def fetch_papers(cfg: Config) -> list[dict]:
    params = {
        "search_query": cfg.arxiv_query,
        "start": 0,
        "max_results": cfg.n_papers,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=60)
    resp.raise_for_status()
    papers = parse_feed(resp.text)
    if not papers:
        raise RuntimeError("arXiv returned no papers. Check GRAPHSCOUT_QUERY.")
    cfg.ensure_dirs()
    cfg.papers_file.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    return papers


def load_papers(cfg: Config) -> list[dict]:
    if not cfg.papers_file.exists():
        raise FileNotFoundError(f"{cfg.papers_file} not found. Run `python -m graphscout fetch` first.")
    return json.loads(cfg.papers_file.read_text(encoding="utf-8"))
