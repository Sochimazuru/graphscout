"""Interactive HTML view of the graph (open outputs/graph.html in a browser)."""
from __future__ import annotations

from rdflib.namespace import RDF, RDFS

from .config import Config
from .schema import KG, RELATION_DESCRIPTIONS

COLOURS = {
    "Paper": "#9aa0a6",
    "Method": "#1a73e8",
    "Task": "#e8710a",
    "Dataset": "#188038",
    "Metric": "#a142f4",
    "Model": "#d93025",
    "Concept": "#5f6368",
}


def render(g, cfg: Config, include_papers: bool = True) -> str:
    from pyvis.network import Network

    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", cdn_resources="remote")
    net.barnes_hut(gravity=-6000, spring_length=140)
    added = set()

    def add(node):
        if node in added:
            return
        t = str(g.value(node, RDF.type)).replace(str(KG), "")
        if t == "Paper" and not include_papers:
            return
        label = str(g.value(node, RDFS.label) or node)
        short = label if len(label) <= 40 else label[:37] + "..."
        net.add_node(str(node), label=short, title=f"{t}: {label}", color=COLOURS.get(t, "#5f6368"),
                     shape="box" if t == "Paper" else "dot", size=10 if t == "Paper" else 16)
        added.add(node)

    for prop in RELATION_DESCRIPTIONS:
        for s, o in g.subject_objects(KG[prop]):
            add(s)
            add(o)
            if s in added and o in added:
                net.add_edge(str(s), str(o), label=prop, title=prop, arrows="to")

    path = cfg.out_dir / "graph.html"
    net.write_html(str(path), notebook=False)
    return str(path)
