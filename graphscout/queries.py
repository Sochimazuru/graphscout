"""Ready-made SPARQL queries. No LLM involved: these show the graph itself is useful.

Each one works on whatever corpus you fetched, because none hard-codes an entity name.
"""
from __future__ import annotations

PREFIXES = """PREFIX kg: <http://example.org/graphscout/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <https://schema.org/>
"""

QUERIES: dict[str, tuple[str, str]] = {
    "hubs": (
        "Entities mentioned by the most papers",
        """SELECT ?entity (COUNT(DISTINCT ?paper) AS ?papers) WHERE {
  ?paper kg:mentions ?e . ?e rdfs:label ?entity .
} GROUP BY ?entity ORDER BY DESC(?papers) LIMIT 15""",
    ),
    "shared_datasets": (
        "Datasets that appear in two or more papers",
        """SELECT ?dataset (COUNT(DISTINCT ?paper) AS ?papers) WHERE {
  ?d a kg:Dataset ; rdfs:label ?dataset .
  ?paper kg:mentions ?d .
} GROUP BY ?dataset HAVING (COUNT(DISTINCT ?paper) >= 2) ORDER BY DESC(?papers)""",
    ),
    "proposal_chains": (
        "What each paper proposes, and what that proposal uses (2-hop)",
        """SELECT ?paper ?proposed ?uses WHERE {
  ?p a kg:Paper ; rdfs:label ?paper ; kg:proposes ?m .
  ?m rdfs:label ?proposed ; kg:uses ?u .
  ?u rdfs:label ?uses .
} ORDER BY ?paper LIMIT 40""",
    ),
    "evaluated_on": (
        "Methods and the datasets their papers evaluate on (2-hop: method <- paper -> dataset)",
        """SELECT DISTINCT ?method ?dataset WHERE {
  ?p kg:proposes ?m . ?m rdfs:label ?method .
  { ?p kg:evaluatedOn ?d } UNION { ?m kg:evaluatedOn ?d }
  ?d a kg:Dataset ; rdfs:label ?dataset .
} ORDER BY ?method""",
    ),
    "outperforms": (
        "Who beats whom",
        """SELECT ?winner ?loser WHERE {
  ?a kg:outperforms ?b . ?a rdfs:label ?winner . ?b rdfs:label ?loser .
} ORDER BY ?winner""",
    ),
    "schema_coverage": (
        "How many extracted relations mapped to the schema vs stayed open (kg:relatedTo)",
        """SELECT ?property (COUNT(*) AS ?facts) WHERE {
  ?st a rdf:Statement ; rdf:predicate ?property .
} GROUP BY ?property ORDER BY DESC(?facts)""",
    ),
}


def run(g, name: str) -> tuple[str, list[str], list[tuple]]:
    title, body = QUERIES[name]
    res = g.query(PREFIXES + body)
    headers = [str(v) for v in res.vars]
    rows = [tuple(_short(x) for x in row) for row in res]
    return title, headers, rows


def run_custom(g, sparql: str) -> tuple[list[str], list[tuple]]:
    text = sparql if "PREFIX" in sparql.upper() else PREFIXES + sparql
    res = g.query(text)
    return [str(v) for v in res.vars], [tuple(_short(x) for x in row) for row in res]


def _short(x) -> str:
    s = "" if x is None else str(x)
    return s.replace("http://example.org/graphscout/", "kg:")


def to_markdown(title: str, headers: list[str], rows: list[tuple]) -> str:
    out = [f"### {title}", ""]
    if not rows:
        return "\n".join(out + ["(no results)", ""])
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "---|" * len(headers))
    for r in rows:
        out.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
    return "\n".join(out) + "\n"
