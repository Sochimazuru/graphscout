"""Entity resolution: decide when two extracted names are the same thing.

Rules (all deterministic, so the result is explainable):
1. Normalise case, hyphens, punctuation and simple plurals.
2. "Retrieval-Augmented Generation (RAG)" links the full name and the acronym.
3. A bare acronym ("RAG") joins a multi-word name whose initials match,
   but only when exactly one such name exists (no guessing on ambiguity).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .schema import align_type

ACRONYM_RE = re.compile(r"^(?P<full>.+?)\s*\((?P<acr>[^()]{2,15})\)\s*$")


def split_acronym(name: str) -> tuple[str, str | None]:
    m = ACRONYM_RE.match(name.strip())
    if m:
        return m.group("full").strip(), m.group("acr").strip()
    return name.strip(), None


def norm_key(name: str) -> str:
    s = name.lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    words = [w for w in s.split() if w not in {"the", "a", "an"}]
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def initials(key: str) -> str:
    return "".join(w[0] for w in key.split())


def slugify(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return s or "entity"


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class Resolver:
    def __init__(self):
        self.uf = _UnionFind()
        self.surface: dict[str, Counter] = defaultdict(Counter)  # key -> surface forms
        self.types: dict[str, Counter] = defaultdict(Counter)
        self.acronym_keys: set[str] = set()
        self._built = False

    def add(self, name: str, etype: str = "Concept") -> None:
        full, acr = split_acronym(name)
        k = norm_key(full)
        if not k:
            return
        self.uf.find(k)
        self.surface[k][full] += 1
        self.types[k][align_type(etype)] += 1
        if acr:
            ak = norm_key(acr)
            if ak:
                self.surface[ak][acr] += 1
                self.acronym_keys.add(ak)
                self.uf.union(k, ak)
        self._built = False

    def build(self) -> None:
        multiword = defaultdict(list)
        for k in list(self.surface):
            if " " in k:
                multiword[initials(k)].append(k)
        for k in list(self.surface):
            if " " not in k and 2 <= len(k) <= 8:
                candidates = multiword.get(k, [])
                if len(candidates) == 1:
                    self.uf.union(candidates[0], k)
        groups: dict[str, list[str]] = defaultdict(list)
        for k in self.surface:
            groups[self.uf.find(k)].append(k)
        self.clusters: dict[str, dict] = {}
        self.key_to_cluster: dict[str, str] = {}
        for root, keys in groups.items():
            forms = Counter()
            types = Counter()
            for k in keys:
                forms.update(self.surface[k])
                types.update(self.types[k])
            label = self._pick_label(keys, forms)
            acronyms = sorted({f for k in keys for f in self.surface[k] if k in self.acronym_keys or " " not in k} - {label})
            etype = self._pick_type(types)
            cid = slugify(label)
            # keep ids unique if two different clusters slugify the same way
            base, n = cid, 2
            while cid in self.clusters:
                cid, n = f"{base}_{n}", n + 1
            self.clusters[cid] = {
                "id": cid,
                "label": label,
                "type": etype,
                "aliases": sorted(set(forms) - {label}),
                "acronyms": acronyms,
            }
            for k in keys:
                self.key_to_cluster[k] = cid
        self._built = True

    @staticmethod
    def _pick_label(keys: list[str], forms: Counter) -> str:
        multi = [f for f in forms if " " in f.strip()]
        pool = multi or list(forms)
        return max(pool, key=lambda f: (forms[f], len(f)))

    @staticmethod
    def _pick_type(types: Counter) -> str:
        specific = Counter({t: c for t, c in types.items() if t != "Concept"})
        if specific:
            return specific.most_common(1)[0][0]
        return "Concept"

    def lookup(self, name: str) -> dict | None:
        if not self._built:
            self.build()
        full, _ = split_acronym(name)
        cid = self.key_to_cluster.get(norm_key(full))
        return self.clusters.get(cid) if cid else None
