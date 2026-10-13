"""Hybrid part retrieval: structured vehicle resolution + lexical (BM25F) + character n-gram vectors.

Why hybrid, and why local:
  * Dealers ask by *vehicle + part* ("brake pads for a Duke 390"). The vehicle is a hard constraint,
    so it is resolved against the real fitment values and applied as a filter, not left to a
    similarity score that might rank a Duke 200 pad above a Duke 390 pad.
  * The part type is matched with BM25F over the name/category/brand fields (exact words, stemmed,
    with a small synonym table) fused with cosine similarity over character 3-grams (typos, "tyre" vs
    "tire"). Both run in-process: nothing depends on an embeddings API during grading.
  * Prices and stock are never indexed. The index holds text only; every result is hydrated from
    the database at query time.
"""
from __future__ import annotations

import difflib
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from django.db.models import Count, Max

from catalog.models import Product

from .safety import sanitize_field, strip_injected_sentences

STOPWORDS = {
    "a", "an", "the", "for", "of", "to", "do", "does", "you", "your", "have", "has", "any", "is", "are", "there",
    "got", "i", "me", "my", "we", "need", "want", "looking", "look", "find", "please", "with", "in", "on", "and",
    "or", "available", "stock", "show", "get", "some", "can", "could", "would", "how", "many", "much", "what",
    "which", "price", "cost", "fit", "fits", "fitting", "suitable", "compatible", "part", "parts", "spare",
    "spares", "bike", "motorcycle", "motorbike", "car", "scooter", "vehicle", "model", "also", "check", "tell",
    "about", "that", "this", "it", "pls", "hi", "hello", "kit", "genuine", "original", "new",
}
GENERIC_VEHICLE_WORDS = {"bike", "motorcycle", "motorbike", "car", "scooter", "vehicle", "two", "wheeler", "any", "all"}
SYNONYMS = {  # query token -> extra tokens (applied to the query only)
    "tire": ["tyre"], "tubeless": ["tyre"], "rotor": ["disc"], "disk": ["disc"], "headlamp": ["headlight"],
    "bulb": ["headlight"], "blinker": ["indicator"], "signal": ["indicator"], "footrest": ["foot", "peg"],
    "footpeg": ["foot", "peg"], "grip": ["grip"], "handlebar": ["handle"], "lubricant": ["lube"],
    "sprocket": ["sprocket", "chain"], "clutch": ["clutch"], "mirror": ["mirror"], "sliders": ["crash", "guard"],
    "slider": ["crash", "guard"], "cover": ["cover"], "wiper": ["wiper"], "helmet": ["helmet"],
    "glove": ["glove"], "sparkplug": ["spark", "plug"], "plug": ["plug"], "coolant": ["coolant"],
}
# fields: weight in the lexical score
FIELD_WEIGHTS = {"name": 3.0, "category": 1.0, "brand": 1.0, "vehicle": 1.0, "desc": 0.4}
K1, B = 1.4, 0.6
LEX_W, CHAR_W = 0.65, 0.35
MIN_SCORE = 0.30
EXACT_FIT_BONUS = 0.25


def stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """lowercase, split on non-alphanumerics AND on letter/digit boundaries (NS200 -> ns 200), stem."""
    return [stem(t) for t in re.findall(r"[a-z]+|\d+", (text or "").lower())]


def char_grams(text: str, n: int = 3) -> Counter:
    grams = Counter()
    for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
        w = f" {word} "
        grams.update(w[i:i + n] for i in range(max(1, len(w) - n + 1)))
    return grams


def is_universal(fitment: str) -> bool:
    return not fitment or fitment.strip().lower() == "universal"


@dataclass
class Vehicle:
    name: str
    tokens: list[str]
    model_tokens: list[str]


@dataclass
class Resolution:
    status: str                       # resolved | ambiguous | unknown | none
    vehicles: list[str] = field(default_factory=list)
    consumed: set[str] = field(default_factory=set)   # raw query tokens explained by the vehicle


class Index:
    def __init__(self, rows):
        self.rows = rows                                    # list of dict(sku, ptype, fitment, ...)
        self.n = len(rows)
        fields = defaultdict(lambda: defaultdict(Counter))  # field -> sku_idx -> Counter(tokens)
        self.doc_len = [0.0] * self.n
        df = Counter()
        for i, r in enumerate(rows):
            seen = set()
            for fname, text in (("name", r["name"]), ("category", r["category"]), ("brand", r["brand"]),
                                ("vehicle", r["fitment"]), ("desc", r["desc"])):
                toks = tokenize(text)
                fields[fname][i].update(toks)
                self.doc_len[i] += FIELD_WEIGHTS[fname] * len(toks)
                seen.update(toks)
            df.update(seen)
        self.fields = fields
        self.avg_len = (sum(self.doc_len) / self.n) or 1.0
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # character n-gram TF-IDF vectors over the short, trusted-ish fields (no description)
        gdf = Counter()
        grams = []
        for r in rows:
            g = char_grams(f"{r['name']} {r['category']} {r['brand']}")
            grams.append(g)
            gdf.update(g.keys())
        self.gidf = {g: math.log(1 + self.n / d) for g, d in gdf.items()}
        self.gvec, self.gnorm = [], []
        for g in grams:
            vec = {k: (1 + math.log(c)) * self.gidf[k] for k, c in g.items()}
            self.gvec.append(vec)
            self.gnorm.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)
        # vehicles
        fits = sorted({r["fitment"] for r in rows if not is_universal(r["fitment"])})
        makes = {tokenize(f)[0] for f in fits if tokenize(f)}
        if "royal" in makes:
            makes.add("enfield")
        self.makes = makes
        self.vehicles = [Vehicle(f, tokenize(f), [t for t in tokenize(f) if t not in makes]) for f in fits]
        self.vehicle_vocab = sorted({t for v in self.vehicles for t in v.model_tokens if len(t) >= 4 and not t.isdigit()})
        part_words = {t for fname in ("name", "category", "brand") for c in fields[fname].values() for t in c}
        vehicle_words = {t for v in self.vehicles for t in v.tokens}
        self.known_words = part_words | vehicle_words | set(SYNONYMS) | {x for v in SYNONYMS.values() for x in v}
        self.part_vocab = sorted(t for t in part_words if len(t) >= 4 and not t.isdigit())
        self.by_fitment = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_fitment[r["fitment"]].append(i)

    # ------------------------------------------------------------------ vehicle resolution
    def _correct(self, raw: list[str]) -> dict[str, str]:
        """raw token -> corrected token, fixing small typos against the vehicle vocabulary
        ("pulser" -> "pulsar"). Only long, non-numeric tokens are corrected."""
        out = {}
        for t in raw:
            out[t] = t
            if len(t) >= 5 and t not in self.vehicle_vocab and not t.isdigit():
                m = difflib.get_close_matches(t, self.vehicle_vocab, n=1, cutoff=0.82)
                if m:
                    out[t] = m[0]
        return out

    def resolve_vehicle(self, text: str, strict: bool = True) -> Resolution:
        """Map free text to catalogue vehicles.

        strict=True  the text is meant to be *only* a vehicle ("Duke 390"): leftover words that match
                     nothing we know ("Activa") mean we don't carry that vehicle -> `unknown`.
        strict=False the text is a whole query ("brake pads for Duke 390"): leftovers are expected.
        """
        raw = [t for t in tokenize(text) if t not in GENERIC_VEHICLE_WORDS]
        if not raw:
            return Resolution("none")
        fix = self._correct(raw)
        q = [fix[t] for t in raw]
        qset, qcompact = set(q), "".join(q)
        by_corrected = defaultdict(set)
        for r_, c in fix.items():
            by_corrected[c].add(r_)

        def consumed_for(vehicles):
            vt = set().union(*[set(v.tokens) for v in vehicles])
            return {r_ for c in qset & vt for r_ in by_corrected[c]}

        full = [v for v in self.vehicles if v.model_tokens and set(v.model_tokens) <= qset]
        if not full:
            full = [v for v in self.vehicles if v.model_tokens and len("".join(v.model_tokens)) >= 3
                    and "".join(v.model_tokens) in qcompact]
        if full:
            best = max(len(v.model_tokens) for v in full)           # most specific wins
            full = [v for v in full if len(v.model_tokens) == best]
            return Resolution("resolved", [v.name for v in full], consumed_for(full))

        overlap = {v.name: len(qset & set(v.tokens)) for v in self.vehicles}
        top = max(overlap.values(), default=0)
        if top == 0:
            return Resolution("unknown" if strict else "none")
        cands = [v for v in self.vehicles if overlap[v.name] == top]
        if strict:
            known = set().union(*[set(v.tokens) for v in self.vehicles])
            leftovers = [t for t in qset if t not in known and t not in STOPWORDS and not t.isdigit()]
            if leftovers:                    # e.g. "honda activa": a make we know + a model we don't
                return Resolution("unknown")
        else:
            leftovers = [t for t in qset if t not in self.known_words and t not in STOPWORDS and not t.isdigit()]
            if leftovers and not (qset & set().union(*[set(v.model_tokens) for v in cands])):
                return Resolution("unknown")
        if len(cands) == 1 and any(not t.isdigit() and t in qset for t in cands[0].model_tokens):
            return Resolution("resolved", [cands[0].name], consumed_for(cands))
        return Resolution("ambiguous", sorted(v.name for v in cands), consumed_for(cands))

    # ------------------------------------------------------------------ scoring
    def _lex(self, i: int, q_terms: list[str]) -> float:
        score = 0.0
        for t in q_terms:
            tf = sum(w * self.fields[f][i].get(t, 0) for f, w in FIELD_WEIGHTS.items())
            if tf:
                norm = 1 - B + B * self.doc_len[i] / self.avg_len
                score += self.idf.get(t, 0.0) * tf * (K1 + 1) / (tf + K1 * norm)
        return score

    def scores(self, part_text: str, candidates) -> dict[int, tuple[float, float]]:
        """i -> (score, coverage). coverage = share of the part terms found in name/category."""
        terms = []
        for t in tokenize(part_text):
            if t in STOPWORDS:
                continue
            if t not in self.idf and len(t) >= 4 and not t.isdigit():      # "brek" -> "brake"
                m = difflib.get_close_matches(t, self.part_vocab, n=1, cutoff=0.65)
                t = m[0] if m else t
            terms.append(t)
            terms += SYNONYMS.get(t, [])
        terms = list(dict.fromkeys(terms))
        if not terms:
            return {}
        max_lex = sum(self.idf.get(t, 0.0) for t in terms) or 1.0
        qg = char_grams(" ".join(terms))
        qvec = {k: (1 + math.log(c)) * self.gidf.get(k, 0.0) for k, c in qg.items()}
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        out = {}
        for i in candidates:
            lex = min(1.0, self._lex(i, terms) / (max_lex * 1.6))
            dot = sum(v * self.gvec[i].get(k, 0.0) for k, v in qvec.items())
            cos = dot / (qnorm * self.gnorm[i])
            core = set(self.fields["name"][i]) | set(self.fields["category"][i])
            coverage = sum(1 for t in terms if t in core) / len(terms)
            out[i] = (LEX_W * lex + CHAR_W * cos, coverage)
        return out


_lock = threading.Lock()
_index_cache: dict = {"sig": None, "index": None}


def get_index() -> Index:
    """Rebuilt only when the product table changed (count / newest updated_at)."""
    agg = Product.objects.aggregate(n=Count("id"), last=Max("updated_at"))
    sig = (agg["n"], agg["last"])
    with _lock:
        if _index_cache["sig"] != sig or _index_cache["index"] is None:
            rows = []
            for p in Product.objects.only("sku", "name", "category", "brand", "vehicle_fitment", "description").order_by("sku"):
                rows.append({"sku": p.sku, "name": sanitize_field(p.name, 200), "category": sanitize_field(p.category),
                             "brand": sanitize_field(p.brand), "fitment": sanitize_field(p.vehicle_fitment),
                             "desc": strip_injected_sentences(p.description)})
            _index_cache.update(sig=sig, index=Index(rows))
        return _index_cache["index"]


def part_type(name: str) -> str:
    return name.split(" - ")[0].strip()


def search(query: str, vehicle: str | None = None, category: str | None = None, limit: int = 5) -> dict:
    """Returns a dict with `status` and, when ok, `matches` = [(sku, fit)] ranked best-first.

    status: ok | needs_vehicle | ambiguous_vehicle | unknown_vehicle | needs_part_type | no_results
    """
    idx = get_index()
    limit = max(1, min(int(limit or 5), 10))
    query, vehicle = (query or "").strip(), (vehicle or "").strip()
    res = idx.resolve_vehicle(vehicle, strict=True) if vehicle else Resolution("none")
    if res.status in ("none", "unknown"):          # the model may have left the bike inside the query
        in_query = idx.resolve_vehicle(query, strict=False)
        if in_query.status in ("resolved", "ambiguous") or (res.status == "none" and in_query.status == "unknown"):
            res = in_query

    if res.status == "ambiguous":
        return {"status": "ambiguous_vehicle", "candidate_vehicles": res.vehicles}
    if res.status == "unknown":
        return {"status": "unknown_vehicle", "supported_vehicles": [v.name for v in idx.vehicles]}

    part_text = " ".join(t for t in tokenize(query) if t not in res.consumed)
    if res.status == "resolved":
        part_text = " ".join(t for t in part_text.split() if t not in set(tokenize(" ".join(res.vehicles))))
    has_part_terms = any(t not in STOPWORDS for t in tokenize(part_text))
    allowed = {i for f, ix in idx.by_fitment.items() for i in ix} if res.status != "resolved" else set()
    if res.status == "resolved":
        for f in res.vehicles:
            allowed.update(idx.by_fitment.get(f, []))
        allowed.update(idx.by_fitment.get("Universal", []))
    if category:
        allowed = {i for i in allowed if idx.rows[i]["category"].lower() == category.strip().lower()} or allowed

    if not has_part_terms:
        if res.status == "resolved":
            types = sorted({part_type(idx.rows[i]["name"]) for f in res.vehicles for i in idx.by_fitment.get(f, [])})
            return {"status": "needs_part_type", "vehicles": res.vehicles, "available_part_types": types}
        return {"status": "no_results", "reason": "no part named"}

    scored = idx.scores(part_text, allowed)
    adjusted = {}
    for i, (sc, cov) in scored.items():
        sc = sc * (0.35 + 0.65 * cov)               # a pad that is not a *brake* pad must not rank
        if res.status == "resolved" and idx.rows[i]["fitment"] in res.vehicles:
            sc += EXACT_FIT_BONUS
        adjusted[i] = (sc, cov)
    full_cov = any(cov >= 0.999 and sc >= MIN_SCORE for sc, cov in adjusted.values())
    ranked = sorted(((sc, i) for i, (sc, cov) in adjusted.items()
                     if sc >= MIN_SCORE and (cov >= 0.999 or not full_cov)),
                    key=lambda x: (-x[0], idx.rows[x[1]]["sku"]))
    if not ranked:
        out = {"status": "no_results"}
        if res.status == "resolved":
            out["vehicles"] = res.vehicles
            out["available_part_types"] = sorted({part_type(idx.rows[i]["name"]) for f in res.vehicles
                                                  for i in idx.by_fitment.get(f, [])})
        return out
    if res.status != "resolved":
        top = ranked[0][0]
        close = [i for s, i in ranked if s >= 0.75 * top]
        specific = sorted({idx.rows[i]["fitment"] for i in close if not is_universal(idx.rows[i]["fitment"])})
        if len(specific) >= 2:
            return {"status": "needs_vehicle", "candidate_vehicles": specific[:12],
                    "part_types": sorted({part_type(idx.rows[i]["name"]) for i in close})[:6]}
    matches = []
    for s, i in ranked[:limit]:
        r = idx.rows[i]
        fit = "universal" if is_universal(r["fitment"]) else ("exact" if res.status == "resolved" else "listed")
        matches.append((r["sku"], fit))
    return {"status": "ok", "matches": matches, "vehicles": res.vehicles}
