from ..citations.parser import parse_citations
from .retrieval import BM25, TfidfIndex, tokenize

TIER_BOOST = {1: 0.15, 2: 0.05, 3: 0.0, 4: -0.1}
RECENCY_SPAN = 60


class HybridSearch:
    def __init__(self, db):
        self.db = db
        self._build()

    def _build(self):
        self.cases = self.db.all_cases()
        self.paragraphs = self.db.all_paragraphs()
        self.statutes = self.db.all_statutes()
        self.case_by_id = {c["case_id"]: c for c in self.cases}

        para_docs = []
        for p in self.paragraphs:
            c = self.case_by_id.get(p["case_id"], {})
            doc = " ".join([
                c.get("case_name") or "", c.get("short_name") or "",
                " ".join(c.get("topics", []) or []), p.get("text") or "",
            ])
            para_docs.append(doc)
        self.para_bm25 = BM25([tokenize(d) for d in para_docs])
        self.para_vec = TfidfIndex([tokenize(d) for d in para_docs])

        statute_docs = [
            f"{s['act']} {s.get('section') or ''} {s.get('title') or ''} {s.get('body') or ''}"
            for s in self.statutes
        ]
        self.stat_bm25 = BM25([tokenize(d) for d in statute_docs])
        self.stat_vec = TfidfIndex([tokenize(d) for d in statute_docs])

    def _citation_hits(self, query: str) -> list[dict]:
        hits = []
        for pc in parse_citations(query):
            if pc.kind == "case_name":
                continue
            target = None
            for c in self.cases:
                cands = [c.get("reported_citation") or "", c.get("neutral_citation") or ""] + (c.get("alt_citations") or [])
                if any(pc.canonical.lower() == str(x).lower() for x in cands if x):
                    target = c
                    break
            if target:
                hit = {"type": "case", "score": 1.0, "reason": f"citation match: {pc.canonical}"}
                hit.update(target)
                hits.append(hit)
        return hits

    def search(self, query: str, mode: str = "hybrid", top_k: int = 10) -> list[dict]:
        qt = tokenize(query)
        results: dict[str, dict] = {}

        def push(key: str, item: dict):
            if key not in results:
                results[key] = item

        if mode in ("hybrid", "keyword"):
            for rank, (i, s) in enumerate(self.para_bm25.search(qt, top_k * 3)):
                p = self.paragraphs[i]
                push(f"para:{p['paragraph_id']}", {"type": "paragraph", "paragraph": p, "scores": {"bm25": s}, "rank_bm25": rank})
            for rank, (i, s) in enumerate(self.stat_bm25.search(qt, top_k)):
                st = self.statutes[i]
                push(f"stat:{st['statute_id']}", {"type": "statute", "statute": st, "scores": {"bm25": s}, "rank_bm25": rank})

        if mode in ("hybrid", "semantic"):
            for rank, (i, s) in enumerate(self.para_vec.search(qt, top_k * 3)):
                p = self.paragraphs[i]
                key = f"para:{p['paragraph_id']}"
                entry = results.get(key) or {"type": "paragraph", "paragraph": p, "scores": {}}
                entry["scores"]["cosine"] = s
                entry["rank_cos"] = rank
                results[key] = entry
            for rank, (i, s) in enumerate(self.stat_vec.search(qt, top_k)):
                st = self.statutes[i]
                key = f"stat:{st['statute_id']}"
                entry = results.get(key) or {"type": "statute", "statute": st, "scores": {}}
                entry["scores"]["cosine"] = s
                entry["rank_cos"] = rank
                results[key] = entry

        fused = []
        for item in results.values():
            scores = item["scores"]
            rrf = 0.0
            rb, rc = item.pop("rank_bm25", None), item.pop("rank_cos", None)
            if rb is not None:
                rrf += 1.0 / (60 + rb)
            if rc is not None:
                rrf += 1.0 / (60 + rc)
            tier, year = 1, None
            if item["type"] == "paragraph":
                tier = item["paragraph"].get("source_tier") or 1
                year = item["paragraph"].get("year")
                item["case"] = self.case_by_id.get(item["paragraph"]["case_id"], {})
            elif item["type"] == "statute":
                tier = item["statute"].get("tier") or 1
            boost = TIER_BOOST.get(tier, 0)
            if year:
                boost += min(max(year - 1950, 0), RECENCY_SPAN) / RECENCY_SPAN * 0.05
            item["final_score"] = round(rrf * (1 + boost) + max(scores.values(), default=0) / 100, 6)
            fused.append(item)

        fused.sort(key=lambda x: -x["final_score"])

        if mode in ("hybrid", "citation"):
            cited = self._citation_hits(query)
            existing_case_ids = {
                item["paragraph"]["case_id"]
                for item in fused[:top_k] if item["type"] == "paragraph"
            }
            for hit in cited:
                if hit["case_id"] not in existing_case_ids:
                    hit["final_score"] = 2.0
                    fused.insert(0, hit)

        return fused[:top_k]

    def search_cases(self, query: str, top_k: int = 8) -> list[dict]:
        seen, out = set(), []
        for item in self.search(query, mode="hybrid", top_k=top_k * 3):
            cid = None
            if item["type"] == "paragraph":
                cid = item["paragraph"]["case_id"]
            elif item["type"] == "case":
                cid = item["case_id"]
            if not cid or cid in seen:
                continue
            seen.add(cid)
            case = item.get("case") or self.case_by_id.get(cid, item)
            snippet = ""
            if item["type"] == "paragraph":
                snippet = item["paragraph"]["text"][:400]
            elif item["type"] == "case":
                snippet = (case.get("holding") or "")[:400]
            out.append({
                "case_id": cid,
                "case_name": case.get("case_name"),
                "short_name": case.get("short_name"),
                "court": case.get("court"),
                "year": case.get("year"),
                "reported_citation": case.get("reported_citation"),
                "neutral_citation": case.get("neutral_citation"),
                "topics": case.get("topics", []),
                "holding": case.get("holding"),
                "snippet": snippet,
                "score": item["final_score"],
            })
            if len(out) >= top_k:
                break
        return out
