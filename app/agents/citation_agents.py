import json


class CitationAgent:
    name = "citation_agent"

    def __init__(self, verifier):
        self.verifier = verifier

    async def run(self, ctx: dict) -> dict:
        texts = []
        if ctx.get("query"):
            texts.append(ctx["query"])
        for doc in ctx.get("documents", []):
            texts.append(doc)
        for auth in ctx.get("authorities", []):
            if isinstance(auth, dict) and auth.get("citation"):
                texts.append(str(auth["citation"]))
        seen, verified = set(), []
        for text in texts:
            from ..citations.parser import parse_citations
            for pc in parse_citations(text):
                key = pc.canonical or pc.raw
                if key in seen or not key:
                    continue
                seen.add(key)
                verified.append(self.verifier.verify(key, proposition=None))
        return {"verified_citations": verified}


class PrecedentAgent:
    name = "precedent_agent"

    def __init__(self, db, precedent_status):
        self.db = db
        self.precedent_status = precedent_status

    async def run(self, ctx: dict) -> dict:
        authorities = ctx.get("authorities") or []
        out = []
        for a in authorities:
            case_id = a.get("case_id")
            if not case_id:
                continue
            case = self.db.case_by_id(case_id)
            if case:
                assessment = self.precedent_status.assess(case)
                out.append({
                    "case_id": case_id,
                    "case_name": case["case_name"],
                    "citation": case.get("reported_citation"),
                    "status_code": assessment["status_code"],
                    "badge": assessment["badge"],
                    "label": assessment["label"],
                    "cited_by_count": assessment["cited_by_count"],
                    "treatments": [
                        {k: t[k] for k in ("treatment", "by_case", "by_case_id", "evidence")}
                        for t in assessment["treatments"]
                    ],
                })
            elif str(case_id).startswith("online_"):
                out.append({
                    "case_id": case_id,
                    "case_name": a.get("case_name") or "Online Legal Authority",
                    "citation": a.get("citation") or a.get("reported_citation"),
                    "status_code": "ONLINE_SOURCE",
                    "badge": "🌐",
                    "label": f"Live Authority ({a.get('court') or 'Online'})",
                    "cited_by_count": 0,
                    "treatments": [],
                })
        return {"precedent_status": out}


class VerificationAgent:
    name = "verification_agent"

    def __init__(self, llm, db):
        self.llm = llm
        self.db = db

    async def run(self, ctx: dict) -> dict:
        authorities = ctx.get("authorities") or []
        known_ids = {c["case_id"] for c in self.db.all_cases()}
        online_cases = {c["case_id"]: c for c in ctx.get("case_hits", []) if str(c.get("case_id", "")).startswith("online_")}
        checks = []
        hallucinated = []
        for a in authorities:
            cid = a.get("case_id")
            if not cid:
                hallucinated.append({"claimed": a.get("case_name") or str(a)[:80], "reason": "No case_id provided"})
                continue
            if cid in known_ids:
                case = self.db.case_by_id(cid)
                checks.append({
                    "case_id": cid,
                    "case_name": case["case_name"],
                    "exists_in_database": True,
                    "is_online_source": False,
                    "citation_matches": bool(case.get("reported_citation")),
                    "claim_in_draft": a.get("why", ""),
                    "stored_holding_excerpt": (case.get("holding") or "")[:300],
                })
            elif cid in online_cases:
                oc = online_cases[cid]
                checks.append({
                    "case_id": cid,
                    "case_name": oc.get("case_name") or a.get("case_name"),
                    "exists_in_database": False,
                    "is_online_source": True,
                    "citation_matches": bool(oc.get("reported_citation") or a.get("citation")),
                    "claim_in_draft": a.get("why", ""),
                    "stored_holding_excerpt": (oc.get("snippet") or oc.get("text") or "")[:300],
                    "url": oc.get("url"),
                })
            else:
                hallucinated.append({"claimed": a.get("case_name") or str(a)[:80], "reason": "case_id not found in verified database or online retrieval"})
        summary = {
            "total_authorities_claimed": len(authorities),
            "confirmed_in_database": len([c for c in checks if not c.get("is_online_source")]),
            "confirmed_online": len([c for c in checks if c.get("is_online_source")]),
            "unverifiable_claims": hallucinated,
        }
        return {"verification": {"summary": summary, "checks": checks}}

