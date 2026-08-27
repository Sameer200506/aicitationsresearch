import re
from ..citations.parser import parse_citations


class CitationVerifier:
    WEIGHTS = {"existence": 0.40, "court": 0.15, "date": 0.15, "format": 0.10, "proposition": 0.15, "status": 0.05}

    def __init__(self, db, llm=None):
        self.db = db
        self.llm = llm

    def _all_citation_variants(self, case: dict) -> list[str]:
        variants = []
        for v in [case.get("reported_citation"), case.get("neutral_citation")] + (case.get("alt_citations") or []):
            if v:
                variants.append(str(v).lower().replace(" ", "").replace(",", ""))
        return variants

    def _match_case(self, pc):
        cases = self.db.all_cases()
        for c in cases:
            if pc.canonical.lower().replace(" ", "") in self._all_citation_variants(c):
                return c, 1.0
        for c in cases:
            for variant in self._all_citation_variants(c):
                flat = variant.replace(" ", "")
                if pc.report == "SCC" and pc.year and pc.page and f"({pc.year})" in flat and flat.endswith(f"scc{pc.page}"):
                    return c, 0.6
                if pc.report == "AIR" and pc.year and pc.page and flat.startswith(f"air{pc.year}") and flat.endswith(str(pc.page)):
                    return c, 0.6
        name = (pc.case_name or "").lower()
        if name:
            best, best_score = None, 0.0
            for c in cases:
                cn = c["case_name"].lower()
                short = (c.get("short_name") or "").lower()
                for label in (cn, short):
                    if not label:
                        continue
                    words = [w for w in re.findall(r"[a-z]{4,}", name)]
                    hits = sum(1 for w in words if w in label)
                    score = hits / max(len(words), 1)
                    year_ok = not pc.year or not c.get("year") or abs(c["year"] - pc.year) <= 2
                    if score > best_score and year_ok:
                        best, best_score = c, score
            if best and best_score >= 0.6:
                return best, min(best_score, 0.85)
        return None, 0.0

    def verify(self, citation_text: str, proposition: str | None = None) -> dict:
        parsed = parse_citations(citation_text)
        formal = [p for p in parsed if p.kind != "case_name"]
        named = [p for p in parsed if p.kind == "case_name"]
        pc = formal[0] if formal else (named[0] if named else None)
        if pc is None:
            return {
                "input": citation_text, "status": "NOT_FOUND", "badge": "🔴",
                "confidence": 0, "confidence_bar": self._bar(0),
                "components": {}, "message": "No recognizable Indian legal citation format found in input.",
            }

        case, match_score = self._match_case(pc)
        components = {"existence": 0.0, "court": 0.0, "date": 0.0, "format": 0.0, "proposition": None, "status": None}
        notes = []

        if case is None:
            return {
                "input": citation_text,
                "parsed": {k: getattr(pc, k) for k in ("raw", "canonical", "kind", "year", "report", "volume", "page")},
                "status": "NOT_FOUND", "badge": "🔴",
                "confidence": round(match_score * 30),
                "confidence_bar": self._bar(round(match_score * 30)),
                "components": components,
                "matched_case": None,
                "notes": ["Citation not present in the verified legal database.",
                          "UNVERIFIED — DO NOT RELY WITHOUT MANUAL CHECK."],
                "proposition_support": None,
            }

        components["existence"] = 1.0 * match_score
        if pc.court and case.get("court"):
            components["court"] = 1.0 if pc.court.lower() in case["court"].lower() else 0.5
        elif case.get("court"):
            components["court"] = 0.75
        if pc.year and case.get("year"):
            diff = abs(pc.year - case["year"])
            components["date"] = 1.0 if diff == 0 else (0.5 if diff <= 1 else 0.0)
        elif case.get("year"):
            components["date"] = 0.75
        if pc.kind != "case_name":
            components["format"] = 1.0
        else:
            components["format"] = 0.4

        prop_result = None
        if proposition:
            prop_result = self.check_proposition(case, proposition)
            components["proposition"] = prop_result["score"]

        precedent = PrecedentStatus(self.db).assess(case)

        confidence = self._weighted(components, include_proposition=proposition is not None)
        status, badge = self._status(confidence, match_score, components, precedent)

        if proposition and prop_result and prop_result["score"] < 0.35 and status == "VERIFIED":
            status, badge = "POTENTIALLY_MISUSED", "⚠️"

        return {
            "input": citation_text,
            "parsed": {k: getattr(pc, k) for k in ("raw", "canonical", "kind", "year", "report", "volume", "page")},
            "status": status,
            "badge": badge,
            "confidence": confidence,
            "confidence_bar": self._bar(confidence),
            "components": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in components.items()},
            "matched_case": {
                "case_id": case["case_id"], "case_name": case["case_name"], "short_name": case.get("short_name"),
                "court": case.get("court"), "year": case.get("year"),
                "reported_citation": case.get("reported_citation"), "neutral_citation": case.get("neutral_citation"),
                "holding": case.get("holding"),
            } if case else None,
            "match_score": match_score,
            "proposition_support": prop_result,
            "precedent_status": precedent,
            "notes": notes,
        }

    def check_proposition(self, case: dict, proposition: str) -> dict:
        paras = self.db.paragraphs_for_case(case["case_id"])
        texts = [(p.get("paragraph_number"), p["text"]) for p in paras]
        if self.llm and self.llm.available():
            try:
                result = self.llm.judge_proposition_sync(proposition, case, texts)
                return result
            except Exception as e:
                pass
        return self._heuristic_proposition(proposition, texts)

    def _heuristic_proposition(self, proposition: str, texts: list) -> dict:
        from ..search.retrieval import TfidfIndex, tokenize
        if not texts:
            return {"score": 0.0, "method": "heuristic", "relevant_paragraphs": [], "reasoning": "No stored judgment text available."}
        idx = TfidfIndex([tokenize(t) for _, t in texts])
        qv = idx.embed_query(tokenize(proposition))
        scored = sorted(
            ((TfidfIndex.cosine(qv, vec), num, t) for vec, (num, t) in zip(idx.vecs, texts)),
            key=lambda x: -x[0],
        )
        top = [{"paragraph_number": num, "text": (t or "")[:300], "similarity": round(s, 3)}
               for s, num, t in scored[:3] if s > 0]
        best = scored[0][0] if scored else 0.0
        score = min(best * 1.6, 1.0)
        verdict = "supports" if score > 0.6 else ("partially supports" if score > 0.35 else "unclear support")
        return {
            "score": round(score, 2), "method": "lexical-overlap-heuristic",
            "verdict": verdict, "relevant_paragraphs": top,
            "reasoning": "Estimated via lexical similarity; enable an LLM key for full entailment analysis.",
        }

    def _weighted(self, components: dict, include_proposition: bool) -> int:
        total = 0.0
        weights = dict(self.WEIGHTS)
        if not include_proposition:
            weights["existence"] += weights.pop("proposition", 0)
            weights["existence"] += weights.pop("status", 0)
            components_eff = {k: v for k, v in components.items() if k not in ("proposition", "status")}
        else:
            comp = dict(components)
            if comp.get("proposition") is None:
                weights["existence"] += weights.pop("proposition")
                comp.pop("proposition")
            if comp.get("status") is None:
                weights["existence"] += weights.pop("status")
                comp.pop("status")
            components_eff = comp
        denom = sum(weights.values()) or 1.0
        for k, v in components_eff.items():
            val = v if isinstance(v, float) else 0.0
            total += weights[k] * val
        return round(100 * total / denom)

    @staticmethod
    def _bar(pct: int) -> str:
        filled = int(round(pct / 100 * 20))
        return "█" * filled + "░" * (20 - filled)

    @staticmethod
    def _status(confidence: int, match_score: float, components: dict, precedent: dict) -> tuple[str, str]:
        if match_score >= 0.99 and confidence >= 80:
            base = "VERIFIED"
        elif confidence >= 55:
            base = "PARTIALLY_VERIFIED"
        else:
            base = "NOT_FOUND"
        ps = precedent.get("status_code")
        if base != "NOT_FOUND" and ps in ("OVERRULED", "REVERSED"):
            return "VERIFIED_BUT_OVERRULED", "🔴"
        if base == "VERIFIED":
            return "VERIFIED", "🟢"
        if base == "PARTIALLY_VERIFIED":
            return "PARTIALLY_VERIFIED", "🟡"
        return base, "🔴"


class PrecedentStatus:
    MAP_IN = {
        "OVERRULES": "OVERRULED", "REVERSES": "REVERSED", "FOLLOWS": "FOLLOWED",
        "RELIES_ON": "RELIED_UPON", "DISTINGUISHES": "DISTINGUISHED", "APPROVES": "APPROVED",
        "QUESTIONS": "QUESTIONED", "NOT_FOLLOWS": "NOT_FOLLOWED",
    }

    def __init__(self, db):
        self.db = db

    def assess(self, case: dict) -> dict:
        nbrs = self.db.neighbors(case["case_id"])
        treatments = []
        for row in nbrs["incoming"]:
            code = self.MAP_IN.get(row["relationship"])
            if code:
                treatments.append({
                    "treatment": code, "by_case": row["case_name"],
                    "by_case_id": row["case_id"], "evidence": row.get("paragraph") or "",
                    "relationship_raw": row["relationship"],
                })
        codes = {t["treatment"] for t in treatments}
        cited_by = len(nbrs["incoming"])
        if codes & {"OVERRULED", "REVERSED"}:
            status_code, badge, label = "OVERRULED_OR_REVERSED", "🔴", "Reversed / Overruled"
        elif codes & {"QUESTIONED", "NOT_FOLLOWED"}:
            status_code, badge, label = "QUESTIONED", "🟡", "Questioned"
        elif codes & {"DISTINGUISHED"}:
            status_code, badge, label = "LIMITED", "🟡", "Limited / Distinguished"
        elif codes:
            status_code, badge, label = "GOOD_LAW", "🟢", "Good law"
        elif cited_by == 0:
            status_code, badge, label = "NO_TREATMENT", "⚪", "No later treatment recorded"
        else:
            status_code, badge, label = "UNCLEAR", "⚪", "Unclear"
        return {
            "status_code": status_code, "badge": badge, "label": label,
            "cited_by_count": cited_by,
            "cites_count": len(nbrs["outgoing"]),
            "treatments": treatments,
        }
