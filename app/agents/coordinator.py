from ..citations.parser import extract_citation_strings
from ..citations.verifier import CitationVerifier, PrecedentStatus
from ..db import now_iso
from ..search.hybrid import HybridSearch
from ..search.online import search_online
from .citation_agents import CitationAgent, PrecedentAgent, VerificationAgent
from .petition_agent import MissingAuthorityDetector, PetitionExtractionAgent
from .research_agents import CaseAgent, CounterArgumentAgent, IssueAgent, StatuteAgent
from .writer_agent import WriterAgent


class ResearchCoordinator:
    def __init__(self, db, search: HybridSearch, llm):
        self.db = db
        self.search = search
        self.llm = llm
        self.verifier = CitationVerifier(db, llm)
        self.precedent_status = PrecedentStatus(db)

        self.issue_agent = IssueAgent(llm)
        self.statute_agent = StatuteAgent(llm)
        self.case_agent = CaseAgent(llm)
        self.counter_agent = CounterArgumentAgent(llm)
        self.writer_agent = WriterAgent(llm)

    async def _base_ctx_async(self, query: str) -> dict:
        case_hits = []
        paragraphs = []
        statute_hits = []

        # Fetch live online legal cases and statutes from IndianKanoon
        try:
            online_hits = await search_online(query, top_k=10)
            for idx, oh in enumerate(online_hits):
                cid = oh.get("case_id") or f"online_{idx}_{abs(hash(oh.get('url', '')))}"
                oh["case_id"] = cid
                case_hits.append({
                    "case_id": cid,
                    "case_name": oh.get("case_name"),
                    "short_name": oh.get("short_name"),
                    "court": oh.get("court"),
                    "year": oh.get("year"),
                    "reported_citation": oh.get("reported_citation") or oh.get("citation"),
                    "neutral_citation": None,
                    "topics": ["online_authority"],
                    "holding": oh.get("text", "")[:400],
                    "snippet": oh.get("text", "")[:400],
                    "score": oh.get("final_score", 1.8),
                    "url": oh.get("url"),
                    "source": "online",
                })
                paragraphs.append({
                    "case_id": cid,
                    "case_name": oh.get("case_name"),
                    "citation": oh.get("reported_citation") or oh.get("citation"),
                    "year": oh.get("year"),
                    "text": oh.get("text", "")[:500],
                    "holding": oh.get("text", "")[:300],
                })
                # If the title/text refers to an Act/Section, surface it as a statute hit
                title_low = oh.get("case_name", "").lower()
                if "section" in title_low or "act" in title_low or "article" in title_low:
                    statute_hits.append({
                        "statute_id": f"stat_{idx}",
                        "act": oh.get("case_name", "")[:60],
                        "section": "",
                        "title": oh.get("case_name", "")[:80],
                        "body": oh.get("text", "")[:300],
                        "score": 1.5,
                    })
        except Exception:
            pass

        return {
            "query": query,
            "search_results": paragraphs[:12],
            "case_hits": case_hits,
            "statute_hits": statute_hits,
        }


    def _base_ctx(self, query: str) -> dict:
        import asyncio
        try:
            return asyncio.run(self._base_ctx_async(query))
        except RuntimeError:
            # Fallback if already in event loop
            results = self.search.search(query, mode="hybrid", top_k=12)
            case_hits = self.search.search_cases(query, top_k=8)
            statute_hits = [
                {"statute_id": r["statute"]["statute_id"], "act": r["statute"]["act"],
                 "section": r["statute"].get("section"), "title": r["statute"].get("title"),
                 "body": r["statute"].get("body", "")[:400], "score": r["final_score"]}
                for r in results if r["type"] == "statute"
            ]
            paragraphs = []
            for r in results:
                if r["type"] == "paragraph":
                    p = r["paragraph"]
                    c = r.get("case") or {}
                    paragraphs.append({
                        "case_id": p["case_id"], "case_name": c.get("case_name"),
                        "citation": c.get("reported_citation"), "year": c.get("year"),
                        "text": p["text"][:500], "holding": (c.get("holding") or "")[:300],
                    })
            return {
                "query": query,
                "search_results": paragraphs[:10],
                "case_hits": case_hits,
                "statute_hits": statute_hits,
            }

    @staticmethod
    async def _safe(fn, ctx, fallback):
        try:
            result = fn(ctx)
            if hasattr(result, "__await__"):
                result = await result
            return result or fallback
        except Exception as e:
            return {**fallback, "_error": str(e)}

    async def run_research(self, query: str, jurisdiction: str = "India", court: str | None = None) -> dict:
        warnings = []
        ctx = await self._base_ctx_async(query)
        ctx["jurisdiction"] = jurisdiction
        if court:
            ctx["court"] = court

        issues = statutes = authorities = contrary = None

        if self.llm.available():
            issues = await self.issue_agent.run(ctx)
            ctx["issues"] = issues

            statutes = await self._safe(self.statute_agent.run, ctx, {"statutes": []})
            ctx["statutes"] = statutes.get("statutes", [])

            authorities = await self._safe(self.case_agent.run, ctx, {"authorities": []})
            ctx["authorities"] = authorities.get("authorities", [])

            contrary = await self._safe(self.counter_agent.run, ctx, {"contrary_authorities": [], "gaps": []})
            ctx["contrary_authorities"] = contrary.get("contrary_authorities", [])
            ctx["gaps"] = (authorities.get("notes") or "") + " " + " ".join(contrary.get("gaps", []))
        else:
            warnings.append("OPENROUTER_API_KEY not set: AI agents skipped. Showing deterministic search + verification only.")
            ctx["issues"] = {"main_issue": query, "issues": [], "research_questions": []}
            ctx["authorities"] = []
            ctx["statutes"] = []
            ctx["contrary_authorities"] = []
            ctx["gaps"] = ""

        citation_agent = CitationAgent(self.verifier)
        cite_result = citation_agent.run(ctx)
        if hasattr(cite_result, "__await__"):
            cite_result = await cite_result
        verified = cite_result["verified_citations"]
        ctx["verified_citations"] = verified

        precedent_agent = PrecedentAgent(self.db, self.precedent_status)
        precedent = await precedent_agent.run(ctx)
        ctx["precedent_status"] = precedent.get("precedent_status", [])

        vagent = VerificationAgent(self.llm, self.db)
        verification = await vagent.run(ctx)
        ctx["verification"] = verification.get("verification", {})
        unverified = verification["verification"].get("unverifiable_claims", [])
        for u in unverified:
            warnings.append(f"Unverifiable authority claim removed from results: {u.get('claimed')}")

        memo_markdown = ""
        if self.llm.available():
            writer_out = await self._safe(self.writer_agent.run, ctx, {"memo_markdown": ""})
            memo_markdown = writer_out.get("memo_markdown", "")
        else:
            memo_markdown = self._fallback_memo(ctx)

        result = {
            "session_id": None,
            "created_at": now_iso(),
            "query": query,
            "issues": issues or {},
            "statutes": (statutes or {}).get("statutes", []),
            "authorities": self._enrich(ctx["authorities"]),
            "contrary_authorities": self._enrich(ctx["contrary_authorities"]),
            "verified_citations": verified,
            "precedent_status": precedent.get("precedent_status", []),
            "verification": verification.get("verification", {}),
            "research_summary": memo_markdown,
            "warnings": warnings,
            "llm_enabled": self.llm.available(),
        }
        result["session_id"] = self.db.save_research(query, result)
        return result

    def _enrich(self, items: list) -> list:
        known = {c["case_id"]: c for c in self.db.all_cases()}
        out = []
        for a in items:
            if not isinstance(a, dict):
                continue
            cid = a.get("case_id")
            if cid in known:
                c = known[cid]
                a.setdefault("case_name", c["case_name"])
                a.setdefault("citation", c.get("reported_citation"))
                a.setdefault("court", c.get("court"))
                a.setdefault("year", c.get("year"))
                ps = self.precedent_status.assess(c)
                a["precedent_badge"] = ps["badge"]
                a["precedent_label"] = ps["label"]
            elif str(cid).startswith("online_"):
                a.setdefault("precedent_badge", "🌐")
                a.setdefault("precedent_label", f"Live Online Authority ({a.get('court') or 'Court'})")
            out.append(a)
        return out


    @staticmethod
    def _fallback_memo(ctx: dict) -> str:
        lines = ["# Legal Research Memo",
                 "",
                 "> Generated without LLM (no OPENROUTER_API_KEY). Deterministic retrieval and citation verification only.",
                 "", "## Supporting Authorities"]
        hits = ctx.get("case_hits", [])[:8]
        if hits:
            for h in hits:
                badge = "🟢" if h.get("reported_citation") else "🟡"
                lines.append(f"- {badge} **{h['case_name']}** ({h.get('court')}, {h.get('year')}) — "
                             f"{h.get('reported_citation')}. {(h.get('holding') or '')[:220]}")
        else:
            lines.append("_No matching authorities found in the database._")
        lines += ["", "## Applicable Statutes"]
        stats = ctx.get("statute_hits", [])
        if stats:
            for s in stats:
                lines.append(f"- {s['act']} {s.get('section') or ''} — {(s.get('title') or '')}")
        else:
            lines.append("_No matching statutes found._")
        return "\n".join(lines)

    async def analyze_petition(self, text: str, filename: str = "petition.txt") -> dict:
        warnings = []
        extraction = {}
        if self.llm.available():
            agent = PetitionExtractionAgent(self.llm)
            try:
                extraction = await agent.run({"petition_text": text[:30000]})
            except Exception:
                warnings.append("LLM extraction returned unexpected format; using deterministic extraction only.")
        else:
            warnings.append("OPENROUTER_API_KEY not set: AI extraction skipped; deterministic citation analysis only.")

        if not extraction:
            extraction = self._deterministic_extraction(text)

        citations_in_doc = list(dict.fromkeys(
            extraction.get("existing_citations") or extract_citation_strings(text)
        ))
        verified = []
        for cit in citations_in_doc:
            verified.append(self.verifier.verify(cit))
            extraction.setdefault("existing_citations", [])
            if cit not in extraction["existing_citations"]:
                extraction["existing_citations"].append(cit)

        unsupported = [v for v in verified if v["status"] == "NOT_FOUND"]

        issue_texts = [i.get("issue", i.get("title", "")) if isinstance(i, dict) else str(i)
                       for i in (extraction.get("legal_issues") or [])]
        supporting_by_issue = {}
        for it in issue_texts:
            if not it.strip():
                continue
            supporting_by_issue[it] = self.search.search_cases(it, top_k=5)
        contrary_by_issue = {}

        stronger_suggestions = self._stronger_authority_suggestions(extraction, verified)

        missing = {"unsupported_propositions": [], "adequately_supported_issues": []}
        if self.llm.available() and issue_texts:
            detector = MissingAuthorityDetector(self.llm)
            ctx = {
                "issues": extraction.get("legal_issues"),
                "case_hits": self.search.search_cases(" ".join(issue_texts), top_k=8),
                "statute_hits": [
                    {"act": s["act"], "section": s.get("section"), "title": s.get("title")}
                    for s in self.db.all_statutes()
                ][:20],
            }
            try:
                missing = await detector.run(ctx)
            except Exception as e:
                warnings.append(f"Missing-authority detection failed: {e}")

        report = self._petition_memo(extraction, verified, missing, supporting_by_issue, stronger_suggestions)

        result = {
            "filename": filename,
            "extraction": extraction,
            "verified_citations": verified,
            "not_found_citations": [v["input"] for v in unsupported],
            "missing_authorities": missing,
            "supporting_by_issue": {k: v[:4] for k, v in supporting_by_issue.items()},
            "stronger_authority_suggestions": stronger_suggestions,
            "research_report": report,
            "warnings": warnings,
            "llm_enabled": self.llm.available(),
        }
        return result

    def _stronger_authority_suggestions(self, extraction: dict, verified: list[dict]) -> list[dict]:
        suggestions = []
        for v in verified:
            mc = v.get("matched_case") or {}
            court = (mc.get("court") or "").lower()
            year = mc.get("year") or 0
            if "high court" in court or year < 1990:
                newer = [
                    c for c in self.db.all_cases()
                    if "supreme" in (c.get("court") or "").lower()
                    and (c.get("year") or 0) >= max(year, 1990)
                    and c["case_id"] != mc.get("case_id")
                ]
                scored = []
                for c in newer[:50]:
                    overlap = len(set((mc.get("holding") or "").lower().split()) &
                                  set((c.get("holding") or "").lower().split()))
                    scored.append((overlap, c))
                scored.sort(key=lambda x: (-x[0], -(x[1].get("year") or 0)))
                for _, c in scored[:2]:
                    suggestions.append({
                        "current_citation": v["input"],
                        "current_case": mc.get("case_name"),
                        "suggested": {
                            "case_name": c["case_name"], "citation": c.get("reported_citation"),
                            "court": c.get("court"), "year": c.get("year"),
                        },
                        "reason": "Later Supreme Court authority in the database addressing related legal territory; "
                                  "verify proposition fit before relying.",
                    })
        seen, dedup = set(), []
        for s in suggestions:
            key = (s["current_citation"], s["suggested"]["case_name"])
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup[:6]

    @staticmethod
    def _deterministic_extraction(text: str) -> dict:
        first_line = text.strip().splitlines()[0][:120] if text.strip() else ""
        return {
            "document_type": "other",
            "parties": [],
            "facts": [],
            "dates_and_events": [],
            "legal_issues": [{"number": 1, "issue": first_line or "Unidentified issue", "area": "unknown"}],
            "reliefs_sought": [],
            "existing_citations": [],
            "statutory_references": [],
        }

    @staticmethod
    def _petition_memo(extraction, verified, missing, supporting_by_issue, suggestions) -> str:
        lines = ["# Petition Research Report", ""]
        lines.append(f"**Document type:** {extraction.get('document_type', 'unknown')}")
        parties = extraction.get("parties") or []
        if parties:
            lines.append("**Parties:** " + "; ".join(
                f"{p.get('name')} ({p.get('role')})" for p in parties if isinstance(p, dict)))
        lines += ["", "## Legal Issues Identified"]
        for i in extraction.get("legal_issues") or []:
            if isinstance(i, dict):
                lines.append(f"{i.get('number', '-')}. {i.get('issue')} _({i.get('area', '')})_")
        lines += ["", "## Reliefs Sought"]
        for r in extraction.get("reliefs_sought") or []:
            lines.append(f"- {r}")
        lines += ["", "## Citation Verification"]
        if verified:
            for v in verified:
                mc = v.get("matched_case") or {}
                name = mc.get("case_name", "—")
                lines.append(f"- {v['badge']} `{v['input']}` → {name} | {v['status']} "
                             f"| confidence {v['confidence']}% | precedent: "
                             f"{(v.get('precedent_status') or {}).get('label', 'n/a')}")
        else:
            lines.append("_No citations found in document._")
        lines += ["", "## Missing / Unsupported Authorities"]
        ups = (missing or {}).get("unsupported_propositions") or []
        if ups:
            for u in ups:
                lines.append(f"- ⚠️ [{u.get('severity', 'medium').upper()}] {u.get('issue')} — {u.get('why_needed')} "
                             f"_Suggested search:_ “{u.get('what_to_search')}”")
        else:
            lines.append("_None flagged._")
        lines += ["", "## Supporting Authorities per Issue"]
        for issue, cases in supporting_by_issue.items():
            lines.append(f"**Issue:** {issue}")
            for h in cases[:3]:
                lines.append(f"- {h.get('case_name')} ({h.get('reported_citation')}), {h.get('court')}, {h.get('year')}")
        if suggestions:
            lines += ["", "## Stronger Authority Suggestions"]
            for s in suggestions:
                sg = s["suggested"]
                lines.append(f"- Instead of `{s['current_citation']}`, consider **{sg['case_name']}** "
                             f"({sg['citation']}, {sg['court']}, {sg['year']}). {s['reason']}")
        lines += ["", "## Limitations",
                  "- Analysis limited to the seeded database of landmark Indian cases; connect live crawlers/statutes for full coverage.",
                  "- All propositions should be independently verified against original judgments before filing."]
        return "\n".join(lines)


def extract_text_from_upload(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            joined = "\n".join(pages).strip()
            if joined:
                return joined
            raise ValueError("No embedded text layer found (scanned PDF?). OCR required.")
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF  (or upload .txt)")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")
