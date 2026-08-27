import asyncio
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag

from ..citations.parser import extract_citation_strings, parse_citations

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SEARCH_TIMEOUT = 5.0
_CACHE: dict[str, list[dict[str, Any]]] = {}


def _infer_court_and_year(title: str, text: str) -> tuple[str, int | None]:
    combined = f"{title} {text}"
    year_m = re.search(r"\b(19\d\d|20\d\d)\b", combined)
    year = int(year_m.group(1)) if year_m else None

    clow = combined.lower()
    if "supreme court" in clow or "sc " in clow or "scc" in clow:
        court = "Supreme Court of India"
    elif "delhi high court" in clow or "delhi hc" in clow:
        court = "Delhi High Court"
    elif "telangana high court" in clow or "telangana hc" in clow or "andhra pradesh" in clow:
        court = "Telangana High Court"
    elif "bombay high court" in clow or "bombay hc" in clow:
        court = "Bombay High Court"
    elif "calcutta high court" in clow or "calcutta hc" in clow:
        court = "Calcutta High Court"
    elif "madras high court" in clow or "madras hc" in clow:
        court = "Madras High Court"
    elif "karnataka high court" in clow or "karnataka hc" in clow:
        court = "Karnataka High Court"
    elif "allahabad high court" in clow or "allahabad hc" in clow:
        court = "Allahabad High Court"
    elif "high court" in clow:
        court = "High Court"
    elif "tribunal" in clow or "nclat" in clow or "nclt" in clow:
        court = "Tribunal"
    else:
        court = "Indian Courts"
    return court, year


def decompose_query(query: str) -> list[str]:
    subqueries: list[str] = []
    q_clean = query.strip()
    if not q_clean:
        return []

    # 1. Cleaned query without legal boilerplates (& Ors, & Anr, and Others, etc.)
    cleaned_full = re.sub(r'\b(?:&|and)\s+(?:ors|anr|others|another|etc)\b\.?', '', q_clean, flags=re.IGNORECASE).strip()
    cleaned_full = re.sub(r'\s+', ' ', cleaned_full)

    # 2. Extract Party vs Party relationships (e.g., K. Gopi v. Sub-Registrar)
    case_match = re.search(r'([A-Za-z0-9\.\s]+?)\s+(?:v\.|vs\.?|versus)\s+([A-Za-z0-9\.\s\-]+)', cleaned_full, re.IGNORECASE)
    if case_match:
        p1 = case_match.group(1).strip()
        p2 = re.sub(r'^(?:The|State of|Union of India)\s+', '', case_match.group(2).strip(), flags=re.IGNORECASE).strip()
        if p1 and p2:
            subqueries.append(f"{p1} {p2}")
            subqueries.append(f"{p1} vs {p2}")
            p1_words = [w for w in p1.split() if len(w) > 1]
            p2_words = [w for w in p2.split() if len(w) > 2 and w.lower() not in {"the", "and", "for", "rep", "state", "district", "ors", "anr"}]
            if p1_words and p2_words:
                subqueries.append(f"{' '.join(p1_words)} {' '.join(p2_words)}")

    words_all = cleaned_full.split()
    if len(words_all) <= 8:
        subqueries.append(cleaned_full)

    # 3. Extract formal citations (e.g. 2025 INSC 462, (1998) 8 SCC 1, AIR 1950 SC 124)
    parsed = parse_citations(q_clean)
    for p in parsed:
        if p.canonical:
            subqueries.append(p.canonical)

    # 4. Extract Appeal / Writ Petition / SLP numbers
    for m in re.finditer(r'((?:Civil Appeal|Criminal Appeal|W\.?P\.?|Writ Petition|S\.?L\.?P\.?|Special Leave Petition|C\.?A\.?)\s*(?:\([A-Za-z]+\))?\s*No\.?\s*\d+\s*(?:of|/)\s*\d+)', q_clean, re.IGNORECASE):
        subqueries.append(m.group(1).strip())

    # 5. Extract Section / Act provisions
    for m in re.finditer(r'((?:Section|Sec\.|Article|Art\.|Rule|Order)\s*\d+[A-Za-z0-9\(\)]*(?:\s+[A-Za-z]+){0,4}(?:\s+Act|\s+Rules|\s+Code|\s+CrPC|\s+IPC|\s+CPC|\s+BNSS|\s+BNS)?)', q_clean, re.IGNORECASE):
        subqueries.append(m.group(1).strip())

    # 6. Fallback keywords
    stop_words = {"the", "and", "for", "with", "from", "that", "this", "can", "may", "whether", "refusal", "effect", "decided", "under", "because", "ors", "anr", "others", "another"}
    key_words = [w for w in re.findall(r'[A-Za-z0-9]{3,}', cleaned_full) if w.lower() not in stop_words]
    if len(key_words) >= 3:
        subqueries.append(" ".join(key_words[:5]))

    seen = set()
    final: list[str] = []
    for sq in subqueries:
        norm = re.sub(r'\s+', ' ', sq).strip()
        if norm and norm.lower() not in seen and len(norm) > 2:
            seen.add(norm.lower())
            final.append(norm)

    return final or [cleaned_full or q_clean]


def _extract_doc_id(href: str) -> str | None:
    """Extract numeric document ID from an IndianKanoon href."""
    m = re.search(r'/(?:docfragment|doc)/(\d+)/', href)
    return m.group(1) if m else None


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


async def fetch_kanoon_page(query: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Scrape IndianKanoon search results directly using BeautifulSoup."""
    results: list[dict[str, Any]] = []
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = "https://indiankanoon.org/"
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            r = await client.get(
                "https://indiankanoon.org/search/",
                params={"formInput": query},
                headers=headers,
            )
            if r.status_code != 200:
                return results

        soup = BeautifulSoup(r.text, "html.parser")

        for block in soup.select(".result")[:top_k]:
            title_el = block.select_one(".result_title a")
            if not title_el:
                continue
            title = title_el.get_text(separator=" ", strip=True)
            href = title_el.get("href", "")

            doc_id = _extract_doc_id(str(href))
            clean_url = (
                f"https://indiankanoon.org/doc/{doc_id}/"
                if doc_id
                else f"https://indiankanoon.org{href}"
            )

            headline_el = block.select_one(".headline")
            snippet = headline_el.get_text(separator=" ", strip=True) if headline_el else ""

            source_el = block.select_one(".docsource")
            docsource = source_el.get_text(strip=True) if source_el else ""

            court, year = _infer_court_and_year(title, f"{docsource} {snippet}")
            if docsource:
                court = docsource

            cites = extract_citation_strings(f"{title} {snippet}")
            short_name = title.split(" vs ")[0].split(" v. ")[0].split(" on ")[0].strip()

            results.append({
                "type": "online",
                "source_name": "IndianKanoon Live",
                "case_name": title,
                "short_name": short_name or title,
                "court": court,
                "year": year,
                "reported_citation": cites[0] if cites else None,
                "citation": cites[0] if cites else None,
                "text": snippet or title,
                "url": clean_url,
                "citations": cites,
            })
    except Exception:
        pass
    return results


async def fetch_ddg_kanoon_page(query: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Fallback search using DuckDuckGo IndianKanoon mirror (avoids cloud IP blocks)."""
    import urllib.parse
    results: list[dict[str, Any]] = []
    clean_q = re.sub(r'["\']', '', query).strip()
    search_q = f"site:indiankanoon.org/doc/ {clean_q}"
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = "https://html.duckduckgo.com/"

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            r = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": search_q},
                headers=headers,
            )
            if r.status_code != 200:
                r = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": search_q},
                    headers=headers,
                )
            if r.status_code != 200:
                return results

        soup = BeautifulSoup(r.text, "html.parser")
        for res in soup.select(".result")[:top_k]:
            title_el = res.select_one(".result__title a")
            snippet_el = res.select_one(".result__snippet")
            url_el = res.select_one(".result__url")
            if not title_el:
                continue

            raw_title = title_el.get_text(separator=" ", strip=True)
            title = re.sub(r"\s*-\s*Indian\s*Kanoon.*$", "", raw_title, flags=re.IGNORECASE).strip()
            if not title or len(title) < 3:
                continue

            snippet = snippet_el.get_text(separator=" ", strip=True) if snippet_el else ""
            raw_href = title_el.get("href", "")
            actual_url = ""
            if "uddg=" in raw_href:
                m = re.search(r"uddg=([^&]+)", raw_href)
                if m:
                    actual_url = urllib.parse.unquote(m.group(1))
            if not actual_url and url_el:
                raw_u = url_el.get_text(strip=True)
                actual_url = "https://" + raw_u if not raw_u.startswith("http") else raw_u
            if not actual_url:
                actual_url = raw_href

            doc_id = _extract_doc_id(actual_url)
            clean_url = f"https://indiankanoon.org/doc/{doc_id}/" if doc_id else actual_url
            court, year = _infer_court_and_year(title, snippet)
            cites = extract_citation_strings(f"{title} {snippet}")
            short_name = title.split(" vs ")[0].split(" v. ")[0].split(" on ")[0].strip()

            results.append({
                "type": "online",
                "source_name": "IndianKanoon Live (Mirror)",
                "case_name": title,
                "short_name": short_name or title,
                "court": court,
                "year": year,
                "reported_citation": cites[0] if cites else None,
                "citation": cites[0] if cites else None,
                "text": snippet or title,
                "url": clean_url,
                "citations": cites,
            })
    except Exception:
        pass
    return results


async def search_online(query: str, top_k: int = 12) -> list[dict[str, Any]]:
    q_clean = query.strip()
    if not q_clean:
        return []
    if q_clean in _CACHE:
        return _CACHE[q_clean][:top_k]

    subqueries = decompose_query(q_clean)
    
    # 1. Attempt direct IndianKanoon scrape
    tasks = [fetch_kanoon_page(sq, top_k=top_k) for sq in subqueries[:3]]
    nested_res = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for item_list in nested_res:
        if not isinstance(item_list, list):
            continue
        for item in item_list:
            url = item.get("url", "")
            title_key = re.sub(r"[^a-zA-Z0-9]", "", item.get("case_name", "").lower())[:40]
            if (url and url in seen_urls) or (title_key and title_key in seen_titles):
                continue
            if url:
                seen_urls.add(url)
            if title_key:
                seen_titles.add(title_key)
            item["final_score"] = round(1.8 - (len(merged) * 0.04), 4)
            merged.append(item)

    # 2. If direct search returned 0 (e.g. Cloudflare / datacenter IP block on Render), use DDG mirror
    if not merged:
        ddg_tasks = [fetch_ddg_kanoon_page(sq, top_k=top_k) for sq in subqueries[:3]]
        ddg_nested = await asyncio.gather(*ddg_tasks, return_exceptions=True)
        for item_list in ddg_nested:
            if not isinstance(item_list, list):
                continue
            for item in item_list:
                url = item.get("url", "")
                title_key = re.sub(r"[^a-zA-Z0-9]", "", item.get("case_name", "").lower())[:40]
                if (url and url in seen_urls) or (title_key and title_key in seen_titles):
                    continue
                if url:
                    seen_urls.add(url)
                if title_key:
                    seen_titles.add(title_key)
                item["final_score"] = round(1.8 - (len(merged) * 0.04), 4)
                merged.append(item)

    if merged:
        _CACHE[q_clean] = merged
    return merged[:top_k]


async def analyze_document_and_search(text: str, llm=None, top_k: int = 15) -> dict[str, Any]:
    """Analyze a legal document with AI/heuristics, extract core issues/statutes/citations, and scrape live authorities."""
    text_clean = text.strip()
    if not text_clean:
        return {
            "document_type": "Unknown",
            "summary": "Empty document provided.",
            "legal_issues": [],
            "statutes_involved": [],
            "extracted_citations": [],
            "search_queries": [],
            "count": 0,
            "results": [],
        }

    extracted_cites = extract_citation_strings(text_clean[:25000])

    analysis = {
        "document_type": "Legal Document",
        "summary": text_clean[:220] + ("..." if len(text_clean) > 220 else ""),
        "legal_issues": [],
        "statutes_involved": [],
        "search_queries": [],
    }

    if llm and llm.available():
        system = """You are an Indian Legal Research Intelligence Agent.
Analyze the uploaded legal document (e.g. petition, deed, contract, application, FIR, order, or notice) and formulate the best search queries to retrieve binding Supreme Court, High Court, and Tribunal citations and precedents on IndianKanoon.

Respond ONLY with JSON:
{
  "document_type": "Precise document category (e.g. Gift Deed Dispute, Quashing Petition under S. 482 CrPC, Bail Application, Commercial Suit, etc.)",
  "summary": "Concise 2-3 sentence overview of the key facts, dispute, and relief sought",
  "legal_issues": ["Issue 1: concise question of law", "Issue 2: ..."],
  "statutes_involved": ["Act & Section (e.g. Section 126 Transfer of Property Act)"],
  "search_queries": [
    "precise search query 1 for Indian judgments (e.g. 'cancellation of gift deed fraud undue influence')",
    "precise search query 2 (e.g. 'Section 126 Transfer of Property Act revocation conditions')",
    "precise search query 3",
    "precise search query 4",
    "precise search query 5"
  ]
}"""
        prompt = f"Document Text:\n\n{text_clean[:25000]}"
        try:
            ai_data = await llm.json_chat(system, prompt, temperature=0.1)
            if isinstance(ai_data, dict):
                if ai_data.get("document_type"):
                    analysis["document_type"] = ai_data["document_type"]
                if ai_data.get("summary"):
                    analysis["summary"] = ai_data["summary"]
                if isinstance(ai_data.get("legal_issues"), list):
                    analysis["legal_issues"] = [str(i) for i in ai_data["legal_issues"] if i]
                if isinstance(ai_data.get("statutes_involved"), list):
                    analysis["statutes_involved"] = [str(s) for s in ai_data["statutes_involved"] if s]
                if isinstance(ai_data.get("search_queries"), list):
                    analysis["search_queries"] = [str(q).strip() for q in ai_data["search_queries"] if q]
        except Exception:
            pass

    # Deterministic fallback / supplemental queries
    if not analysis["search_queries"]:
        decomposed = decompose_query(text_clean[:4000])
        analysis["search_queries"] = decomposed[:6]

    # Include formal citations found inside the document as queries too
    combined_queries = list(analysis["search_queries"])
    for cite in extracted_cites[:4]:
        if cite not in combined_queries:
            combined_queries.append(cite)

    # Scrape live authorities for each query in parallel
    tasks = [fetch_kanoon_page(q, top_k=6) for q in combined_queries[:5] if q]
    results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for idx, item_list in enumerate(results_nested):
        if not isinstance(item_list, list):
            continue
        trigger_query = combined_queries[idx] if idx < len(combined_queries) else ""
        for item in item_list:
            url = item.get("url", "")
            title_key = re.sub(r"[^a-zA-Z0-9]", "", item.get("case_name", "").lower())[:40]
            if (url and url in seen_urls) or (title_key and title_key in seen_titles):
                continue
            if url:
                seen_urls.add(url)
            if title_key:
                seen_titles.add(title_key)
            item_copy = dict(item)
            item_copy["matched_query"] = trigger_query
            merged.append(item_copy)

    # Fallback to DDG mirror if direct scraping returned 0
    if not merged:
        ddg_tasks = [fetch_ddg_kanoon_page(q, top_k=6) for q in combined_queries[:4] if q]
        ddg_nested = await asyncio.gather(*ddg_tasks, return_exceptions=True)
        for idx, item_list in enumerate(ddg_nested):
            if not isinstance(item_list, list):
                continue
            trigger_query = combined_queries[idx] if idx < len(combined_queries) else ""
            for item in item_list:
                url = item.get("url", "")
                title_key = re.sub(r"[^a-zA-Z0-9]", "", item.get("case_name", "").lower())[:40]
                if (url and url in seen_urls) or (title_key and title_key in seen_titles):
                    continue
                if url:
                    seen_urls.add(url)
                if title_key:
                    seen_titles.add(title_key)
                item_copy = dict(item)
                item_copy["matched_query"] = trigger_query
                merged.append(item_copy)

    # Fallback to AI legal precedent engine if scrapers returned 0
    if not merged and llm and llm.available():
        import urllib.parse
        system = """You are an Indian Legal Research Intelligence Agent.
Given the legal issues and statutes in this document, return 4 to 8 authoritative Indian Supreme Court or High Court precedent judgments with formal citations and ratio decidendi.
Respond ONLY with JSON:
{
  "results": [
    {"case_name": "Full Title", "court": "Court", "year": 2023, "citation": "SCC / AIR cite", "holding": "Legal proposition", "search_query": "search query"}
  ]
}"""
        try:
            issues_summary = "\n".join([f"- {i}" for i in analysis.get("legal_issues", [])]) or analysis.get("summary", text_clean[:500])
            ai_data = await llm.json_chat(system, f"Document Summary: {analysis.get('summary')}\nLegal Issues:\n{issues_summary}", temperature=0.1)
            for idx, r in enumerate(ai_data.get("results", [])):
                sq = r.get("search_query") or r.get("case_name", "")
                k_url = f"https://indiankanoon.org/search/?formInput={urllib.parse.quote(sq)}"
                c_cite = r.get("citation")
                merged.append({
                    "type": "ai_precedent",
                    "source_name": "AI Legal Authority",
                    "case_name": r.get("case_name", "Landmark Precedent"),
                    "short_name": r.get("case_name", "").split(" v.")[0].split(" vs")[0].strip(),
                    "court": r.get("court", "Supreme Court of India"),
                    "year": r.get("year"),
                    "reported_citation": c_cite,
                    "citation": c_cite,
                    "text": r.get("holding", ""),
                    "url": k_url,
                    "matched_query": "AI Precedent Inference",
                    "citations": [c_cite] if c_cite else [],
                })
        except Exception:
            pass

    # Sort and score
    for rank, item in enumerate(merged):
        item["score"] = round(2.0 - (rank * 0.03), 3)

    return {
        "document_type": analysis["document_type"],
        "summary": analysis["summary"],
        "legal_issues": analysis["legal_issues"],
        "statutes_involved": analysis["statutes_involved"],
        "extracted_citations": extracted_cites,
        "search_queries": analysis["search_queries"],
        "count": len(merged[:top_k]),
        "results": merged[:top_k],
    }

