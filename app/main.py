import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents.coordinator import ResearchCoordinator, extract_text_from_upload
from .citations.parser import extract_citation_strings, parse_citations
from .citations.verifier import CitationVerifier, PrecedentStatus
from .config import settings
from .db import Database
from .llm import OpenRouterClient, get_llm
from .search.hybrid import HybridSearch
from .search.online import analyze_document_and_search, search_online


async def _keep_alive_task():
    """Periodically ping /healthz to prevent cloud providers (like Render) from sleeping after 15m."""
    # Render sets RENDER_EXTERNAL_URL automatically (e.g. https://yourapp.onrender.com)
    url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("SELF_PING_URL")
    if not url:
        return
    endpoint = f"{url.rstrip('/')}/healthz"
    # Delay initial ping by 3 minutes
    await asyncio.sleep(180)
    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.get(endpoint)
        except Exception:
            pass
        # Ping every 10 minutes (600s) - well below Render's 15m idle threshold
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_keep_alive_task())
    yield
    task.cancel()


app = FastAPI(title="AI Legal Research & Citation Intelligence", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(settings.db_path)
llm: OpenRouterClient = get_llm()
verifier = CitationVerifier(db, llm)
precedent_status = PrecedentStatus(db)
hybrid_search = HybridSearch(db)
coordinator = ResearchCoordinator(db, hybrid_search, llm)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = 10


class VerifyRequest(BaseModel):
    citation: str
    proposition: str | None = None


class ExtractRequest(BaseModel):
    text: str


class ResearchRequest(BaseModel):
    query: str
    jurisdiction: str = "India"
    court: str | None = None


class ProjectCreateRequest(BaseModel):
    name: str
    description: str | None = ""


class AddCitationRequest(BaseModel):
    case_name: str
    citation: str | None = None
    court: str | None = None
    year: int | None = None
    holding: str | None = None
    text: str | None = None
    url: str | None = None
    notes: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "mode": "live_online_search", "llm_enabled": llm.available()}


@app.get("/api/v1/models")
async def models():
    if not llm.available():
        return {"llm_enabled": False, "models": [], "note": "Set OPENROUTER_API_KEY to enable AI agents."}
    try:
        chain = await llm.discover_models()
        return {"llm_enabled": True, "models": chain[:12], "active_model": (chain or [""])[0]}
    except Exception as e:
        return {"llm_enabled": True, "models": [], "error": str(e)}


@app.get("/api/v1/search")
async def search_api(
    q: str = Query(...),
    mode: str = Query("hybrid"),
    k: int = Query(10),
):
    out = []
    try:
        online_results = await search_online(q, top_k=min(k, 15))
        for idx, r in enumerate(online_results):
            out.append({
                "type": "online",
                "source": "online",
                "source_name": r.get("source_name", "IndianKanoon Live"),
                "case_id": r.get("case_id") or f"online_{idx}_{abs(hash(r.get('url', '')))}",
                "case_name": r.get("case_name"),
                "short_name": r.get("short_name"),
                "court": r.get("court"),
                "year": r.get("year"),
                "citation": r.get("citation") or r.get("reported_citation"),
                "reported_citation": r.get("reported_citation"),
                "text": r.get("text"),
                "url": r.get("url"),
                "score": r.get("final_score", 1.8),
                "citations": r.get("citations", []),
            })
    except Exception:
        pass

    # If online scraping yields 0 results, fall back to seeded database landmark cases
    if not out:
        try:
            db_cases = hybrid_search.search_cases(q, top_k=min(k, 10))
            for c in db_cases:
                out.append({
                    "type": "database",
                    "source": "database",
                    "source_name": "Verified Legal Precedents DB",
                    "case_id": c.get("case_id"),
                    "case_name": c.get("case_name"),
                    "short_name": c.get("short_name"),
                    "court": c.get("court"),
                    "year": c.get("year"),
                    "citation": c.get("reported_citation") or c.get("citation"),
                    "reported_citation": c.get("reported_citation"),
                    "text": c.get("holding") or c.get("text"),
                    "url": c.get("source_url") or (f"https://indiankanoon.org/doc/{c.get('case_id')}/" if c.get('case_id') else None),
                    "score": c.get("final_score", 1.5),
                    "citations": [c.get("reported_citation")] if c.get("reported_citation") else [],
                })
        except Exception:
            pass

    return {"query": q, "mode": mode, "source": "online", "count": len(out[:k]), "results": out[:k]}


@app.post("/api/v1/search/document")
async def search_document_api(request: Request, k: int = Query(15)):
    ctype = request.headers.get("content-type", "").lower()
    raw_text = ""
    filename = "document.txt"

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            filename = upload.filename or filename
            data = await upload.read()
            if len(data) > 15 * 1024 * 1024:
                raise HTTPException(413, "File exceeds 15MB limit")
            try:
                raw_text = extract_text_from_upload(filename, data)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
        pasted = form.get("text")
        if not raw_text and isinstance(pasted, str):
            raw_text = pasted
    elif ctype.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body")
        raw_text = str(payload.get("text") or "")
        filename = str(payload.get("filename") or filename)
    else:
        body = (await request.body()).decode("utf-8", errors="replace")
        raw_text = body

    if not raw_text.strip():
        raise HTTPException(400, "Please upload a file (.pdf, .txt) or paste legal document text.")
    if len(raw_text.strip()) < 20:
        raise HTTPException(400, "Document text is too short to analyze.")

    try:
        result = await analyze_document_and_search(raw_text, llm=llm, top_k=min(k, 25))
        result["filename"] = filename
        return result
    except Exception as e:
        raise HTTPException(502, f"Document search failed: {e}")





@app.get("/api/v1/cases")
def list_cases():
    cases = db.all_cases()
    for c in cases:
        ps = precedent_status.assess(c)
        c["precedent_badge"] = ps["badge"]
        c["precedent_label"] = ps["label"]
        c["cited_by_count"] = ps["cited_by_count"]
    return {"count": len(cases), "cases": cases}


@app.get("/api/v1/cases/{case_id}")
def get_case(case_id: str):
    case = db.case_by_id(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    paras = db.paragraphs_for_case(case_id)
    nbrs = db.neighbors(case_id)
    ps = precedent_status.assess(case)
    ai_analysis = None
    return {
        **case,
        "paragraphs": paras,
        "cites": [{"relationship": n["relationship"], "case_id": n["case_id"],
                   "case_name": n["case_name"], "evidence": n.get("paragraph")}
                  for n in nbrs["outgoing"]],
        "cited_by": [{"relationship": n["relationship"], "case_id": n["case_id"],
                      "case_name": n["case_name"], "evidence": n.get("paragraph")}
                     for n in nbrs["incoming"]],
        "precedent_status": ps,
    }


@app.get("/api/v1/judgments/{case_id}")
def get_judgment(case_id: str):
    rows = db.query("SELECT * FROM judgments WHERE case_id=?", (case_id,))
    if not rows:
        raise HTTPException(404, "No judgment stored for this case")
    j = rows[0]
    j["paragraphs"] = db.paragraphs_for_case(case_id)
    return j


@app.get("/api/v1/statutes")
def statutes(q: str | None = None):
    all_stats = db.all_statutes()
    if q:
        matched = []
        ql = q.lower()
        for s in all_stats:
            hay = f"{s['act']} {s.get('section') or ''} {s.get('title') or ''} {s.get('body') or ''}".lower()
            terms = [t for t in ql.replace('"', ' ').split() if len(t) > 2]
            score = sum(1 for t in terms if t in hay)
            if score:
                matched.append((score, s))
        matched.sort(key=lambda x: -x[0])
        return {"count": len(matched), "statutes": [s for _, s in matched]}
    return {"count": len(all_stats), "statutes": all_stats}


@app.post("/api/v1/citations/extract")
def citations_extract(req: ExtractRequest):
    parsed = parse_citations(req.text)
    return {"count": len(parsed), "citations": [p.to_dict() for p in parsed],
            "unique_strings": extract_citation_strings(req.text)}


@app.post("/api/v1/citations/verify")
async def verify(req: VerifyRequest):
    result = verifier.verify(req.citation, req.proposition)
    return result


@app.post("/api/v1/research")
async def research(req: ResearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is required")
    try:
        return await coordinator.run_research(req.query, req.jurisdiction, req.court)
    except Exception as e:
        raise HTTPException(502, f"Research pipeline failed: {e}")


@app.post("/api/v1/petitions/analyze")
async def petition_analyze(request: Request):
    import json as _json

    ctype = request.headers.get("content-type", "").lower()
    raw_text = ""
    filename = "pasted.txt"
    mime = "text/plain"
    data = b""

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            filename = upload.filename or filename
            mime = upload.content_type or mime
            data = await upload.read()
            if len(data) > 10 * 1024 * 1024:
                raise HTTPException(413, "File exceeds 10MB limit")
            try:
                raw_text = extract_text_from_upload(filename, data)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
        pasted = form.get("text")
        if not raw_text and isinstance(pasted, str):
            raw_text = pasted
    elif ctype.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body")
        raw_text = str(payload.get("text") or "")
        filename = str(payload.get("filename") or filename)
    else:
        body = (await request.body()).decode("utf-8", errors="replace")
        raw_text = body

    if not raw_text.strip():
        raise HTTPException(400, "Provide a file upload or 'text' field")
    if len(raw_text.strip()) < 30:
        raise HTTPException(400, "Document too short to analyze")
    doc_id = db.save_document(filename, mime, data or raw_text.encode("utf-8"),
                              raw_text, None)
    try:
        result = await coordinator.analyze_petition(raw_text, filename)
    except Exception as e:
        raise HTTPException(502, f"Petition analysis failed: {e}")
    result["doc_id"] = doc_id
    db.execute("UPDATE documents SET analysis_json=? WHERE doc_id=?",
               (_json.dumps(result)[:100000], doc_id))
    return result


@app.get("/api/v1/graph/{case_id}")
def graph(case_id: str, depth: int = Query(1, ge=1, le=3)):
    case = db.case_by_id(case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(cid):
        if cid in nodes:
            return
        c = db.case_by_id(cid)
        if c:
            nodes[cid] = {"id": cid, "label": c.get("short_name") or c["case_name"],
                          "year": c.get("year"), "court": c.get("court")}

    def walk(cid, d):
        if d > depth:
            return
        nbrs = db.neighbors(cid)
        for row in nbrs["outgoing"]:
            add_node(row["case_id"])
            edges.append({"source": cid, "target": row["case_id"],
                          "relationship": row["relationship"], "direction": "out"})
            walk(row["case_id"], d + 1)
        for row in nbrs["incoming"]:
            add_node(row["case_id"])
            edges.append({"source": row["case_id"], "target": cid,
                          "relationship": row["relationship"], "direction": "in"})
            walk(row["case_id"], d + 1)

    add_node(case_id)
    walk(case_id, 1)
    return {"center": case_id, "nodes": list(nodes.values()), "edges": edges}


@app.get("/api/v1/verification/summary")
def verification_summary():
    cases = db.all_cases()
    out = []
    for c in cases:
        v = verifier.verify(c.get("reported_citation") or c["case_name"])
        out.append({
            "case_id": c["case_id"], "case_name": c["case_name"],
            "citation": v["input"], "status": v["status"], "badge": v["badge"],
            "confidence": v["confidence"],
            "precedent": precedent_status.assess(c),
        })
    buckets = {"verified": [], "partial": [], "problems": []}
    for row in out:
        if row["status"] == "VERIFIED":
            buckets["verified"].append(row)
        elif row["status"] == "PARTIALLY_VERIFIED":
            buckets["partial"].append(row)
        elif row["precedent"]["status_code"] == "OVERRULED_OR_REVERSED" or row["status"] != "VERIFIED":
            buckets["problems"].append(row)
    return {"all": out, "buckets": buckets}


@app.get("/api/v1/projects")
def list_projects_api():
    return {"projects": db.list_projects()}


@app.post("/api/v1/projects")
def create_project_api(req: ProjectCreateRequest):
    if not req.name.strip():
        raise HTTPException(400, "Project name is required")
    p = db.create_project(req.name.strip(), (req.description or "").strip())
    return p


@app.get("/api/v1/projects/{project_id}")
def get_project_api(project_id: str):
    p = db.get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@app.delete("/api/v1/projects/{project_id}")
def delete_project_api(project_id: str):
    db.delete_project(project_id)
    return {"status": "deleted", "project_id": project_id}


@app.post("/api/v1/projects/{project_id}/citations")
def add_citation_to_project_api(project_id: str, req: AddCitationRequest):
    p = db.get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    item = db.add_project_citation(project_id, req.model_dump())
    return item


@app.delete("/api/v1/projects/{project_id}/citations/{item_id}")
def delete_citation_from_project_api(project_id: str, item_id: str):
    db.delete_project_citation(item_id)
    return {"status": "deleted", "item_id": item_id}


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.head("/api/v1/ping-openrouter")
async def ping_openrouter():
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://openrouter.ai/api/v1/models")
        return {"reachable": r.status_code == 200}
