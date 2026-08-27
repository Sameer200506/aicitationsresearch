# AI Legal Research & Citation Intelligence Platform (MVP)

An AI-powered Indian legal research platform implementing the core MVP loop from the PRD:

```
User question → hybrid search → top judgments → citation extraction →
citation verification → precedent status → multi-agent reasoning → research memo
```

**Principle:** the legal database is the source of truth. The AI reasons over retrieved material but can never invent a citation — every authority is checked against the database and flagged 🟢/🟡/🔴/⚠️.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then put your key in OPENROUTER_API_KEY=
.venv\Scripts\python cli.py serve        # http://127.0.0.1:8000
```

Optional (PDF petitions): `pip install PyMuPDF`

### OpenRouter (free models)

The LLM layer uses **OpenRouter with `:free` models only**. On first use it fetches
`/api/v1/models`, keeps every `:free` id, and ranks them by preference
(GLM → Nemotron-3-Super → Gemma-4 → Inkling → …), with automatic fallback across
models on rate-limits/errors. Override with `OPENROUTER_MODEL=<model-id>` in `.env`.

Without a key the platform still works: search, citation parsing/verification,
precedent status, graph, petition citation checks all run deterministically;
LLM agents degrade gracefully with warnings.

### CLI

```powershell
python cli.py research "Can a writ petition be maintained when an alternative statutory remedy exists?"
python cli.py verify "(1998) 8 SCC 1" --proposition "alternative remedy does not bar writ jurisdiction"
python cli.py search "right to privacy proportionality" -k 5
python cli.py serve
```

## Web UI (http://127.0.0.1:8000)

| Tab | Purpose |
|---|---|
| Legal Search | keyword / semantic / citation / hybrid modes |
| Research Workspace | runs the full agent pipeline, renders the memo |
| Petition Analyzer | upload PDF/TXT or paste → facts, issues, reliefs, citation verification, missing-authority flags, stronger-authority suggestions |
| Verification Center | per-citation verification cards + whole-database summary |
| Citation Graph | interactive relationship graph (click nodes) |

## Architecture

```
web/index.html ──► FastAPI (app/main.py)
                     ├─ HybridSearch   BM25 + TF-IDF cosine + citation lookup, RRF fusion,
                     │                 tier/recency reranking   (app/search/)
                     ├─ CitationEngine parser/normalizer + verifier +
                     │                 precedent-status analyzer  (app/citations/)
                     ├─ Agents         Issue · Statute · Case · Citation · Precedent ·
                     │                 CounterArgument · Verification · Writer ·
                     │                 PetitionExtractor · MissingAuthorityDetector
                     │                 orchestrated by ResearchCoordinator (app/agents/)
                     ├─ OpenRouter     free-model discovery + fallback chain (app/llm.py)
                     └─ SQLite         cases · paragraphs · edges · statutes · documents ·
                                       research sessions (app/db.py, seeded in app/seed_data.py)
```

### Hallucination protection (PRD §30)

- Every agent prompt embeds hard grounding rules.
- The Verification Agent re-checks each claimed authority against the DB; unverifiable claims are stripped from results and surfaced as warnings.
- Confidence scoring decomposes into existence/court/date/format/proposition/status (PRD §29); unverifiable citations print **UNVERIFIED — DO NOT RELY WITHOUT MANUAL CHECK**.
- Precedent status derives from recorded treatment edges (FOLLOWED/OVERRULED/DISTINGUISHED…), never guessed.

### Seed corpus

20 landmark Supreme Court cases (Kesavananda Bharati, Maneka Gandhi, Whirlpool, Puttaswamy, Navtej Johar, Anuradha Bhasin, …) with accurate citations, holdings summaries and a hand-built citation graph, plus 8 constitutional/statutory provisions. These are demo-grade summaries — connect real crawlers before production use.

## API

```
GET  /healthz                      GET  /api/v1/models
GET  /api/v1/search?q=&mode=&k=    GET  /api/v1/cases · /api/v1/cases/{id} · /api/v1/judgments/{id}
GET  /api/v1/statutes?q=           GET  /api/v1/graph/{case_id}?depth=
POST /api/v1/citations/extract     POST /api/v1/citations/verify {citation, proposition?}
POST /api/v1/research {query}      POST /api/v1/petitions/analyze (multipart file OR JSON text)
GET  /api/v1/verification/summary
```

## Tests

```powershell
.venv\Scripts\python -m pytest tests -q
```

## PRD phase coverage

| Phase | Status |
|---|---|
| 1 Legal Search Foundation | ✅ hybrid search over seeded corpus (swap TF-IDF → pgvector/BGE-M3, SQLite → Postgres/OpenSearch later) |
| 2 Citation Engine | ✅ parse/normalize/verify/graph |
| 3 AI Research | ✅ multi-agent pipeline via OpenRouter free models |
| 4 Petition Intelligence | ✅ upload → extract → verify → missing/stronger authorities |
| 5 Precedent Intelligence | 🟡 treatment-based status; deepen with full-text later-treatment detection |
| 6 Production Platform | ⬜ orgs/auth/billing/exports |

## Disclaimer

Research aid only — not legal advice. Always verify every proposition against the original judgment before filing.
