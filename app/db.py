import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    slug TEXT UNIQUE,
    case_name TEXT NOT NULL,
    short_name TEXT,
    court TEXT,
    jurisdiction TEXT DEFAULT 'India',
    judges TEXT,
    decided_on TEXT,
    year INTEGER,
    neutral_citation TEXT,
    reported_citation TEXT,
    alt_citations TEXT,
    case_number TEXT,
    source_url TEXT,
    source_tier INTEGER DEFAULT 1,
    holding TEXT,
    topics TEXT
);
CREATE TABLE IF NOT EXISTS judgments (
    judgment_id TEXT PRIMARY KEY,
    case_id TEXT REFERENCES cases(case_id),
    source TEXT,
    publication_date TEXT,
    full_text TEXT,
    document_hash TEXT
);
CREATE TABLE IF NOT EXISTS paragraphs (
    paragraph_id TEXT PRIMARY KEY,
    judgment_id TEXT REFERENCES judgments(judgment_id),
    case_id TEXT REFERENCES cases(case_id),
    paragraph_number TEXT,
    kind TEXT DEFAULT 'principle',
    text TEXT NOT NULL,
    is_summary INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS citation_edges (
    edge_id TEXT PRIMARY KEY,
    source_case_id TEXT REFERENCES cases(case_id),
    target_case_id TEXT REFERENCES cases(case_id),
    relationship TEXT NOT NULL,
    paragraph TEXT,
    confidence REAL DEFAULT 0.9,
    evidence TEXT
);
CREATE TABLE IF NOT EXISTS statutes (
    statute_id TEXT PRIMARY KEY,
    act TEXT NOT NULL,
    section TEXT,
    title TEXT,
    body TEXT,
    tier INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    org TEXT DEFAULT 'default',
    filename TEXT,
    mime TEXT,
    size INTEGER,
    sha256 TEXT,
    created_at TEXT,
    extracted_text TEXT,
    analysis_json TEXT
);
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT,
    query TEXT,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS project_citations (
    item_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id) ON DELETE CASCADE,
    case_name TEXT NOT NULL,
    citation TEXT,
    court TEXT,
    year INTEGER,
    holding TEXT,
    url TEXT,
    notes TEXT,
    created_at TEXT
);
"""


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def execute(self, sql: str, params=()):
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params=()):
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params=()):
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def upsert_case(self, c: dict) -> str:
        existing = self.one("SELECT case_id FROM cases WHERE slug=?", (c["slug"],))
        if existing:
            return existing["case_id"]
        case_id = c.get("case_id") or uid("case")
        self.execute(
            """INSERT INTO cases (case_id, slug, case_name, short_name, court, jurisdiction, judges,
               decided_on, year, neutral_citation, reported_citation, alt_citations, case_number,
               source_url, source_tier, holding, topics)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id, c["slug"], c["case_name"], c.get("short_name"), c.get("court", "Supreme Court of India"),
                c.get("jurisdiction", "India"), c.get("judges"), c.get("decided_on"), c.get("year"),
                c.get("neutral_citation"), c.get("reported_citation"),
                json.dumps(c.get("alt_citations", [])), c.get("case_number"),
                c.get("source_url"), c.get("source_tier", 1), c.get("holding"),
                json.dumps(c.get("topics", [])),
            ),
        )
        return case_id

    def add_judgment_with_paragraphs(self, case_id: str, principles: list) -> None:
        jid = uid("jdg")
        self.execute(
            "INSERT INTO judgments (judgment_id, case_id, source, publication_date, full_text, document_hash) VALUES (?,?,?,?,?,?)",
            (jid, case_id, "seed", None, "\n\n".join(p["text"] for p in principles), ""),
        )
        for p in principles:
            self.execute(
                "INSERT INTO paragraphs (paragraph_id, judgment_id, case_id, paragraph_number, kind, text, is_summary) VALUES (?,?,?,?,?,?,1)",
                (uid("para"), jid, case_id, p.get("number"), p.get("kind", "principle"), p["text"]),
            )

    def add_statute(self, s: dict) -> str:
        sid = s.get("statute_id") or uid("stat")
        self.execute(
            "INSERT OR REPLACE INTO statutes (statute_id, act, section, title, body, tier) VALUES (?,?,?,?,?,?)",
            (sid, s["act"], s.get("section"), s.get("title"), s.get("body"), s.get("tier", 1)),
        )
        return sid

    def add_edge(self, source_case_id: str, target_case_id: str, relationship: str, paragraph: str = "", confidence: float = 0.9) -> str:
        row = self.one(
            "SELECT edge_id FROM citation_edges WHERE source_case_id=? AND target_case_id=? AND relationship=?",
            (source_case_id, target_case_id, relationship),
        )
        if row:
            return row["edge_id"]
        eid = uid("edge")
        self.execute(
            "INSERT INTO citation_edges (edge_id, source_case_id, target_case_id, relationship, paragraph, confidence) VALUES (?,?,?,?,?,?)",
            (eid, source_case_id, target_case_id, relationship, paragraph, confidence),
        )
        return eid

    def save_document(self, filename: str, mime: str, data: bytes, extracted_text: str, analysis: dict | None, org: str = "default") -> str:
        doc_id = uid("doc")
        self.execute(
            "INSERT INTO documents (doc_id, org, filename, mime, size, sha256, created_at, extracted_text, analysis_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, org, filename, mime, len(data), sha256_bytes(data), now_iso(), extracted_text, json.dumps(analysis or {})),
        )
        return doc_id

    def save_research(self, query: str, result: dict) -> str:
        sid = uid("rs")
        self.execute(
            "INSERT INTO research_sessions (session_id, created_at, query, result_json) VALUES (?,?,?,?)",
            (sid, now_iso(), query, json.dumps(result)),
        )
        return sid

    def all_cases(self) -> list:
        rows = self.query("SELECT * FROM cases")
        for r in rows:
            r["alt_citations"] = json.loads(r.get("alt_citations") or "[]")
            r["topics"] = json.loads(r.get("topics") or "[]")
        return rows

    def all_paragraphs(self) -> list:
        return self.query(
            """SELECT p.*, c.case_name, c.short_name, c.court, c.year, c.reported_citation, c.source_tier, c.holding, c.topics
               FROM paragraphs p JOIN cases c ON p.case_id=c.case_id"""
        )

    def all_statutes(self) -> list:
        return self.query("SELECT * FROM statutes")

    def case_by_slug(self, slug: str):
        return self.one("SELECT * FROM cases WHERE slug=?", (slug,))

    def case_by_id(self, case_id: str):
        row = self.one("SELECT * FROM cases WHERE case_id=?", (case_id,))
        if row:
            row["alt_citations"] = json.loads(row.get("alt_citations") or "[]")
            row["topics"] = json.loads(row.get("topics") or "[]")
        return row

    def paragraphs_for_case(self, case_id: str) -> list:
        return self.query("SELECT * FROM paragraphs WHERE case_id=? ORDER BY kind, paragraph_number", (case_id,))

    def neighbors(self, case_id: str) -> dict:
        outgoing = self.query(
            """SELECT e.relationship, e.paragraph, e.confidence, c.* FROM citation_edges e
               JOIN cases c ON e.target_case_id=c.case_id WHERE e.source_case_id=?""",
            (case_id,),
        )
        incoming = self.query(
            """SELECT e.relationship, e.paragraph, e.confidence, c.* FROM citation_edges e
               JOIN cases c ON e.source_case_id=c.case_id WHERE e.target_case_id=?""",
            (case_id,),
        )
        return {"outgoing": outgoing, "incoming": incoming}

    def create_project(self, name: str, description: str = "") -> dict:
        pid = uid("proj")
        ts = now_iso()
        self.execute("INSERT INTO projects (project_id, name, description, created_at) VALUES (?,?,?,?)",
                     (pid, name, description, ts))
        return {"project_id": pid, "name": name, "description": description, "created_at": ts, "citations_count": 0}

    def list_projects(self) -> list:
        rows = self.query("""
            SELECT p.*, COUNT(c.item_id) as citations_count
            FROM projects p
            LEFT JOIN project_citations c ON p.project_id = c.project_id
            GROUP BY p.project_id
            ORDER BY p.created_at DESC
        """)
        return rows

    def get_project(self, project_id: str) -> dict | None:
        p = self.one("SELECT * FROM projects WHERE project_id=?", (project_id,))
        if not p:
            return None
        cites = self.query("SELECT * FROM project_citations WHERE project_id=? ORDER BY created_at DESC", (project_id,))
        p["citations"] = cites
        p["citations_count"] = len(cites)
        return p

    def delete_project(self, project_id: str):
        self.execute("DELETE FROM project_citations WHERE project_id=?", (project_id,))
        self.execute("DELETE FROM projects WHERE project_id=?", (project_id,))

    def find_project_citation(self, project_id: str, case_name: str, url: str | None = None, citation: str | None = None) -> dict | None:
        if url:
            row = self.one("SELECT * FROM project_citations WHERE project_id=? AND url=?", (project_id, url))
            if row:
                return row
        if citation and citation.strip():
            row = self.one("SELECT * FROM project_citations WHERE project_id=? AND LOWER(citation)=LOWER(?)", (project_id, citation.strip()))
            if row:
                return row
        if case_name and case_name.strip():
            row = self.one("SELECT * FROM project_citations WHERE project_id=? AND LOWER(case_name)=LOWER(?)", (project_id, case_name.strip()))
            if row:
                return row
        return None

    def add_project_citation(self, project_id: str, data: dict) -> dict:
        case_name = data.get("case_name", "Untitled Case").strip()
        url = data.get("url")
        citation = data.get("citation")

        existing = self.find_project_citation(project_id, case_name, url=url, citation=citation)
        if existing:
            return {"already_exists": True, "item_id": existing["item_id"], "project_id": project_id, **existing}

        item_id = uid("cite")
        ts = now_iso()
        self.execute("""
            INSERT INTO project_citations (item_id, project_id, case_name, citation, court, year, holding, url, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            item_id, project_id, case_name,
            citation, data.get("court"), data.get("year"),
            data.get("holding") or data.get("text"), url,
            data.get("notes", ""), ts
        ))
        return {"already_exists": False, "item_id": item_id, "project_id": project_id, "created_at": ts, **data}

    def delete_project_citation(self, item_id: str):
        self.execute("DELETE FROM project_citations WHERE item_id=?", (item_id,))

