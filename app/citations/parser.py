import re
from dataclasses import dataclass, field, asdict

COURT_TOKENS = {
    "SC": "Supreme Court of India",
    "INSC": "Supreme Court of India",
}

HIGH_COURT_TOKENS = {
    "CAL": "Calcutta High Court", "MAD": "Madras High Court", "BOM": "Bombay High Court",
    "ALL": "Allahabad High Court", "KER": "Kerala High Court", "PAT": "Patna High Court",
    "RAJ": "Rajasthan High Court", "MP": "Madhya Pradesh High Court", "GUJ": "Gujarat High Court",
    "AP": "Andhra Pradesh High Court", "KAR": "Karnataka High Court", "DEL": "Delhi High Court",
    "ORI": "Orissa High Court", "GAU": "Gauhati High Court", "HP": "Himachal Pradesh High Court",
    "P&H": "Punjab & Haryana High Court", "WB": "Calcutta High Court",
}


@dataclass
class ParsedCitation:
    raw: str = ""
    start: int = -1
    end: int = -1
    kind: str = ""
    year: int | None = None
    report: str = ""
    volume: str = ""
    page: str = ""
    court_token: str = ""
    court: str = ""
    case_name: str = ""
    canonical: str = ""

    def to_dict(self):
        return asdict(self)


_PATTERNS = [
    ("scc_paren", re.compile(r"\((\d{4})\)\s*(\d{1,2})\s*SCC\s*(\d{1,4})\b")),
    ("scc_supp_year_first", re.compile(r"\b(\d{4})\s+Supp(?:\.|\s*\()?\s*\(?(\d{1,2})\)?\s*SCC\s*(\d{1,4})\b")),
    ("scc_online", re.compile(r"\b(\d{4})\s+SCC\s+OnLine\s+([A-Za-z&]{1,6})\s+(\d{1,5})\b")),
    ("air", re.compile(r"\bAIR\s+(\d{4})\s+([A-Za-z&]{1,8})\s+(\d{1,4})\b", re.IGNORECASE)),
    ("neutral_insc", re.compile(r"\b(\d{4})\s+INSC\s+(\d{1,4})\b")),
    ("manu", re.compile(r"\bMANU/([A-Za-z]{2,4})/(\d{1,4})/(\d{4})\b")),
]

_V_CASE = re.compile(
    r"([A-Z][A-Za-z0-9.&'\-\s]*?)\s+(?:vs?\.?|Vs\.?|V\.?|versus|Versus)\s+([A-Z][A-Za-z0-9.&'\-()\s]+?)(?=[,.;:\n]|\s+\(|$)"
)


def _court_for(token: str) -> tuple[str, str]:
    t = token.upper()
    if t in COURT_TOKENS:
        return COURT_TOKENS[t], t
    if t in HIGH_COURT_TOKENS:
        return HIGH_COURT_TOKENS[t], t
    return "", token


def _mk(raw, start, end, kind, year=None, report="", volume="", page="", court_token="") -> ParsedCitation:
    court, _ = _court_for(court_token)
    return ParsedCitation(raw=raw, start=start, end=end, kind=kind, year=year, report=report,
                          volume=str(volume), page=str(page), court_token=court_token, court=court)


def parse_citations(text: str) -> list[ParsedCitation]:
    found: list[ParsedCitation] = []
    taken: list[tuple[int, int]] = []

    def overlaps(s, e):
        return any(not (e <= ts or s >= te) for ts, te in taken)

    def add(p):
        if p.raw and not overlaps(p.start, p.end):
            found.append(p)
            taken.append((p.start, p.end))

    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            g = m.groups()
            if kind == "scc_paren":
                c = _mk(m.group(0), m.start(), m.end(), kind, int(g[0]), "SCC", g[1], g[2])
                c.court = "Supreme Court of India"
                c.canonical = f"({g[0]}) {int(g[1])} SCC {int(g[2])}"
            elif kind == "scc_supp_year_first":
                c = _mk(m.group(0), m.start(), m.end(), kind, int(g[0]), "SCC Supp", g[1], g[2])
                c.court = "Supreme Court of India"
                c.canonical = f"{g[0]} Supp ({int(g[1])}) SCC {int(g[2])}"
            elif kind == "scc_online":
                c = _mk(m.group(0), m.start(), m.end(), kind, int(g[0]), f"SCC OnLine {g[1].upper()}", "", g[2], g[1])
                c.canonical = f"{g[0]} SCC OnLine {g[1].upper()} {int(g[2])}"
            elif kind == "air":
                tok = g[1].upper()
                c = _mk(m.group(0), m.start(), m.end(), kind, int(g[0]), "AIR", "", g[2], tok)
                c.canonical = f"AIR {g[0]} {tok} {int(g[2])}"
            elif kind == "neutral_insc":
                c = _mk(m.group(0), m.start(), m.end(), kind, int(g[0]), "INSC", "", g[1], "INSC")
                c.court = "Supreme Court of India"
                c.canonical = f"{g[0]} INSC {int(g[1])}"
            else:
                c = _mk(m.group(0), m.start(), m.end(), kind, int(g[2]), "MANU", "", g[1], g[0])
                c.canonical = f"MANU/{g[0]}/{g[1]}/{g[2]}"
            add(c)

    for m in _V_CASE.finditer(text):
        name = re.sub(r"\s+", " ", (m.group(1).strip() + " v. " + m.group(2).strip()))
        if len(name) > 120 or len(name.split()) < 3:
            continue
        c = ParsedCitation(raw=name, start=m.start(), end=m.end(), kind="case_name")
        c.case_name = name
        add(c)

    found.sort(key=lambda p: p.start)
    return found


def extract_citation_strings(text: str) -> list[str]:
    seen, out = set(), []
    for p in parse_citations(text):
        key = p.canonical or p.raw
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


CASE_NAME_HINTS = ["union of india", "state of", "state v", "commissioner", "registrar"]


def normalize_case_name(name: str) -> dict:
    clean = re.sub(r"\s+", " ", name).strip(" .,;")
    parts = re.split(r"\s+(?:v|vs|versus)\.?\s+", clean, maxsplit=1, flags=re.IGNORECASE)
    year_m = re.search(r"\b(19|20)\d{2}\b", clean)
    return {
        "case_name": clean,
        "appellant": parts[0].strip() if parts else "",
        "respondent": parts[1].strip() if len(parts) > 1 else "",
        "year": int(year_m.group(0)) if year_m else None,
    }


if __name__ == "__main__":
    sample = """In Whirlpool Corporation v Registrar of Trade Marks, (1998) 8 SCC 1 the Court held...
    See also AIR 1950 SC 124 and 2022 SCC OnLine SC 929; Maneka Gandhi v. Union of India, (1978) 1 SCC 248.
    Also 2017 INSC 822 and MANU/SC/0123/2024 and AIR 1997 SC 3011."""
    for c in parse_citations(sample):
        print(f"{c.kind:10s} | {c.canonical or c.raw}")
