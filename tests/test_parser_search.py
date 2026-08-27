from app.citations.parser import extract_citation_strings, normalize_case_name, parse_citations
from app.search.retrieval import BM25, TfidfIndex, tokenize


def test_scc_paren():
    out = parse_citations("As held in (1998) 8 SCC 1 by the Supreme Court.")
    assert any(c.canonical == "(1998) 8 SCC 1" for c in out)
    assert any(c.year == 1998 and c.volume == "8" and c.page == "1" for c in out)


def test_air():
    out = parse_citations("See AIR 1950 SC 124.")
    assert any(c.canonical == "AIR 1950 SC 124" and c.report == "AIR" for c in out)


def test_scc_online():
    out = parse_citations("Vijay Madanlal, 2022 SCC OnLine SC 929.")
    assert any(c.canonical == "2022 SCC OnLine SC 929" for c in out)


def test_neutral_and_manu():
    out = parse_citations("2017 INSC 822 and MANU/SC/0123/2024")
    kinds = {c.kind for c in out}
    assert "neutral_insc" in kinds and "manu" in kinds


def test_supp_variant():
    out = parse_citations("1992 Supp (2) SCC 651")
    assert any("Supp" in c.report for c in out)


def test_case_name():
    out = parse_citations("In Whirlpool Corporation v Registrar of Trade Marks the Court held that writs remain.")
    names = [c for c in out if c.kind == "case_name"]
    assert names and "Whirlpool" in names[0].case_name


def test_extract_unique():
    text = "Cited (1998) 8 SCC 1 and again (1998) 8 SCC 1 plus AIR 1950 SC 124."
    uniq = extract_citation_strings(text)
    assert len(uniq) == 2


def test_normalize_case_name():
    n = normalize_case_name("Maneka Gandhi v. Union of India, (1978) 1 SCC 248")
    assert n["appellant"] == "Maneka Gandhi"
    assert "Union of India" in n["respondent"]
    assert n["year"] == 1978


def test_bm25_ranking():
    docs = [tokenize(d) for d in [
        "writ petition alternative remedy Article 226 high court",
        "speedy trial undertrial prisoners bail criminal appeal",
        "natural justice audi alteram partem fair procedure writ",
    ]]
    bm25 = BM25(docs)
    res = bm25.search(tokenize("alternative remedy writ"), top_k=3)
    assert res[0][0] == 0


def test_tfidf_similarity():
    idx = TfidfIndex([tokenize(t) for t in [
        "the right to privacy is intrinsic to life and liberty",
        "speedy trial is part of article 21",
    ]])
    q = idx.embed_query(tokenize("privacy life liberty"))
    assert idx.cosine(q, idx.vecs[0]) > idx.cosine(q, idx.vecs[1])
