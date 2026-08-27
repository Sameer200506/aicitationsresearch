import argparse
import asyncio
import json
import sys


def cmd_research(args):
    from app.main import coordinator
    result = asyncio.run(coordinator.run_research(args.query))
    print("=" * 70)
    print(result.get("research_summary") or "(no memo)")
    print("=" * 70)
    print(f"\nSession: {result['session_id']}")
    for w in result.get("warnings", []):
        print(f"WARNING: {w}")
    return 0


def cmd_verify(args):
    from app.main import verifier
    v = verifier.verify(args.citation, args.proposition)
    print(json.dumps(v, indent=2, ensure_ascii=False))
    return 0


def cmd_search(args):
    from app.main import search
    results = search.search(args.query, mode=args.mode, top_k=args.k)
    for r in results:
        if r["type"] == "paragraph":
            c = r.get("case") or {}
            print(f"[{r['final_score']:.4f}] {c.get('short_name') or c.get('case_name')} "
                  f"({c.get('year')}) {c.get('reported_citation')}")
            print(f"          {r['paragraph']['text'][:140]}…")
        elif r["type"] == "statute":
            s = r["statute"]
            print(f"[{r['final_score']:.4f}] 📜 {s['act']} {s.get('section')} — {s.get('title')}")
        else:
            print(f"[{r['final_score']:.4f}] 📁 {r.get('case_name')} ({r.get('year')})")
    return 0


def cmd_serve(args):
    import uvicorn
    from app.config import settings
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="AI Legal Research & Citation Intelligence CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("research", help="Run the multi-agent research pipeline on a question")
    r.add_argument("query")
    r.set_defaults(fn=cmd_research)

    v = sub.add_parser("verify", help="Verify a citation (optionally against a proposition)")
    v.add_argument("citation")
    v.add_argument("--proposition", default=None)
    v.set_defaults(fn=cmd_verify)

    s = sub.add_parser("search", help="Hybrid legal search over the database")
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "keyword", "semantic", "citation"])
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(fn=cmd_search)

    srv = sub.add_parser("serve", help="Start the web server + API")
    srv.set_defaults(fn=cmd_serve)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
