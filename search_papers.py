#!/usr/bin/env python3
"""
Generalized Local Quant Paper Search & Discovery Engine.
Quickly search any keyword (e.g. 'statistical arbitrage', 'pairs trading', 'limit order book'),
screen for high-quality mathematical & empirical papers you HAVEN'T downloaded yet,
score them deterministically offline (or with free APIs), and download PDFs politely.

Usage:
  python search_papers.py "statistical arbitrage" --limit 5 --download
  python search_papers.py --interactive
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.database import Database
from scraper.manager import ScraperManager
from analyzer.evaluator import PaperEvaluator
from pdf_processor.pdf_downloader import PDFDownloader
from graph.graph_builder import KnowledgeGraphBuilder

def format_status(is_downloaded: bool) -> str:
    if is_downloaded:
        return "[ALREADY DOWNLOADED]"
    return "[NOT DOWNLOADED - FRESH]"

def search_and_evaluate(
    query: str,
    limit: int = 5,
    min_score: int = 65,
    only_undownloaded: bool = True,
    download_pdfs: bool = False,
    backend: str = "local",
    from_scratch: bool = False,
    db_path: str = "data/papers.db",
    pdf_dir: str = "data/pdfs"
) -> List[Dict[str, Any]]:
    """
    Core function for searching, scoring, and optionally downloading papers for a keyword.
    """
    db = Database(db_path)
    if from_scratch:
        print("[Reset] --from-scratch enabled: Clearing old papers and entities from database...")
        db.reset_database(backup=True)
        print("[Reset] Database reset. Starting clean discovery.")
    scraper_mgr = ScraperManager(db)
    evaluator = PaperEvaluator(backend=backend)
    downloader = PDFDownloader(download_dir=pdf_dir)

    print("\n" + "=" * 76)
    print(f"  QUANT PAPER DISCOVERY: '{query}'")
    print(f"  Target Limit: {limit} papers | Min Score: {min_score} | Backend: {backend.upper()}")
    print(f"  Filter: {'Only un-downloaded papers' if only_undownloaded else 'All matching papers'}")
    print("=" * 76 + "\n")

    print(f"[Search] Fetching candidates from arXiv & OpenAlex for '{query}'...")
    candidates = scraper_mgr.fetch_candidates(
        search_query=query,
        limit=limit * 2,  # fetch extra to account for score screening
        only_undownloaded=only_undownloaded,
        include_all=not only_undownloaded
    )

    if not candidates:
        print(f"[Search] No new candidate papers found for '{query}'.")
        if only_undownloaded:
            print("[Search] Tip: You may have already downloaded all matching papers, or try a broader query.")
        return []

    print(f"[Evaluate] Screening and scoring {len(candidates)} candidates using {backend} engine...\n")
    scored_papers = []

    for idx, paper in enumerate(candidates, 1):
        eval_result = evaluator.evaluate(paper, query=query)
        score = eval_result.get("score", 0)
        paper.update(eval_result)
        paper["score"] = score

        # Check local download status
        is_dl = db.is_pdf_downloaded(paper["external_id"])
        paper["is_downloaded"] = is_dl

        scored_papers.append(paper)

    # Sort by score descending
    scored_papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    qualified = [p for p in scored_papers if p.get("score", 0) >= min_score][:limit]

    if not qualified:
        print(f"[Search] No papers reached the minimum score threshold of {min_score}.")
        print("[Search] Top candidate scores were:")
        for p in scored_papers[:3]:
            print(f"  - [{p.get('score')}/100] {p.get('title')}")
        return []

    # Display results
    print("-" * 76)
    for i, p in enumerate(qualified, 1):
        score = p.get("score", 0)
        tier = "ALPHA / ELITE" if score >= 85 else ("NOTABLE" if score >= 70 else "SCREENED")
        dl_status = format_status(p.get("is_downloaded", False))
        src = p.get("source", "arxiv").upper()
        cat = p.get("category", "General Quant")
        pub = p.get("published_date", "Unknown")
        cites = p.get("citations", 0)
        cite_str = f" | Citations: {cites}" if cites else ""

        print(f"\n#{i} [{score}/100 - {tier}] {dl_status}")
        print(f"    Title:    {p.get('title')}")
        print(f"    Source:   {src} ({cat}) | Published: {pub}{cite_str}")
        print(f"    Authors:  {', '.join(p.get('authors', [])[:4])}")
        print(f"    Hook:     {p.get('hook')}")
        if p.get("takeaway"):
            print(f"    Takeaway: {p.get('takeaway')}")
        if p.get("concepts"):
            print(f"    Concepts: {', '.join(p.get('concepts', [])[:5])}")
        print(f"    PDF URL:  {p.get('pdf_url', 'N/A')}")

    print("\n" + "-" * 76)

    # Download handling
    to_download = [p for p in qualified if not p.get("is_downloaded") and p.get("pdf_url")]

    if download_pdfs and to_download:
        print(f"\n[Downloader] Queueing {len(to_download)} un-downloaded papers for polite PDF retrieval...")
        for p in to_download:
            # 1. Save metadata to DB
            paper_id = db.save_paper(p)
            p["id"] = paper_id
            db.save_evaluation(paper_id, p)

            # 2. Download PDF
            pdf_url = p.get("pdf_url")
            local_path = downloader.download_pdf(pdf_url, paper_id)
            if local_path and os.path.exists(local_path):
                db.update_local_pdf_path(paper_id, local_path)
                p["local_pdf_path"] = local_path
                p["is_downloaded"] = True
                print(f"  [OK] Saved #{paper_id}: {local_path}")
            else:
                print(f"  [--] Could not download PDF for #{paper_id} (indexed in database).")

        # Update Knowledge Graph
        try:
            graph_builder = KnowledgeGraphBuilder(db)
            graph_builder.export_interactive_html()
            print("[Graph] Updated knowledge graph at data/graph.html.")
        except Exception as e:
            pass

        print(f"\n[Complete] Successfully processed and downloaded new papers for '{query}'.")

    return qualified

def main():
    parser = argparse.ArgumentParser(
        description="Search, evaluate, and download quantitative finance research papers by keyword."
    )
    parser.add_argument("query", nargs="?", default="", help="Keyword or topic to search (e.g. 'statistical arbitrage')")
    parser.add_argument("--limit", type=int, default=5, help="Number of papers to find (default: 5)")
    parser.add_argument("--min-score", type=int, default=65, help="Minimum score threshold (default: 65)")
    parser.add_argument("--download", action="store_true", help="Automatically download qualifying PDFs")
    parser.add_argument("--all", action="store_true", help="Include papers that are already downloaded")
    parser.add_argument("--from-scratch", "--reset", action="store_true", help="Reset existing database and graph to start completely from scratch")
    parser.add_argument("--backend", choices=["local", "gemini", "glm"], default="local", help="Evaluation backend (default: local)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt mode")

    args = parser.parse_args()

    query = args.query.strip()

    # Interactive prompt if no query provided
    if not query or args.interactive:
        print("\n" + "=" * 65)
        print("  LOCAL QUANT RESEARCH PAPER KEYWORD DISCOVERY")
        print("=" * 65)
        query = input("Enter search keyword or topic (e.g. 'statistical arbitrage'): ").strip()
        if not query:
            print("No query entered. Exiting.")
            sys.exit(0)

        limit_in = input("How many top papers to look for? [default: 5]: ").strip()
        limit = int(limit_in) if limit_in.isdigit() and int(limit_in) > 0 else 5

        dl_in = input("Download PDFs for qualifying un-downloaded papers? (y/n) [default: y]: ").strip().lower()
        download = dl_in != "n"
    else:
        limit = args.limit
        download = args.download

    search_and_evaluate(
        query=query,
        limit=limit,
        min_score=args.min_score,
        only_undownloaded=not args.all,
        download_pdfs=download,
        backend=args.backend,
        from_scratch=args.from_scratch
    )

if __name__ == "__main__":
    main()