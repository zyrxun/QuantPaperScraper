#!/usr/bin/env python3
"""
Reset Corpus & Graph Utility.
Safely wipes all papers, evaluations, token data, and graph entities from SQLite
after creating an automatic timestamped backup, allowing you to build your corpus
and knowledge graph completely from scratch.

Usage:
  python reset_corpus.py
  python reset_corpus.py --yes
  python reset_corpus.py --clear-pdfs
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.database import Database
from graph.graph_builder import KnowledgeGraphBuilder

def reset_corpus(clear_pdfs: bool = False, skip_confirm: bool = False):
    db_path = "data/papers.db"
    graph_path = "data/graph.html"
    
    if not os.path.exists(db_path):
        print("[Reset] No database found at data/papers.db.")
        return

    db = Database(db_path)
    stats = db.get_stats()
    total_papers = stats.get("total_papers", 0)

    print("\n" + "=" * 68)
    print("  CORPUS & KNOWLEDGE GRAPH RESET UTILITY")
    print("=" * 68)
    print(f"  Current Database:  data/papers.db")
    print(f"  Total Papers:      {total_papers}")
    print(f"  Graph Entities:    {stats.get('unique_concepts', 0)} unique concepts")
    print(f"  Tokenized PDFs:    {stats.get('tokenized_pdfs', 0)}")
    print(f"  Clear Local PDFs:  {'YES (All PDFs in data/pdfs will be deleted)' if clear_pdfs else 'NO (PDF files preserved)'}")
    print("=" * 68)

    if not skip_confirm:
        print("\n[!] WARNING: This will clear all stored papers and graph entities.")
        print("    An automatic timestamped backup will be created in data/papers.db.backup_*")
        confirm = input("\nAre you sure you want to reset and start from scratch? (y/n): ").strip().lower()
        if confirm != "y":
            print("[Reset] Operation cancelled. No changes made.")
            return

    # Perform database reset with backup
    backup_file = db.reset_database(backup=True, clear_pdfs=clear_pdfs)
    print(f"\n[Reset] Safe backup created at: {backup_file}")
    print("[Reset] Database tables (papers, evaluations, graph_entities, tokens) cleared.")

    # Remove or reset graph.html
    if os.path.exists(graph_path):
        try:
            os.remove(graph_path)
            print("[Reset] Removed old data/graph.html.")
        except Exception as e:
            print(f"[Reset] Note removing graph.html: {e}")

    # Regenerate empty/clean graph
    try:
        builder = KnowledgeGraphBuilder(db, output_html=graph_path)
        builder.export_interactive_html()
        print("[Reset] Fresh empty knowledge graph initialized at data/graph.html.")
    except Exception as e:
        pass

    print("\n" + "=" * 68)
    print("  SUCCESS: Corpus and Knowledge Graph are now clean and empty!")
    print("  You can now search and ingest papers completely from scratch:")
    print("    python search_papers.py \"statistical arbitrage\" --limit 5 --download")
    print("=" * 68 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Reset research corpus and knowledge graph to start from scratch.")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--clear-pdfs", action="store_true", help="Also delete downloaded PDF files from data/pdfs/")
    args = parser.parse_args()

    reset_corpus(clear_pdfs=args.clear_pdfs, skip_confirm=args.yes)

if __name__ == "__main__":
    main()