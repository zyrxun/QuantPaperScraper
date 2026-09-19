"""
Database sanitization utility:
Scans the local papers database, removes any non-quantitative finance papers,
and regenerates the interactive knowledge graph with clean, verified quant research.
"""

import os
import sys
import sqlite3

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from storage.database import Database
from graph.graph_builder import KnowledgeGraphBuilder

QUANT_KEYWORDS = [
    "trading", "trade", "trader", "market", "finance", "financial", "asset",
    "portfolio", "volatility", "option", "pricing", "alpha", "arbitrage",
    "liquidity", "limit order", "order book", "order flow", "hedge", "hedging",
    "stock", "equity", "bond", "derivative", "greeks", "stochastic volatility",
    "econometric", "econometrics", "macroeconomic", "microstructure", "slippage",
    "execution", "bid-ask", "spread", "hft", "high-frequency", "jump diffusion",
    "black-scholes", "capm", "factor model", "sharpe", "risk parity", "forex",
    "foreign exchange", "yield curve", "interest rate", "fixed income"
]

NON_QUANT_EXCLUSION_TERMS = [
    "lung disease", "mri", "left ventricle", "cardiac", "food security",
    "agriculture", "crop", "scikit-learn", "tensorflow", "federated learning",
    "cancer", "patient", "clinical", "collision at", "s\\bar{s}", "neural machine translation"
]

def is_quant_record(source: str, category: str, title: str, abstract: str) -> bool:
    source = (source or "").lower()
    cat = (category or "").lower()
    t = (title or "").lower()
    a = (abstract or "").lower()
    combined = f"{cat} {t} {a}"

    # Check for overt non-quant exclusion phrases
    for bad in NON_QUANT_EXCLUSION_TERMS:
        if bad in combined:
            return False

    if source == "arxiv":
        # arXiv papers must have category starting with q-fin or econ.EM
        return cat.startswith("q-fin") or cat.startswith("econ.em")

    # OpenAlex papers: must have a finance/econometric topic or mention core quant terms
    if any(k in cat for k in ["finance", "trading", "stock", "portfolio", "volatility", "econometric", "market microstructure", "asset pricing", "price discovery"]):
        return True

    return any(k in combined for k in QUANT_KEYWORDS)

def clean_database(db_path: str = "data/papers.db"):
    if not os.path.exists(db_path):
        print(f"[Cleaner] Database file not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT id, external_id, source, title, category, abstract FROM papers")
    rows = cursor.fetchall()
    total_papers = len(rows)
    print(f"\n[Cleaner] 🔍 Scanning {total_papers} papers in '{db_path}'...")

    purged_ids = []
    purged_details = []
    kept_count = 0

    for r in rows:
        pid = r["id"]
        src = r["source"]
        cat = r["category"]
        title = r["title"]
        abstract = r["abstract"]

        if not is_quant_record(src, cat, title, abstract):
            purged_ids.append(pid)
            purged_details.append((r["external_id"], title[:50], cat))
        else:
            kept_count += 1

    if purged_ids:
        print(f"[Cleaner] Found {len(purged_ids)} non-quant papers to remove:\n")
        for ext_id, title, cat in purged_details[:15]:
            print(f"  ❌ [{cat or 'Unknown'}] {title}... ({ext_id})")
        if len(purged_details) > 15:
            print(f"  ...and {len(purged_details) - 15} more.")

        # Batch delete
        id_placeholders = ",".join(["?"] * len(purged_ids))
        cursor.execute(f"DELETE FROM evaluations WHERE paper_id IN ({id_placeholders})", purged_ids)
        cursor.execute(f"DELETE FROM pdf_tokens WHERE paper_id IN ({id_placeholders})", purged_ids)
        cursor.execute(f"DELETE FROM graph_entities WHERE paper_id IN ({id_placeholders})", purged_ids)
        cursor.execute(f"DELETE FROM discord_history WHERE paper_id IN ({id_placeholders})", purged_ids)
        cursor.execute(f"DELETE FROM papers WHERE id IN ({id_placeholders})", purged_ids)
        conn.commit()
        print(f"\n[Cleaner] 🧹 Successfully purged {len(purged_ids)} papers from '{db_path}'.")
        print(f"[Cleaner] ✅ Verified Quantitative Finance papers remaining: {kept_count}")
    else:
        print(f"[Cleaner] ✅ Clean! All {total_papers} papers are verified Quantitative Finance.")

    conn.close()

    # Regenerate graph
    try:
        db = Database(db_path)
        graph_builder = KnowledgeGraphBuilder(db, output_html="data/graph.html")
        graph_builder.export_interactive_html()
        print("[Cleaner] 🌐 Rebuilt Knowledge Graph at 'data/graph.html' with verified quant clusters.")
    except Exception as e:
        print(f"[Cleaner] Note on graph export: {e}")

if __name__ == "__main__":
    target_db = sys.argv[1] if len(sys.argv) > 1 else "data/papers.db"
    clean_database(target_db)
