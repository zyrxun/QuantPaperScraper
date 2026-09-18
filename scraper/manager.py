"""
Scraper manager that queries multiple academic sources and filters out papers
already present in the local database.
"""

from typing import List, Dict, Any
from .arxiv_scraper import ArxivScraper
from .openalex_scraper import OpenAlexScraper
from storage.database import Database

class ScraperManager:
    def __init__(self, db: Database, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.arxiv = ArxivScraper()
        self.openalex = OpenAlexScraper()

    def fetch_candidates(self, search_query: str = "", limit: int = 15) -> List[Dict[str, Any]]:
        """
        Fetches papers from both arXiv and OpenAlex based on user search query or configured topics.
        Filters out any papers already ingested into the database.
        """
        filters_config = self.config.get("filters", {})
        arxiv_cats = filters_config.get("arxiv_categories", ["q-fin.TR", "q-fin.PM", "q-fin.CP", "q-fin.PR", "q-fin.RM", "q-fin.ST"])
        interests = filters_config.get("interests", ["Quantitative Finance", "Market Microstructure", "Statistical Arbitrage"])

        candidates = []
        seen_ids = set()

        # 1. Fetch from arXiv
        arxiv_limit = max(limit // 2, 5)
        arxiv_papers = self.arxiv.search(
            query=search_query,
            categories=arxiv_cats if not search_query else None,
            max_results=arxiv_limit
        )
        for p in arxiv_papers:
            if not self.db.paper_exists(p["external_id"]) and p["external_id"] not in seen_ids:
                seen_ids.add(p["external_id"])
                candidates.append(p)

        # 2. Fetch from OpenAlex (adds high-interest / classic / cited papers across fields)
        openalex_limit = max(limit // 2, 5)
        oa_query = search_query if search_query else (interests[0] if interests else "Quantitative Finance")
        openalex_papers = self.openalex.search(
            query=oa_query,
            min_citations=20 if not search_query else 0,
            max_results=openalex_limit
        )
        for p in openalex_papers:
            if not self.db.paper_exists(p["external_id"]) and p["external_id"] not in seen_ids:
                seen_ids.add(p["external_id"])
                candidates.append(p)

        return candidates[:limit]
