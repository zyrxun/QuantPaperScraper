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

    def stream_candidates(self, target_count: int = 100, search_query: str = ""):
        """
        Generator streaming candidate papers in succession across arXiv and OpenAlex.
        Automatically checks SQLite database on-the-fly and only yields fresh, uningested papers.
        """
        filters_config = self.config.get("filters", {})
        arxiv_cats = filters_config.get(
            "arxiv_categories", 
            ["q-fin.TR", "q-fin.PM", "q-fin.CP", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.GN"]
        )
        interests = filters_config.get("interests", ["Quantitative Finance", "Market Microstructure", "Statistical Arbitrage"])

        yielded = 0
        seen_ids = set()

        # Allocate target roughly: 75% arXiv, 25% OpenAlex (or 100% search query)
        arxiv_target = max(int(target_count * 0.75), 10) if not search_query else target_count
        oa_target = target_count - arxiv_target if not search_query else target_count

        # 1. Stream from arXiv in successive batches
        arxiv_stream = self.arxiv.stream_papers(
            query=search_query,
            categories=arxiv_cats if not search_query else None,
            max_total=arxiv_target * 2, # request extra to account for already-ingested papers
            batch_size=50,
            delay_seconds=3.0
        )

        for paper in arxiv_stream:
            ext_id = paper.get("external_id")
            if ext_id and ext_id not in seen_ids and not self.db.paper_exists(ext_id):
                seen_ids.add(ext_id)
                yield paper
                yielded += 1
                if yielded >= target_count:
                    return

        # 2. Stream from OpenAlex for classic, highly-cited or foundational papers
        if yielded < target_count:
            oa_query = search_query if search_query else (interests[0] if interests else "Quantitative Finance")
            oa_stream = self.openalex.stream_papers(
                query=oa_query,
                min_citations=15 if not search_query else 0,
                max_total=(target_count - yielded) * 2,
                per_page=50,
                delay_seconds=1.0
            )

            for paper in oa_stream:
                ext_id = paper.get("external_id")
                if ext_id and ext_id not in seen_ids and not self.db.paper_exists(ext_id):
                    seen_ids.add(ext_id)
                    yield paper
                    yielded += 1
                    if yielded >= target_count:
                        return
