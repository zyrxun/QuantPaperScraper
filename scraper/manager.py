"""
Scraper manager that queries academic sources strictly for quantitative finance
and financial econometrics papers, filtering out duplicates and non-domain papers.
"""

from typing import List, Dict, Any
from .arxiv_scraper import ArxivScraper
from .openalex_scraper import OpenAlexScraper
from storage.database import Database

def is_quant_paper(paper: Dict[str, Any]) -> bool:
    """Verifies that a paper candidate is genuinely related to quantitative finance."""
    source = paper.get("source", "").lower()
    cat = (paper.get("category") or "").lower()
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()

    if source == "arxiv":
        # Must have category in q-fin.* or econ.EM
        return cat.startswith("q-fin") or cat.startswith("econ.em")

    # For OpenAlex: verify category or content mentions quantitative finance terminology
    quant_terms = [
        "finance", "financial", "trading", "trader", "stock", "equity", "portfolio",
        "volatility", "econometric", "econometrics", "market microstructure",
        "asset pricing", "arbitrage", "order book", "order flow", "option pricing",
        "liquidity", "algorithmic trading", "high-frequency", "risk parity",
        "derivatives", "stochastic volatility", "alpha", "execution"
    ]
    combined = f"{cat} {title} {abstract}"
    return any(term in combined for term in quant_terms)

class ScraperManager:
    def __init__(self, db: Database, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.arxiv = ArxivScraper()
        self.openalex = OpenAlexScraper()

    def fetch_candidates(
        self,
        search_query: str = "",
        limit: int = 15,
        only_undownloaded: bool = False,
        include_all: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetches papers from both arXiv and OpenAlex based on user search query or configured topics.
        Strictly enforces quantitative finance categories and subfields.
        
        Args:
            search_query: Target topic keyword (e.g. 'statistical arbitrage')
            limit: Total candidate limit
            only_undownloaded: If True, specifically returns papers that do NOT yet have a local PDF
            include_all: If True, includes previously ingested papers annotated with their download status
        """
        filters_config = self.config.get("filters", {})
        arxiv_cats = filters_config.get(
            "arxiv_categories",
            ["q-fin.TR", "q-fin.PM", "q-fin.CP", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.MF", "q-fin.GN", "q-fin.EC", "econ.EM"]
        )
        interests = filters_config.get("interests", ["Quantitative Finance", "Market Microstructure", "Statistical Arbitrage"])

        candidates = []
        seen_ids = set()

        # Allocate limits evenly between arXiv and OpenAlex
        req_limit = max(limit, 10)
        arxiv_limit = max(req_limit // 2, 6)
        openalex_limit = max(req_limit // 2, 6)

        # 1. Fetch from arXiv: High-precision relevance & phrase search
        arxiv_papers = self.arxiv.search(
            query=search_query,
            categories=arxiv_cats,
            max_results=arxiv_limit
        )
        for p in arxiv_papers:
            ext_id = p["external_id"]
            if ext_id in seen_ids or not is_quant_paper(p):
                continue
            seen_ids.add(ext_id)

            is_downloaded = self.db.is_pdf_downloaded(ext_id)
            p["is_downloaded"] = is_downloaded
            p["download_status"] = "downloaded" if is_downloaded else "not_downloaded"

            if only_undownloaded and is_downloaded:
                continue
            if not include_all and not only_undownloaded and self.db.paper_exists(ext_id):
                continue

            candidates.append(p)

        # 2. Fetch from OpenAlex
        oa_query = search_query if search_query else (interests[0] if interests else "Quantitative Finance")
        openalex_papers = self.openalex.search(
            query=oa_query,
            min_citations=0 if search_query else 20,
            subfields=["2003", "2002"],
            max_results=openalex_limit
        )
        for p in openalex_papers:
            ext_id = p["external_id"]
            if ext_id in seen_ids or not is_quant_paper(p):
                continue
            seen_ids.add(ext_id)

            is_downloaded = self.db.is_pdf_downloaded(ext_id)
            p["is_downloaded"] = is_downloaded
            p["download_status"] = "downloaded" if is_downloaded else "not_downloaded"

            if only_undownloaded and is_downloaded:
                continue
            if not include_all and not only_undownloaded and self.db.paper_exists(ext_id):
                continue

            candidates.append(p)

        return candidates[:limit]

    def stream_candidates(self, target_count: int = 100, search_query: str = ""):
        """
        Generator streaming candidate papers in succession across arXiv and OpenAlex.
        Strictly enforces quantitative finance categories and subfields across all batches.
        Automatically checks SQLite database on-the-fly and only yields fresh, uningested papers.
        """
        filters_config = self.config.get("filters", {})
        arxiv_cats = filters_config.get(
            "arxiv_categories", 
            ["q-fin.TR", "q-fin.PM", "q-fin.CP", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.MF", "q-fin.GN", "q-fin.EC", "econ.EM"]
        )
        interests = filters_config.get("interests", ["Quantitative Finance", "Market Microstructure", "Statistical Arbitrage"])

        yielded = 0
        seen_ids = set()

        # Allocate target roughly: 75% arXiv, 25% OpenAlex (or 100% search query)
        arxiv_target = max(int(target_count * 0.75), 10) if not search_query else target_count
        oa_target = target_count - arxiv_target if not search_query else target_count

        # 1. Stream from arXiv in successive batches with quant category constraint
        arxiv_stream = self.arxiv.stream_papers(
            query=search_query,
            categories=arxiv_cats,
            max_total=arxiv_target * 2, # request extra to account for already-ingested papers
            batch_size=50,
            delay_seconds=3.0
        )

        for paper in arxiv_stream:
            ext_id = paper.get("external_id")
            if ext_id and ext_id not in seen_ids and not self.db.paper_exists(ext_id) and is_quant_paper(paper):
                seen_ids.add(ext_id)
                yield paper
                yielded += 1
                if yielded >= target_count:
                    return

        # 2. Stream from OpenAlex for classic, highly-cited papers strictly in Finance & Econometrics
        if yielded < target_count:
            oa_query = search_query if search_query else (interests[0] if interests else "Quantitative Finance")
            oa_stream = self.openalex.stream_papers(
                query=oa_query,
                min_citations=15 if not search_query else 0,
                subfields=["2003", "2002"],
                max_total=(target_count - yielded) * 2,
                per_page=50,
                delay_seconds=1.0
            )

            for paper in oa_stream:
                ext_id = paper.get("external_id")
                if ext_id and ext_id not in seen_ids and not self.db.paper_exists(ext_id) and is_quant_paper(paper):
                    seen_ids.add(ext_id)
                    yield paper
                    yielded += 1
                    if yielded >= target_count:
                        return
