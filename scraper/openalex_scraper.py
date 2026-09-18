"""
OpenAlex API scraper for discovering high-impact, classic, or diverse scientific papers.
Fulfills user preference: 'They don't have to be recent papers - papers that are interesting.'
"""

import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any, Optional

class OpenAlexScraper:
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, mailto: str = "researcher@example.com"):
        # OpenAlex requests a polite User-Agent with an email address for priority routing
        self.headers = {"User-Agent": f"PaperScraperBot/1.0 (mailto:{mailto})"}

    def _reconstruct_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """Reconstructs text from OpenAlex's abstract_inverted_index format."""
        if not inverted_index:
            return ""
        
        position_word_map = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word_map[pos] = word
        
        sorted_words = [position_word_map[k] for k in sorted(position_word_map.keys())]
        return " ".join(sorted_words)

    def search(self, query: str = "", min_citations: int = 25, is_open_access: bool = True, max_results: int = 15) -> List[Dict[str, Any]]:
        """
        Searches OpenAlex for fascinating papers, with optional citation thresholding.
        """
        filters = []
        if is_open_access:
            filters.append("open_access.is_oa:true")
        if min_citations > 0:
            filters.append(f"cited_by_count:>{min_citations}")

        params = {
            "per-page": min(max_results, 50),
            "sort": "relevance_score:desc" if query else "cited_by_count:desc"
        }
        
        if query.strip():
            params["search"] = query.strip()
        
        if filters:
            params["filter"] = ",".join(filters)

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self.headers)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[OpenAlexScraper] Error querying OpenAlex: {e}")
            return []

        results = []
        for work in data.get("results", []):
            work_id = work.get("id", "").split("/")[-1]
            title = work.get("title") or "Untitled Work"
            
            # Reconstruct abstract
            abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract:
                # Skip papers without abstract as GLM needs the abstract to evaluate
                continue

            # Authors
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name")
                if name:
                    authors.append(name)

            # Dates & Links
            published_date = work.get("publication_date") or str(work.get("publication_year", ""))
            doi = work.get("doi") or ""
            
            # PDF URL
            pdf_url = ""
            best_oa = work.get("best_oa_location") or {}
            if best_oa.get("pdf_url"):
                pdf_url = best_oa["pdf_url"]
            elif work.get("open_access", {}).get("oa_url"):
                pdf_url = work["open_access"]["oa_url"]

            # Concepts / Topics
            concepts = [c.get("display_name") for c in work.get("concepts", []) if c.get("display_name")]

            results.append({
                "external_id": f"openalex:{work_id}",
                "source": "openalex",
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published_date": published_date,
                "pdf_url": pdf_url,
                "doi": doi,
                "citations": work.get("cited_by_count", 0),
                "category": concepts[0] if concepts else "Science"
            })

            if len(results) >= max_results:
                break

        return results
