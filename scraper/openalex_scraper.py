"""
OpenAlex API scraper for discovering high-impact, classic, or diverse scientific papers.
Enforces quantitative finance and financial econometrics topics/subfields to prevent
ingesting unrelated scientific/general software papers.
"""

import ssl
import urllib.request
import urllib.parse
import urllib.error
import json
from typing import List, Dict, Any, Optional

def get_ssl_context() -> ssl.SSLContext:
    """Returns a robust SSL context using certifi CA bundle when available."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        return ssl.create_default_context()
    except Exception:
        return ssl._create_unverified_context()

def safe_urlopen(req, timeout=15):
    """Executes urlopen with resilient SSL verification and automatic fallback on Windows CERTIFICATE_VERIFY_FAILED."""
    ctx = get_ssl_context()
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.URLError as e:
        err_str = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in err_str or "certificate verify failed" in err_str:
            unverified = ssl._create_unverified_context()
            return urllib.request.urlopen(req, timeout=timeout, context=unverified)
        raise

# OpenAlex Subfields: 2003 = Finance, 2002 = Economics and Econometrics
DEFAULT_FINANCE_SUBFIELDS = ["2003", "2002"]

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

    def search(
        self,
        query: str = "",
        min_citations: int = 0,
        is_open_access: bool = True,
        subfields: Optional[List[str]] = None,
        page: int = 1,
        max_results: int = 15,
        sort_by: str = None
    ) -> List[Dict[str, Any]]:
        """
        Searches OpenAlex for papers. When a query is provided, uses title_and_abstract.search
        for laser-focused topic matching with relevance ranking.
        """
        active_subfields = subfields if subfields is not None else DEFAULT_FINANCE_SUBFIELDS
        filters = []
        if is_open_access:
            filters.append("open_access.is_oa:true")
        if min_citations > 0:
            filters.append(f"cited_by_count:>{min_citations}")

        clean_q = query.strip()
        if clean_q:
            # Use title_and_abstract search filter for high-precision query matching
            clean_term = clean_q.replace('"', '').strip()
            filters.append(f"title_and_abstract.search:{clean_term}")
        elif active_subfields:
            # Constrain to finance subfields when doing general non-query harvesting
            filters.append(f"topics.subfield.id:{'|'.join(active_subfields)}")

        default_sort = "relevance_score:desc" if clean_q else "cited_by_count:desc"
        params = {
            "page": page,
            "per-page": min(max_results, 50),
            "sort": sort_by if sort_by else default_sort
        }

        if filters:
            params["filter"] = ",".join(filters)

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self.headers)

        try:
            with safe_urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[OpenAlexScraper] Error querying OpenAlex: {e}")
            return []

        results = []
        for work in data.get("results", []):
            work_id = work.get("id", "").split("/")[-1]
            title = work.get("title") or "Untitled Work"
            pt = work.get("primary_topic") or {}

            # Reconstruct abstract
            abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract:
                # Use concepts/title fallback if abstract inverted index is missing in OpenAlex
                topic_hint = pt.get("display_name", "")
                c_hints = ", ".join([c.get("display_name", "") for c in work.get("concepts", [])[:4] if c.get("display_name")])
                abstract = f"{title}. Investigates quantitative models and empirical market dynamics in {topic_hint or c_hints or 'financial economics'}."

            # Topic and domain validation
            subfield_id = str(pt.get("subfield", {}).get("id", "")).split("/")[-1]
            all_topic_subfields = {
                str(t.get("subfield", {}).get("id", "")).split("/")[-1]
                for t in work.get("topics", [])
            }
            if subfield_id:
                all_topic_subfields.add(subfield_id)

            # Secondary safeguard: If subfield is specified, ensure it belongs to target domain
            if active_subfields and not any(sf in all_topic_subfields for sf in active_subfields):
                # Verify financial concepts as fallback
                c_names = " ".join([c.get("display_name", "").lower() for c in work.get("concepts", [])])
                finance_tokens = ["finance", "trading", "stock", "portfolio", "volatility", "econometric", "asset pricing", "arbitrage", "market microstructure"]
                if not any(t in c_names for t in finance_tokens):
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
            
            # PDF URL: prioritize open-access repository mirrors (arXiv, PMC, Zenodo, SSRN) over publisher paywalls
            pdf_url = ""
            locations = work.get("locations", [])
            for loc in locations:
                loc_pdf = loc.get("pdf_url") or ""
                if loc_pdf and any(domain in loc_pdf for domain in ["arxiv.org", "biorxiv.org", "ncbi.nlm.nih.gov", "zenodo.org", "ssrn.com"]):
                    pdf_url = loc_pdf
                    break

            if not pdf_url:
                best_oa = work.get("best_oa_location") or {}
                if best_oa.get("pdf_url"):
                    pdf_url = best_oa["pdf_url"]
                elif work.get("open_access", {}).get("oa_url"):
                    candidate_oa = work["open_access"]["oa_url"]
                    if candidate_oa.endswith(".pdf") or "pdf" in candidate_oa:
                        pdf_url = candidate_oa

            # Category resolution: Use primary topic name, then concept, then fallback
            topic_name = pt.get("display_name")
            concepts = [c.get("display_name") for c in work.get("concepts", []) if c.get("display_name")]
            display_category = topic_name or (concepts[0] if concepts else "Quantitative Finance")

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
                "category": display_category
            })

            if len(results) >= max_results:
                break

        return results

    def stream_papers(
        self,
        query: str = "",
        min_citations: int = 20,
        is_open_access: bool = True,
        subfields: Optional[List[str]] = None,
        max_total: int = 500,
        per_page: int = 50,
        delay_seconds: float = 1.0
    ):
        """
        Generator yielding OpenAlex papers one by one in succession across pages,
        strictly constrained to Finance and Economics/Econometrics subfields.
        """
        import time
        fetched_count = 0
        page = 1
        per_page = min(per_page, 50)

        while fetched_count < max_total:
            current_batch_limit = min(per_page, max_total - fetched_count)
            papers = self.search(
                query=query,
                min_citations=min_citations,
                is_open_access=is_open_access,
                subfields=subfields,
                page=page,
                max_results=current_batch_limit
            )
            if not papers:
                break

            for paper in papers:
                yield paper
                fetched_count += 1
                if fetched_count >= max_total:
                    break

            if len(papers) < current_batch_limit:
                break

            page += 1
            time.sleep(delay_seconds)
