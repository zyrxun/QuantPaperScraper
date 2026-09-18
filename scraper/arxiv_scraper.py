"""
ArXiv API scraper for fetching research papers and preprints.
"""

import ssl
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Any

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

class ArxivScraper:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, user_agent: str = "PaperScraperBot/1.0 (academic-research)"):
        self.user_agent = user_agent

    def search(self, query: str = "", categories: List[str] = None, start: int = 0, max_results: int = 15, sort_by: str = "submittedDate") -> List[Dict[str, Any]]:
        """
        Searches arXiv for papers matching query or categories.
        Supports pagination offset via start parameter.
        sort_by options: 'submittedDate', 'lastUpdatedDate', 'relevance'
        """
        # Build query string
        query_parts = []
        if query.strip():
            query_parts.append(f"all:{query.strip()}")
        
        if categories:
            cat_query = " OR ".join([f"cat:{c.strip()}" for c in categories])
            query_parts.append(f"({cat_query})")
        
        search_query = " AND ".join(query_parts) if query_parts else "(cat:q-fin.TR OR cat:q-fin.PM OR cat:q-fin.CP OR cat:q-fin.PR OR cat:q-fin.RM OR cat:q-fin.ST)"

        params = {
            "search_query": search_query,
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": sort_by,
            "sortOrder": "descending"
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        
        try:
            with safe_urlopen(req, timeout=15) as resp:
                xml_data = resp.read().decode("utf-8")
        except Exception as e:
            print(f"[ArxivScraper] Error querying arXiv: {e}")
            return []

        return self._parse_feed(xml_data)

    def stream_papers(self, query: str = "", categories: List[str] = None, max_total: int = 500, batch_size: int = 50, delay_seconds: float = 3.0):
        """
        Generator yielding papers one by one in succession.
        Paginates through start=0, 50, 100... with polite delays between requests to adhere to arXiv's 3-second rate limit.
        """
        fetched_count = 0
        start = 0
        batch_size = min(batch_size, 100)

        while fetched_count < max_total:
            current_batch_limit = min(batch_size, max_total - fetched_count)
            papers = self.search(
                query=query,
                categories=categories,
                start=start,
                max_results=current_batch_limit
            )
            if not papers:
                break

            for paper in papers:
                yield paper
                fetched_count += 1
                if fetched_count >= max_total:
                    break

            start += len(papers)
            if len(papers) < current_batch_limit:
                break

            # Polite rate limiting for arXiv
            time.sleep(delay_seconds)

    def _parse_feed(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parses the Atom XML feed returned by arXiv."""
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"
        }

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"[ArxivScraper] XML Parse Error: {e}")
            return []

        papers = []
        for entry in root.findall("atom:entry", namespaces):
            # ID
            raw_id = entry.find("atom:id", namespaces).text.strip()
            # Extract clean arxiv identifier, e.g. 2401.12345
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
            
            # Title
            title_el = entry.find("atom:title", namespaces)
            title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else "Untitled"

            # Abstract / Summary
            summary_el = entry.find("atom:summary", namespaces)
            abstract = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""

            # Published Date
            published_el = entry.find("atom:published", namespaces)
            published_date = published_el.text.split("T")[0] if published_el is not None and published_el.text else ""

            # Authors
            authors = []
            for author_el in entry.findall("atom:author", namespaces):
                name_el = author_el.find("atom:name", namespaces)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # Links (PDF & DOI)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            doi = ""
            for link in entry.findall("atom:link", namespaces):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", pdf_url)
                if link.attrib.get("title") == "doi":
                    doi = link.attrib.get("href", "")

            # Categories
            primary_cat = entry.find("arxiv:primary_category", namespaces)
            cat_name = primary_cat.attrib.get("term", "") if primary_cat is not None else ""

            papers.append({
                "external_id": f"arxiv:{arxiv_id}",
                "source": "arxiv",
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published_date": published_date,
                "pdf_url": pdf_url,
                "doi": doi,
                "category": cat_name
            })

        return papers
