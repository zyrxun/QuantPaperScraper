"""
Polite multi-threaded downloader for fetching full-text scientific PDFs concurrently
without getting rate-limited, cancelled, or IP-banned by academic repositories (e.g., arXiv).
"""

import os
import time
import random
import threading
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

class PDFDownloader:
    def __init__(
        self,
        download_dir: str = "data/pdfs",
        max_workers: int = 3,
        request_delay: float = 1.5,
        max_retries: int = 3,
        timeout: int = 35,
        user_agent: str = "QuantPaperBot/1.0 (academic-research; mailto:quantresearcher@example.com)"
    ):
        self.download_dir = download_dir
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent
        self._lock = threading.Lock()
        self._last_request_time = 0.0

        os.makedirs(self.download_dir, exist_ok=True)

    def _wait_for_rate_limit(self):
        """
        Enforces a polite request delay with random jitter across threads
        to prevent triggering anti-scraping 429/IP-ban alarms on academic servers.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            # Jitter: +/- 25% of request_delay
            target_delay = self.request_delay * random.uniform(0.75, 1.25)
            if elapsed < target_delay:
                sleep_time = target_delay - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def download_pdf(self, pdf_url: str, paper_id: int) -> Optional[str]:
        """
        Downloads a single PDF with automatic retries, exponential backoff,
        and validation of PDF magic bytes.
        """
        if not pdf_url:
            return None

        clean_filename = f"paper_{paper_id}.pdf"
        target_path = os.path.join(self.download_dir, clean_filename)

        # Skip if already downloaded and valid
        if os.path.exists(target_path) and os.path.getsize(target_path) > 3000:
            return target_path

        # Normalize URL (arXiv /abs/ to /pdf/)
        if "arxiv.org/abs/" in pdf_url:
            pdf_url = pdf_url.replace("/abs/", "/pdf/")
        if "arxiv.org" in pdf_url and not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"

        req = urllib.request.Request(
            pdf_url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,application/octet-stream,*/*",
                "Referer": "https://arxiv.org/"
            }
        )

        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()

                # Validate minimum size and PDF header (%PDF-)
                if len(data) > 2000 and (data.startswith(b"%PDF") or b"%PDF-" in data[:1024]):
                    with open(target_path, "wb") as f:
                        f.write(data)
                    return target_path
                else:
                    # Received HTML (e.g. Cloudflare or 403 page) instead of PDF
                    print(f"[PDFDownloader] Received non-PDF data for {pdf_url} (attempt {attempt}/{self.max_retries})")
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    # Rate-limited by repository: exponential backoff with jitter
                    backoff = (2 ** attempt) * random.uniform(1.5, 3.0)
                    print(f"[PDFDownloader] HTTP {e.code} (Rate Limited) on {pdf_url}. Backing off for {backoff:.1f}s...")
                    time.sleep(backoff)
                elif e.code == 404:
                    print(f"[PDFDownloader] HTTP 404 Not Found for {pdf_url}")
                    break
                else:
                    print(f"[PDFDownloader] HTTP {e.code} for {pdf_url}: {e.reason}")
            except Exception as e:
                print(f"[PDFDownloader] Attempt {attempt} failed for {pdf_url}: {e}")
                time.sleep(1.0 * attempt)

        # Cleanup failed artifact if any
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        return None

    def download_batch(
        self,
        papers: List[Dict[str, Any]],
        on_download_complete: Optional[Callable[[Dict[str, Any], str], None]] = None
    ) -> Dict[int, Optional[str]]:
        """
        Downloads a batch of papers concurrently using a polite worker pool.
        Calls optional callback `on_download_complete(paper, local_path)` upon each completion.
        """
        results: Dict[int, Optional[str]] = {}
        if not papers:
            return results

        print(f"[PDFDownloader] Launching polite multithreaded download for {len(papers)} papers (workers: {self.max_workers})...")

        def _worker(paper: Dict[str, Any]):
            paper_id = paper.get("id")
            pdf_url = paper.get("pdf_url")
            if not pdf_url:
                return paper_id, None
            path = self.download_pdf(pdf_url, paper_id)
            return paper_id, path

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(_worker, paper): paper for paper in papers if paper.get("pdf_url")
            }

            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                paper_id = paper.get("id")
                try:
                    pid, local_path = future.result()
                    results[pid] = local_path
                    if local_path:
                        print(f"[PDFDownloader] Successfully downloaded: #{pid} '{paper.get('title', '')[:40]}...'")
                        if on_download_complete:
                            on_download_complete(paper, local_path)
                    else:
                        print(f"[PDFDownloader] Could not download PDF for #{pid}")
                except Exception as e:
                    print(f"[PDFDownloader] Worker error on #{paper_id}: {e}")
                    results[paper_id] = None

        return results
