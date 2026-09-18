"""
Core pipeline coordinating scraping, GLM abstract evaluation, PDF downloading,
tokenization, local database storage, and knowledge graph generation.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from storage.database import Database
from scraper.manager import ScraperManager
from analyzer.glm_client import GLMClient
from analyzer.evaluator import PaperEvaluator
from pdf_processor.pdf_downloader import PDFDownloader
from pdf_processor.tokenizer import PDFTokenizer
from graph.graph_builder import KnowledgeGraphBuilder

class PaperPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

        storage_cfg = self.config.get("storage", {})
        downloader_cfg = self.config.get("downloader", {})
        self.db = Database(db_path=storage_cfg.get("db_path", "data/papers.db"))
        self.scraper_manager = ScraperManager(self.db, self.config)
        
        # Analyzer with GLM
        self.glm_client = GLMClient()
        self.evaluator = PaperEvaluator(self.glm_client, self.config)

        # Multi-Threaded Polite PDF Downloader & Tokenization
        self.downloader = PDFDownloader(
            download_dir=storage_cfg.get("pdf_dir", "data/pdfs"),
            max_workers=downloader_cfg.get("max_workers", 3),
            request_delay=downloader_cfg.get("request_delay_seconds", 1.5),
            max_retries=downloader_cfg.get("max_retries", 3),
            timeout=downloader_cfg.get("timeout_seconds", 35)
        )
        self.tokenizer = PDFTokenizer()

        # Knowledge Graph
        self.graph_builder = KnowledgeGraphBuilder(
            self.db,
            output_html=storage_cfg.get("graph_output", "data/graph.html")
        )

    def load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_config(self):
        """Saves current configuration to YAML."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

    def update_filter_interests(self, new_interests: List[str]):
        """Dynamically updates user interest topics (can be called via Discord DM/command)."""
        if "filters" not in self.config:
            self.config["filters"] = {}
        self.config["filters"]["interests"] = new_interests
        self.save_config()

    def update_min_score(self, min_score: int):
        """Dynamically updates minimum score filter."""
        if "filters" not in self.config:
            self.config["filters"] = {}
        self.config["filters"]["min_score"] = int(min_score)
        self.save_config()

    def _process_downloaded_pdf(self, paper: Dict[str, Any], local_path: str):
        """Helper callback to tokenize PDF and save token stats into DB."""
        paper_id = paper.get("id")
        if not paper_id or not local_path:
            return

        try:
            self.db.update_local_pdf_path(paper_id, local_path)
            token_data = self.tokenizer.tokenize_document(local_path)
            self.db.save_token_info(
                paper_id,
                total_tokens=token_data["total_tokens"],
                section_breakdown=token_data["section_breakdown"],
                keywords=token_data["extracted_keywords"]
            )
            print(f"[Pipeline] Tokenized #{paper_id}: {token_data['total_tokens']:,} tokens extracted.")
        except Exception as e:
            print(f"[Pipeline] Error tokenizing PDF for #{paper_id}: {e}")

    def run_daily_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Executes the full automated daily cycle:
        1. Fetch fresh paper candidates from arXiv and OpenAlex.
        2. Evaluate all abstracts with GLM.
        3. Save candidates & scores to SQLite database.
        4. Concurrently download PDFs with rate-limiting & backoff.
        5. Tokenize documents and build knowledge graph.
        6. Returns the top paper dictionary for Discord posting.
        """
        print("[Pipeline] Starting daily paper harvest cycle...")
        candidates_limit = self.config.get("scheduler", {}).get("candidates_per_fetch", 15)
        candidates = self.scraper_manager.fetch_candidates(limit=candidates_limit)
        print(f"[Pipeline] Found {len(candidates)} new candidates to evaluate.")

        evaluated_candidates = []
        for paper in candidates:
            try:
                # Save paper to DB
                paper_id = self.db.save_paper(paper)
                paper["id"] = paper_id
                print(f"[Pipeline] Evaluating [{paper['source'].upper()}] '{paper['title'][:45]}...'")
                
                # GLM Abstract Evaluation
                eval_result = self.evaluator.evaluate(paper, self.config.get("filters", {}))
                self.db.save_evaluation(paper_id, eval_result)
                paper.update(eval_result)
                evaluated_candidates.append(paper)
                print(f"[Pipeline] -> Score: {eval_result['score']}/100 | Hook: {eval_result['hook']}")
            except Exception as e:
                print(f"[Pipeline] Error evaluating paper {paper.get('title')}: {e}")

        # Multi-Threaded PDF Download & Tokenization (rate-limited to avoid bans)
        downloader_cfg = self.config.get("downloader", {})
        if downloader_cfg.get("download_all_candidates", True):
            print("[Pipeline] Initiating polite multithreaded PDF downloading for candidates...")
            self.downloader.download_batch(
                evaluated_candidates,
                on_download_complete=self._process_downloaded_pdf
            )

        # Select the best unposted paper
        min_score = self.config.get("filters", {}).get("min_score", 70)
        top_paper = self.db.get_top_unposted_paper(min_score=min_score)

        if top_paper:
            paper_id = top_paper["id"]
            pdf_url = top_paper.get("pdf_url")
            print(f"[Pipeline] Selected daily top paper: #{paper_id} '{top_paper['title']}' (Score: {top_paper['score']})")

            # If top paper wasn't downloaded in batch, download individually
            if pdf_url and not top_paper.get("local_pdf_path"):
                print(f"[Pipeline] Downloading top pick PDF from {pdf_url}...")
                local_path = self.downloader.download_pdf(pdf_url, paper_id)
                if local_path:
                    self._process_downloaded_pdf(top_paper, local_path)

            # Update Knowledge Graph
            print("[Pipeline] Updating knowledge graph...")
            self.graph_builder.export_interactive_html()
            print("[Pipeline] Knowledge graph updated at data/graph.html.")

        return top_paper

    def search_and_ingest(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        On-demand search command for CLI or Discord. Fetches, evaluates, and stores papers.
        Downloads PDFs concurrently with polite rate-limiting.
        """
        papers = self.scraper_manager.fetch_candidates(search_query=query, limit=limit)
        results = []
        for paper in papers:
            paper_id = self.db.save_paper(paper)
            paper["id"] = paper_id
            eval_result = self.evaluator.evaluate(paper, self.config.get("filters", {}))
            self.db.save_evaluation(paper_id, eval_result)
            paper.update(eval_result)
            results.append(paper)

        # Multi-threaded download for search results
        if results:
            self.downloader.download_batch(
                results,
                on_download_complete=self._process_downloaded_pdf
            )

        # Refresh graph
        self.graph_builder.export_interactive_html()
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
