"""
Core pipeline coordinating scraping, GLM abstract evaluation, PDF downloading,
tokenization, local database storage, and knowledge graph generation.
"""

import os
import sys
import yaml
from typing import Dict, Any, List, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
        
        # Analyzer with GLM (Dynamic model from env/config, defaulting to glm-5.3-plus)
        llm_cfg = self.config.get("llm", {})
        active_model = os.getenv("GLM_MODEL") or llm_cfg.get("model") or "glm-5.3-plus"
        self.glm_client = GLMClient(model=active_model)
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

    def search_and_ingest(
        self,
        query: str,
        limit: int = 5,
        only_undownloaded: bool = True,
        download_pdfs: bool = True
    ) -> List[Dict[str, Any]]:
        """
        On-demand search command for CLI or Discord.
        Fetches, evaluates with query relevance, and stores papers.
        Prioritizes un-downloaded papers and downloads PDFs politely.
        """
        papers = self.scraper_manager.fetch_candidates(
            search_query=query,
            limit=limit,
            only_undownloaded=only_undownloaded
        )
        results = []
        for paper in papers:
            paper_id = self.db.save_paper(paper)
            paper["id"] = paper_id
            eval_result = self.evaluator.evaluate(paper, self.config.get("filters", {}), query=query)
            self.db.save_evaluation(paper_id, eval_result)
            paper.update(eval_result)
            paper["is_downloaded"] = self.db.is_pdf_downloaded(paper_id)
            results.append(paper)

        # Multi-threaded download for qualified un-downloaded papers
        to_download = [p for p in results if not p.get("is_downloaded") and p.get("pdf_url")]
        if download_pdfs and to_download:
            self.downloader.download_batch(
                to_download,
                on_download_complete=self._process_downloaded_pdf
            )

        # Refresh graph
        try:
            self.graph_builder.export_interactive_html()
        except Exception:
            pass

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def run_bulk_harvest(
        self,
        target_count: int = 100,
        min_score: int = 70,
        download_pdfs: bool = True,
        search_query: str = ""
    ) -> Dict[str, Any]:
        """
        Streams and evaluates papers in continuous succession (hundreds, thousands, or tens of thousands):
        - Paginates through arXiv and OpenAlex continuously.
        - Feeds each incoming abstract to GLM model to screen for quantitative finance alpha.
        - Stores all paper metadata, categories, and evaluation results in SQLite.
        - For papers meeting or exceeding min_score, downloads PDFs politely and tokenizes them.
        - Safe interruption via KeyboardInterrupt (Ctrl+C).
        """
        import time
        start_time = time.time()
        model_display = getattr(self.glm_client, "model_name", getattr(self.glm_client, "model", "glm-5.3-plus"))
        print(f"\n[Bulk Harvest] 🚀 Starting successive ingestion of up to {target_count:,} papers...")
        print(f"[Bulk Harvest] 🤖 Evaluation model: {model_display}")
        print(f"[Bulk Harvest] 🎯 PDF download threshold: score >= {min_score}")
        print(f"[Bulk Harvest] ℹ️  Press Ctrl+C anytime to pause/stop safely with 0 data loss.\n")

        screened = 0
        accepted = 0
        downloaded = 0
        accepted_buffer = []

        try:
            candidate_stream = self.scraper_manager.stream_candidates(
                target_count=target_count,
                search_query=search_query
            )

            for paper in candidate_stream:
                try:
                    # 1. Save paper to SQLite
                    paper_id = self.db.save_paper(paper)
                    paper["id"] = paper_id
                    cat = paper.get("category", "General Quant")
                    src = paper.get("source", "unknown").upper()
                    
                    # 2. Feed abstract to GLM model for evaluation
                    eval_result = self.evaluator.evaluate(paper, self.config.get("filters", {}))
                    self.db.save_evaluation(paper_id, eval_result)
                    paper.update(eval_result)
                    score = eval_result.get("score", 0)
                    screened += 1

                    # 3. Live progress logging
                    tier = "💎 ALPHA/ELITE" if score >= 85 else ("⚡ NOTABLE" if score >= 70 else "⚪ SCREENED OUT")
                    print(f"[{screened:,}/{target_count:,}] [{src}|{cat}] '{paper['title'][:42]}...'")
                    print(f"       -> Score: {score}/100 [{tier}] | Hook: {eval_result.get('hook', '')[:65]}")

                    # 4. Check if paper passes qualification threshold
                    if score >= min_score:
                        accepted += 1
                        if download_pdfs and paper.get("pdf_url"):
                            accepted_buffer.append(paper)

                    # Flush download buffer when it reaches batch size of 5
                    if len(accepted_buffer) >= 5:
                        print(f"\n[Bulk Harvest] 📥 Downloading batch of {len(accepted_buffer)} qualified PDFs...")
                        self.downloader.download_batch(
                            accepted_buffer,
                            on_download_complete=self._process_downloaded_pdf
                        )
                        downloaded += len(accepted_buffer)
                        accepted_buffer = []

                except Exception as e:
                    print(f"[Bulk Harvest] Warning: Error processing paper '{paper.get('title', '')}': {e}")

            # Flush any remaining accepted papers
            if accepted_buffer and download_pdfs:
                print(f"\n[Bulk Harvest] 📥 Downloading final batch of {len(accepted_buffer)} qualified PDFs...")
                self.downloader.download_batch(
                    accepted_buffer,
                    on_download_complete=self._process_downloaded_pdf
                )
                downloaded += len(accepted_buffer)

        except KeyboardInterrupt:
            print("\n\n[Bulk Harvest] ⏸️ Ingestion paused by user (Ctrl+C). Saving progress...")
            if accepted_buffer and download_pdfs:
                print(f"[Bulk Harvest] Downloading pending {len(accepted_buffer)} qualified PDFs...")
                self.downloader.download_batch(
                    accepted_buffer,
                    on_download_complete=self._process_downloaded_pdf
                )
                downloaded += len(accepted_buffer)

        # Update Knowledge Graph
        try:
            print("[Bulk Harvest] 🌐 Updating Knowledge Graph...")
            self.graph_builder.export_interactive_html()
        except Exception as e:
            print(f"[Bulk Harvest] Could not update graph: {e}")

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print("\n" + "=" * 65)
        print("          🏁 BULK HARVEST & SCREENING COMPLETE")
        print("=" * 65)
        print(f" Papers Screened by GLM:  {screened:,}")
        print(f" Papers Accepted (Score>={min_score}): {accepted:,}")
        print(f" Full PDFs Processed:     {downloaded:,}")
        print(f" Total Time Elapsed:      {mins}m {secs}s")
        print("=" * 65)

        return {
            "screened": screened,
            "accepted": accepted,
            "downloaded": downloaded,
            "elapsed_seconds": elapsed
        }
