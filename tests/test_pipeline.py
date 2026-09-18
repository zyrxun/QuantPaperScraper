"""
Unit and integration tests for the Paper Scraper pipeline.
"""

import os
import unittest
import tempfile
import shutil
import json
from storage.database import Database
from analyzer.evaluator import PaperEvaluator
from analyzer.glm_client import GLMClient
from pdf_processor.tokenizer import PDFTokenizer
from graph.graph_builder import KnowledgeGraphBuilder

class TestPaperPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_papers.db")
        self.db = Database(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_paper_and_eval_lifecycle(self):
        """Verify paper insertion, deduplication, evaluation and unposted selection."""
        paper_data = {
            "external_id": "arxiv:2401.99999",
            "source": "arxiv",
            "title": "Optimal Execution in Limit Order Books with Deep Reinforcement Learning",
            "authors": ["Dr. Quantitative", "Quant Analyst"],
            "abstract": "We analyze transient market impact, inventory risk, and execution costs under high-frequency order arrival.",
            "published_date": "2024-01-15",
            "pdf_url": "https://arxiv.org/pdf/2401.99999.pdf"
        }

        # Check paper existence before saving
        self.assertFalse(self.db.paper_exists("arxiv:2401.99999"))

        # Save paper
        paper_id = self.db.save_paper(paper_data)
        self.assertGreater(paper_id, 0)
        self.assertTrue(self.db.paper_exists("arxiv:2401.99999"))

        # Save evaluation
        eval_data = {
            "score": 94,
            "hook": "Presents a provably convergent execution algorithm outperforming Almgren-Chriss in volatile regimes.",
            "breakthrough_summary": "Introduces state-dependent transaction cost kernels parameterized by neural networks.",
            "takeaway": "Achieves 14 basis points slippage reduction across equity order books.",
            "concepts": ["optimal execution", "limit order books", "market impact", "reinforcement learning"]
        }
        self.db.save_evaluation(paper_id, eval_data)

        # Retrieve top unposted paper
        top = self.db.get_top_unposted_paper(min_score=80)
        self.assertIsNotNone(top)
        self.assertEqual(top["id"], paper_id)
        self.assertEqual(top["score"], 94)
        self.assertIn("optimal execution", top["concepts"])

        # Mark as posted
        self.db.mark_as_posted(paper_id, "123456789", "987654321")
        
        # Verify it is no longer returned as unposted
        unposted = self.db.get_top_unposted_paper(min_score=80)
        self.assertIsNone(unposted)

    def test_evaluator_parsing(self):
        """Test robust JSON parsing in evaluator."""
        evaluator = PaperEvaluator()
        
        # Standard json markdown block
        sample_output = """
```json
{
  "score": 88,
  "hook": "Profound discovery in brain-machine interfaces.",
  "breakthrough_summary": "Direct bidirectional synaptic telemetry achieved.",
  "takeaway": "Opens new avenues for cognitive neuroprosthetics.",
  "concepts": ["synaptic plasticity", "neuroprosthetics", "neural telemetry"]
}
```
"""
        parsed = evaluator._parse_response(sample_output)
        self.assertEqual(parsed["score"], 88)
        self.assertIn("synaptic plasticity", parsed["concepts"])

    def test_tokenizer_and_keywords(self):
        """Test BPE token counting and keyword extraction."""
        tokenizer = PDFTokenizer()
        text = "Deep neural networks utilize transformer attention mechanisms to model complex sequences in natural language processing."
        
        tokens = tokenizer.count_tokens(text)
        self.assertGreater(tokens, 5)

        keywords = tokenizer.extract_keywords(text, top_k=3)
        self.assertTrue(len(keywords) > 0)

    def test_graph_builder_and_export(self):
        """Test NetworkX knowledge graph construction and Vis.js HTML export."""
        paper_id1 = self.db.save_paper({
            "external_id": "test:001",
            "source": "test",
            "title": "Neuro-symbolic Computation",
            "authors": ["Dr. Researcher"],
            "abstract": "Abstract text here...",
            "published_date": "2024"
        })
        self.db.save_evaluation(paper_id1, {
            "score": 85,
            "hook": "Hook 1",
            "breakthrough_summary": "Summary 1",
            "takeaway": "Takeaway 1",
            "concepts": ["symbolic ai", "logic programming"]
        })

        html_file = os.path.join(self.test_dir, "graph.html")
        builder = KnowledgeGraphBuilder(self.db, output_html=html_file)
        
        G = builder.build_graph()
        self.assertTrue(G.has_node(f"paper:{paper_id1}"))
        self.assertTrue(G.has_node("concept:symbolic ai"))

        exported_path = builder.export_interactive_html()
        self.assertTrue(os.path.exists(exported_path))
        with open(exported_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Neuro-symbolic Computation", content)
            self.assertIn("vis.Network", content)

    def test_downloader_batch_empty_and_politeness(self):
        """Test that download_batch safely handles empty or local mock lists with polite pool."""
        from pdf_processor.pdf_downloader import PDFDownloader
        downloader = PDFDownloader(
            download_dir=os.path.join(self.test_dir, "pdfs"),
            max_workers=2,
            request_delay=0.01
        )
        # Empty batch returns empty dict
        results = downloader.download_batch([])
        self.assertEqual(results, {})

if __name__ == "__main__":
    unittest.main()
