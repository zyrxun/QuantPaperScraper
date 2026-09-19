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

    def test_knowledge_graph_optimizations(self):
        """Test TF-IDF sparse similarity, community detection, and concept extraction."""
        from graph.concept_extractor import extract_domain_concepts
        
        # Test concept extractor
        hft_concepts = extract_domain_concepts("High-Frequency Trading in Limit Order Books", "Market microstructure and optimal execution analysis")
        self.assertIn("high-frequency trading", hft_concepts)
        self.assertIn("limit order book", hft_concepts)

        # Seed 3 distinct papers in test DB
        p1 = self.db.save_paper({
            "title": "Optimal Liquidation in Order Books",
            "abstract": "We study market microstructure and high-frequency trading.",
            "source": "arxiv", "external_id": "test:001"
        })
        self.db.save_evaluation(p1, {
            "score": 92, "hook": "Alpha execution", "breakthrough_summary": "LOB optimal liquidation",
            "takeaway": "Reduce slippage",
            "concepts": ["market microstructure", "limit order book", "optimal execution"]
        })

        p2 = self.db.save_paper({
            "title": "High Frequency Market Making Dynamics",
            "abstract": "Analysis of limit order books and optimal execution strategies.",
            "source": "arxiv", "external_id": "test:002"
        })
        self.db.save_evaluation(p2, {
            "score": 88, "hook": "Microstructure alpha", "breakthrough_summary": "Order book dynamics",
            "takeaway": "Tighten spreads",
            "concepts": ["market microstructure", "limit order book", "optimal execution"]
        })

        p3 = self.db.save_paper({
            "title": "Deep Reinforcement Learning for Alpha Mining",
            "abstract": "Neural network policy gradient framework.",
            "source": "arxiv", "external_id": "test:003"
        })
        self.db.save_evaluation(p3, {
            "score": 85, "hook": "Deep RL alpha", "breakthrough_summary": "Neural network actor critic",
            "takeaway": "Unsupervised feature extraction",
            "concepts": ["deep reinforcement learning", "neural networks"]
        })

        html_file = os.path.join(self.test_dir, "optimized_graph.html")
        builder = KnowledgeGraphBuilder(self.db, output_html=html_file)
        G = builder.build_graph(top_k_similar=2, sim_threshold=0.20)

        # p1 and p2 should have a similarity edge (shared microstructure concepts)
        edge_key = tuple(sorted([f"paper:{p1}", f"paper:{p2}"]))
        self.assertTrue(G.has_edge(edge_key[0], edge_key[1]))

        # p3 shares 0 concepts with p1, so no similarity edge should exist (NOT a complete graph)
        self.assertFalse(G.has_edge(f"paper:{p1}", f"paper:{p3}"))

        # Verify community detection assigned cluster metadata
        self.assertIn("cluster_id", G.nodes[f"paper:{p1}"])
        self.assertIn("cluster_name", G.nodes[f"paper:{p1}"])

        # Test degree-2 filtering (omits single-occurrence concepts)
        G_deg2 = builder.build_graph(min_concept_degree=2)
        self.assertNotIn("concept:deep reinforcement learning", G_deg2)

        # Test interactive HTML export features
        exported = builder.export_interactive_html()
        self.assertTrue(os.path.exists(exported))
        with open(exported, "r", encoding="utf-8") as f:
            html = f.read()
            self.assertIn("search-input", html)
            self.assertIn("cluster-filter", html)
            self.assertIn("inspector", html)
            self.assertIn("barnesHut", html)
            self.assertIn("toggleLeafNodes", html)

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

    def test_database_detailed_stats(self):
        """Test that get_detailed_stats provides category breakdown, score distribution, and top concepts."""
        p1 = self.db.save_paper({
            "external_id": "test:quant_1",
            "source": "arxiv",
            "title": "Stochastic Volatility in Rough Heston Models",
            "authors": ["M. Gatheral"],
            "abstract": "Analysis of fractional Brownian motion in rough volatility.",
            "category": "q-fin.PR"
        })
        self.db.save_evaluation(p1, {
            "score": 92,
            "hook": "Rough volatility breakthrough",
            "breakthrough_summary": "Fractional Heston model",
            "takeaway": "Direct alpha in volatility surfaces",
            "concepts": ["rough heston", "fractional brownian motion"]
        })

        p2 = self.db.save_paper({
            "external_id": "test:quant_2",
            "source": "arxiv",
            "title": "High Frequency Market Making",
            "authors": ["A. Avellaneda"],
            "abstract": "Optimal inventory control and quoting.",
            "category": "q-fin.TR"
        })
        self.db.save_evaluation(p2, {
            "score": 75,
            "hook": "Avellaneda-Stoikov market making",
            "breakthrough_summary": "Closed-form quotes",
            "takeaway": "Optimal bid-ask spread calculation",
            "concepts": ["market making", "inventory risk"]
        })

        stats = self.db.get_detailed_stats()
        self.assertEqual(stats["total_papers"], 2)
        self.assertEqual(stats["evaluated_papers"], 2)
        self.assertIn("q-fin.PR", stats["categories"])
        self.assertIn("q-fin.TR", stats["categories"])
        self.assertEqual(stats["categories"]["q-fin.PR"], 1)
        self.assertEqual(stats["categories"]["q-fin.TR"], 1)
        self.assertEqual(stats["score_distribution"]["elite"], 1)
        self.assertEqual(stats["score_distribution"]["notable"], 1)
        self.assertEqual(stats["score_distribution"]["screened"], 0)
        self.assertEqual(stats["score_distribution"]["avg_score"], 83.5)

    def test_bulk_harvest_logic(self):
        """Test bulk harvest candidate streaming and evaluation logic."""
        from pipeline import PaperPipeline
        pipeline = PaperPipeline()
        # Mock stream_candidates to return 2 sample papers
        mock_candidates = [
            {
                "external_id": "mock:1",
                "source": "arxiv",
                "title": "Deep RL for Execution",
                "authors": ["Quant A"],
                "abstract": "Fast execution with deep Q-networks.",
                "category": "q-fin.TR",
                "pdf_url": ""
            },
            {
                "external_id": "mock:2",
                "source": "openalex",
                "title": "Covariance Estimation in High Dimensions",
                "authors": ["Quant B"],
                "abstract": "Shrinkage methods for portfolio risk.",
                "category": "q-fin.PM",
                "pdf_url": ""
            }
        ]
        pipeline.scraper_manager.stream_candidates = lambda target_count, search_query="": iter(mock_candidates)
        
        result = pipeline.run_bulk_harvest(target_count=2, min_score=70, download_pdfs=False)
        self.assertEqual(result["screened"], 2)
        self.assertGreaterEqual(result["accepted"], 0)

    def test_glm_client_model_name_and_security(self):
        """Verify GLMClient dynamic model resolution, model_name property/setter, and masked key security."""
        client = GLMClient()
        self.assertTrue(hasattr(client, "model_name"))
        self.assertEqual(client.model_name, client.model)
        self.assertIn("5.3", client.model_name)

        # Dynamic setter with alias normalization
        client.model_name = "glm-5.3-plus"
        self.assertEqual(client.model_name, "glm-5.3")
        self.assertEqual(client.model, "glm-5.3")

        # Masked key security (never reveals full key)
        client.api_key = "mock_secret_key_for_testing_5678"
        masked = client.masked_key
        self.assertNotIn("secret_key", masked)
        self.assertTrue(masked.startswith("mock"))
        self.assertTrue(masked.endswith("5678"))

        # Empty / placeholder key handling
        client.api_key = ""
        self.assertEqual(client.masked_key, "Not Set (Heuristic Mode)")
        self.assertFalse(client.is_configured())

    def test_arxiv_scraper_url_encoding(self):
        """Ensure arxiv query URLs use urlencode without control characters or raw spaces."""
        from scraper.arxiv_scraper import ArxivScraper
        scraper = ArxivScraper()
        recorded_url = []
        def mock_urlopen(req, timeout=15, context=None):
            recorded_url.append(req.full_url)
            class MockResp:
                def read(self):
                    return b"<feed></feed>"
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockResp()

        import urllib.request
        orig = urllib.request.urlopen
        urllib.request.urlopen = mock_urlopen
        try:
            scraper.search(categories=["q-fin.TR", "q-fin.PM"])
            self.assertTrue(len(recorded_url) > 0)
            self.assertNotIn(" ", recorded_url[0])
            self.assertIn("cat%3Aq-fin.TR", recorded_url[0])
        finally:
            urllib.request.urlopen = orig

    def test_safe_urlopen_ssl_fallback(self):
        """Verify safe_urlopen catches CERTIFICATE_VERIFY_FAILED and retries with unverified fallback context."""
        from scraper.arxiv_scraper import safe_urlopen
        import urllib.error
        attempts = []
        def mock_ssl_urlopen(req, timeout=15, context=None):
            attempts.append(context)
            if len(attempts) == 1:
                raise urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate")
            class MockResp:
                def read(self):
                    return b"OK"
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockResp()

        import urllib.request
        orig = urllib.request.urlopen
        urllib.request.urlopen = mock_ssl_urlopen
        try:
            req = urllib.request.Request("https://example.com")
            resp = safe_urlopen(req)
            self.assertEqual(resp.read(), b"OK")
            self.assertEqual(len(attempts), 2)
        finally:
            urllib.request.urlopen = orig

if __name__ == "__main__":
    unittest.main()



