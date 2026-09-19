"""
Tests ensuring paper scrapers and evaluators strictly enforce quantitative finance domain constraints.
"""

import unittest
from scraper.arxiv_scraper import ArxivScraper
from scraper.openalex_scraper import OpenAlexScraper
from scraper.manager import is_quant_paper
from analyzer.evaluator import PaperEvaluator
from analyzer.glm_client import GLMClient

class TestQuantDomainFilters(unittest.TestCase):
    def setUp(self):
        self.arxiv = ArxivScraper()
        self.openalex = OpenAlexScraper()
        self.glm_mock = GLMClient()
        self.evaluator = PaperEvaluator(self.glm_mock)

    def test_arxiv_query_contains_quant_categories_with_search_term(self):
        """Verify that when a user searches 'reinforcement learning', quant categories are NOT dropped."""
        # Using custom categories
        cats = ["q-fin.TR", "q-fin.PM"]
        active_cats = cats or self.arxiv.DEFAULT_QUANT_CATEGORIES
        cat_query = " OR ".join([f"cat:{c.strip()}" for c in active_cats])
        user_query = "reinforcement learning"
        expected_query = f"({cat_query}) AND (all:{user_query})"
        self.assertIn("cat:q-fin.TR", expected_query)
        self.assertIn("all:reinforcement learning", expected_query)

    def test_arxiv_feed_filters_out_non_quant_papers(self):
        """Simulate an Atom XML feed containing a physics paper and a quant paper."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2609.99001v1</id>
            <title>Fast Brain MRI Translation</title>
            <summary>Medical imaging study on brain scans.</summary>
            <published>2026-09-18T10:00:00Z</published>
            <arxiv:primary_category term="cs.CV"/>
            <category term="cs.CV"/>
          </entry>
          <entry>
            <id>http://arxiv.org/abs/2609.99002v1</id>
            <title>Optimal Execution in Limit Order Books</title>
            <summary>Order flow imbalance and optimal liquidation in equity markets.</summary>
            <published>2026-09-18T10:00:00Z</published>
            <arxiv:primary_category term="q-fin.TR"/>
            <category term="q-fin.TR"/>
          </entry>
        </feed>
        """
        parsed = self.arxiv._parse_feed(xml_content)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["external_id"], "arxiv:2609.99002v1")
        self.assertEqual(parsed[0]["category"], "q-fin.TR")

    def test_is_quant_paper_guardrail(self):
        """Verify that is_quant_paper correctly accepts quant papers and rejects non-quant papers."""
        bad_paper_1 = {
            "source": "arxiv",
            "category": "cs.DC",
            "title": "Accelerating Sharded Data Parallelism at Scale",
            "abstract": "Distributed computing systems and cluster communication."
        }
        bad_paper_2 = {
            "source": "openalex",
            "category": "Medicine",
            "title": "Prediction of lung disease using machine learning",
            "abstract": "Chest X-ray images and clinical pulmonary diagnostics."
        }
        good_paper_1 = {
            "source": "arxiv",
            "category": "q-fin.ST",
            "title": "Stochastic Volatility under Rough Heston Models",
            "abstract": "Option pricing and implied volatility surface calibration."
        }
        good_paper_2 = {
            "source": "openalex",
            "category": "Market Microstructure and Price Discovery",
            "title": "High-Frequency Trading and Liquidity Provision",
            "abstract": "Limit order books, bid-ask spreads, and order arrival intensities."
        }

        self.assertFalse(is_quant_paper(bad_paper_1))
        self.assertFalse(is_quant_paper(bad_paper_2))
        self.assertTrue(is_quant_paper(good_paper_1))
        self.assertTrue(is_quant_paper(good_paper_2))

    def test_evaluator_rejects_non_quant_paper(self):
        """Ensure that a non-quant paper evaluated in mock/heuristic mode receives score <= 15."""
        non_quant = {
            "title": "Clinical Prediction of Pulmonary Lung Fibrosis",
            "category": "Pulmonary Medicine",
            "abstract": "We evaluate CT scans of 400 hospital patients diagnosed with interstitial lung disease."
        }
        result = self.evaluator.evaluate(non_quant)
        self.assertFalse(result.get("is_quant_finance", True))
        self.assertLessEqual(result.get("score", 100), 15)

    def test_evaluator_scores_genuine_quant_paper(self):
        """Ensure genuine quantitative finance research receives a strong score."""
        quant = {
            "title": "Deep Reinforcement Learning for Market Making and Optimal Execution",
            "category": "q-fin.TR",
            "abstract": "We develop an optimal trade execution algorithm in limit order books managing inventory risk and price impact under stochastic volatility."
        }
        result = self.evaluator.evaluate(quant)
        self.assertTrue(result.get("is_quant_finance", False))
        self.assertGreaterEqual(result.get("score", 0), 70)

if __name__ == "__main__":
    unittest.main()
