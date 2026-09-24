"""
Local Quantitative Research Paper Evaluator (100% Offline & Free).
Scores papers deterministically (0-100) based on:
1. Topic & Query Relevance (Title/Abstract keyword density & exact phrase matching)
2. Mathematical Rigor (Stochastic calculus, PDEs, cointegration, martingales, Kalman filters, etc.)
3. Empirical Validation (Tick data, backtesting, Sharpe ratio, drawdowns, slippage, order books)
4. Academic Significance (Citation count, recency, core quant categories)
5. Exclusion Filtering (Penalizes medical, fraud detection, agricultural, or general ML noise)
"""

import re
from typing import Dict, Any, List, Optional
from graph.concept_extractor import extract_domain_concepts

MATH_RIGOR_TERMS = {
    # High-rigor models (weight: 4)
    "cointegration": 4, "ornstein-uhlenbeck": 4, "kalman filter": 4,
    "stochastic volatility": 4, "martingale": 4, "eigenvalue": 4, "eigenvalues": 4,
    "pca": 3, "principal component": 4, "markov switching": 4, "regime-switching": 4,
    "jump diffusion": 4, "ito calculus": 4, "hamilton-jacobi-bellman": 4, "hjb": 4,
    "copula": 4, "copulas": 4, "garch": 3, "vector autoregression": 4, "var model": 3,
    "partial differential": 4, "brownian motion": 4, "levy process": 4, "spectral": 3,
    "mean reversion": 4, "mean-reverting": 4, "stochastic control": 4,
    # Algorithmic & ML models (weight: 3)
    "reinforcement learning": 3, "deep learning": 3, "q-learning": 3,
    "actor-critic": 3, "neural network": 3, "transformer": 3, "convex optimization": 3,
    "quadratic programming": 3, "dynamic programming": 3, "optimal transport": 3,
    # Structural mechanics (weight: 2)
    "no-arbitrage": 3, "risk-neutral": 3, "greeks": 3, "implied volatility": 3,
    "term structure": 2, "yield curve": 2, "cross-sectional": 2, "factor exposure": 2,
    "factor model": 3, "fama-french": 3
}

EMPIRICAL_TERMS = {
    # Strategy & execution (weight: 3)
    "sharpe ratio": 4, "sortino": 3, "drawdown": 4, "pnl": 3, "profit and loss": 3,
    "backtest": 4, "backtesting": 4, "out-of-sample": 4, "cross-validation": 3,
    "slippage": 4, "transaction cost": 4, "transaction costs": 4, "market impact": 4,
    "limit order book": 4, "order book": 3, "tick-by-tick": 4, "high-frequency": 3,
    "intraday": 3, "pairs trading": 4, "alpha": 3, "alpha signals": 4,
    # Real market data validation (weight: 2)
    "empirical": 2, "nyse": 3, "nasdaq": 3, "s&p 500": 3, "equities": 2,
    "futures": 2, "options": 2, "order flow": 3, "bid-ask spread": 3, "order flow imbalance": 4
}

NON_QUANT_EXCLUSIONS = [
    "lung disease", "mri", "cardiac", "cancer", "patient", "clinical trial",
    "clinical medicine", "agriculture", "crop", "credit card fraud", "fraud detection",
    "collision at", "particle physics", "e-commerce recommendation", "traffic congestion"
]

class LocalPaperEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def evaluate(self, paper: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """
        Evaluates a paper candidate locally without external API dependencies.
        Returns score (0-100), hook, breakthrough_summary, takeaway, and extracted concepts.
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        category = paper.get("category", "")
        citations = int(paper.get("citations", 0) or 0)
        published_date = str(paper.get("published_date", ""))
        source = paper.get("source", "arxiv").lower()

        combined_text = f"{title} {abstract} {category}".lower()

        # 1. Non-quant exclusion check
        is_bad = any(bad in combined_text for bad in NON_QUANT_EXCLUSIONS)
        if is_bad:
            return {
                "is_quant_finance": False,
                "score": 12,
                "hook": f"Irrelevant non-financial study: {title[:45]}...",
                "breakthrough_summary": "Paper does not investigate quantitative finance, market microstructure, or asset pricing.",
                "takeaway": "Rejected: outside quantitative finance scope.",
                "concepts": ["non-quant", "unrelated"],
                "model_name": "local-heuristic"
            }

        # 2. Topic & Query Relevance (0 to 35 points)
        relevance_score = 0
        clean_q = query.strip().lower()
        if clean_q:
            clean_phrase = clean_q.strip('"').strip("'")
            words = [w for w in clean_phrase.split() if len(w) > 2]

            # Exact phrase in title (massive boost)
            if clean_phrase in title.lower():
                relevance_score += 25
            elif words and all(w in title.lower() for w in words):
                relevance_score += 18
            elif words and any(w in title.lower() for w in words):
                relevance_score += 10

            # Exact phrase in abstract
            if clean_phrase in abstract.lower():
                relevance_score += 15
            elif words:
                count_in_abs = sum(abstract.lower().count(w) for w in words)
                relevance_score += min(12, count_in_abs * 3)

            relevance_score = min(35, relevance_score)
        else:
            # Baseline domain relevance
            core_quant = ["trading", "arbitrage", "volatility", "portfolio", "microstructure", "pricing", "alpha"]
            domain_hits = sum(1 for k in core_quant if k in combined_text)
            relevance_score = min(30, domain_hits * 6 + 10)

        # 3. Mathematical Rigor (0 to 30 points)
        rigor_score = 0
        matched_rigor = []
        for term, weight in MATH_RIGOR_TERMS.items():
            if term in combined_text:
                rigor_score += weight
                matched_rigor.append(term)
        rigor_score = min(30, rigor_score)

        # 4. Empirical Validation & Strategy Metrics (0 to 20 points)
        empirical_score = 0
        matched_empirical = []
        for term, weight in EMPIRICAL_TERMS.items():
            if term in combined_text:
                empirical_score += weight
                matched_empirical.append(term)
        empirical_score = min(20, empirical_score)

        # 5. Significance & Citation Weight (0 to 15 points)
        significance_score = 0
        if citations >= 100:
            significance_score = 15
        elif citations >= 50:
            significance_score = 12
        elif citations >= 20:
            significance_score = 9
        elif citations >= 5:
            significance_score = 6
        elif citations >= 1:
            significance_score = 3
        else:
            # For recent arXiv papers, reward recency and core quant categories
            if published_date and any(yr in published_date for yr in ["2024", "2025", "2026"]):
                significance_score += 8
            elif published_date and any(yr in published_date for yr in ["2022", "2023"]):
                significance_score += 5
            if any(c in category for c in ["q-fin.PM", "q-fin.TR", "q-fin.CP", "q-fin.ST", "q-fin.MF"]):
                significance_score += 4
            significance_score = min(15, significance_score)

        # Total Raw Score
        raw_total = relevance_score + rigor_score + empirical_score + significance_score

        # Domain baseline guarantee: if clean_q was matched strongly in title, ensure high score
        if clean_q and clean_q in title.lower():
            raw_total = max(raw_total, 76)

        final_score = min(98, max(25, raw_total))

        # 6. Extract dynamic hook, summary, and takeaway
        hook = self._generate_hook(title, abstract, clean_q, matched_rigor, matched_empirical)
        breakthrough = self._generate_breakthrough(title, abstract, matched_rigor)
        takeaway = self._generate_takeaway(title, abstract, matched_empirical, final_score)
        concepts = extract_domain_concepts(title, abstract, category)

        return {
            "is_quant_finance": True,
            "score": final_score,
            "hook": hook,
            "breakthrough_summary": breakthrough,
            "takeaway": takeaway,
            "concepts": concepts,
            "model_name": "local-quant-evaluator"
        }

    def _generate_hook(self, title: str, abstract: str, query: str, rigor_terms: List[str], empirical_terms: List[str]) -> str:
        # Search abstract for sentences highlighting contributions
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        for s in sentences:
            s_clean = s.strip()
            if any(s_clean.lower().startswith(lead) for lead in ["we propose", "this paper introduces", "we demonstrate", "we develop", "our results show", "we construct"]):
                if len(s_clean) > 20 and len(s_clean) < 160:
                    return s_clean

        # Synthesize from title & recognized quantitative mechanics
        mechanics = rigor_terms[:2] if rigor_terms else (empirical_terms[:2] if empirical_terms else ["quantitative modeling"])
        return f"Advances {', '.join(mechanics)} to model and capture alpha in {title[:50]}."

    def _generate_breakthrough(self, title: str, abstract: str, rigor_terms: List[str]) -> str:
        if rigor_terms:
            return f"Employs {', '.join(rigor_terms[:3])} to formulate a rigorous mathematical framework with empirical market validation."
        return f"Investigates quantitative market dynamics and statistical behavior in {title[:60]}."

    def _generate_takeaway(self, title: str, abstract: str, empirical_terms: List[str], score: int) -> str:
        if empirical_terms:
            return f"Actionable for systematic strategies evaluating {', '.join(empirical_terms[:2])} under realistic execution dynamics."
        return "Provides empirical and computational insights relevant to quantitative asset pricing and risk management."