"""
Evaluates research paper candidates using:
1. High-precision Local Quant Evaluator (Default, 100% offline & free, no API key needed)
2. Free Google Gemini API (gemini-1.5-flash / gemini-2.0-flash via Google AI Studio)
3. Zhipu GLM API (glm-4-flash free tier or paid glm-5.3-plus)
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from .local_evaluator import LocalPaperEvaluator
from .glm_client import GLMClient
from .gemini_client import GeminiClient

class PaperEvaluator:
    def __init__(self, glm_client: GLMClient = None, config: Dict[str, Any] = None, backend: str = "local"):
        self.config = config or {}
        self.backend = os.getenv("EVAL_BACKEND", backend)
        self.local_evaluator = LocalPaperEvaluator(self.config)
        self.glm_client = glm_client or GLMClient()
        self.gemini_client = GeminiClient()

    def evaluate(self, paper: Dict[str, Any], filter_settings: Dict[str, Any] = None, query: str = "") -> Dict[str, Any]:
        """
        Evaluates a paper candidate using the selected backend (default: local).
        Guarantees reliable, realistic scores without random numbers.
        """
        # 1. Local Evaluator (Default & 100% offline)
        if self.backend == "local":
            return self.local_evaluator.evaluate(paper, query=query)

        # 2. Google Gemini Free Tier
        if self.backend == "gemini" and self.gemini_client.is_configured():
            gemini_res = self.gemini_client.evaluate_paper(paper, query=query)
            if gemini_res:
                return gemini_res
            return self.local_evaluator.evaluate(paper, query=query)

        # 3. GLM API Backend
        if self.backend == "glm" and self.glm_client.is_configured():
            try:
                active_filters = filter_settings or self.config.get("filters", {})
                messages = self._build_prompt(paper, active_filters, query)
                raw_output = self.glm_client.chat_completion(messages)
                parsed = self._parse_response(raw_output)
                if parsed and parsed.get("score"):
                    return parsed
            except Exception as e:
                print(f"[Evaluator] GLM call failed ({e}). Using local evaluator.")

        # Default fallback: Local high-precision evaluator
        return self.local_evaluator.evaluate(paper, query=query)

    def _build_prompt(self, paper: Dict[str, Any], filter_settings: Dict[str, Any], query: str = "") -> List[Dict[str, str]]:
        interests = filter_settings.get("interests", [
            "Market Microstructure, Limit Order Books & High-Frequency Trading",
            "Statistical Arbitrage, Machine Learning & Quantitative Alpha Signals",
            "Stochastic Volatility Models, Rough Volatility & Exotic Derivatives Pricing",
            "Portfolio Optimization, Factor Investing & Risk Parity"
        ])

        system_message = (
            "You are a quantitative research director at a systematic hedge fund. "
            "Evaluate research paper abstracts strictly for quantitative finance relevance, mathematical rigor, and empirical depth. "
            "Respond ONLY with valid JSON."
        )

        user_content = f"""
Target Search Query: {query or 'General Quantitative Finance'}
Title: {paper.get('title', 'Unknown')}
Category: {paper.get('category', 'Quantitative Finance')}
Abstract:
{paper.get('abstract', '')}

Output JSON with schema:
{{
  "is_quant_finance": <true or false>,
  "score": <integer from 1 to 100>,
  "hook": "<punchy 1-sentence hook>",
  "breakthrough_summary": "<2-3 sentences mathematical innovation>",
  "takeaway": "<1-2 sentences practical trading takeaway>",
  "concepts": ["<c1>", "<c2>", "<c3>"]
}}
"""
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        clean_text = raw_output.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict) and "score" in parsed:
                return parsed
        except Exception:
            pass

        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "score" in parsed:
                    return parsed
            except Exception:
                pass
        return {}
