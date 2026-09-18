"""
Evaluates research paper abstracts using GLM, scoring their intellectual interest,
novelty, and relevance to user criteria.
"""

import json
import re
from typing import Dict, Any, List
from .glm_client import GLMClient

class PaperEvaluator:
    def __init__(self, glm_client: GLMClient = None, config: Dict[str, Any] = None):
        self.glm_client = glm_client or GLMClient()
        self.config = config or {}

    def _build_prompt(self, paper: Dict[str, Any], filter_settings: Dict[str, Any]) -> List[Dict[str, str]]:
        interests = filter_settings.get("interests", [
            "Market Microstructure, Limit Order Books & High-Frequency Trading",
            "Statistical Arbitrage, Machine Learning & Quantitative Alpha Signals",
            "Stochastic Volatility Models, Rough Volatility & Exotic Derivatives Pricing",
            "Portfolio Optimization, Factor Investing & Risk Parity",
            "Reinforcement Learning for Trade Execution and Market Making"
        ])
        criteria = filter_settings.get("preferred_criteria", [
            "Novel quantitative alpha, mathematical rigor, and realistic no-arbitrage conditions",
            "Groundbreaking insights into market microstructure or order flow dynamics",
            "Innovative computational algorithms for high-dimensional portfolio or risk modeling"
        ])

        system_message = (
            "You are an elite quantitative researcher and director of quantitative research at a premier systematic hedge fund. "
            "Your goal is to evaluate quantitative finance research paper abstracts and select the most intellectually captivating, "
            "mathematically rigorous, and practically profound papers for a quantitative trading and research team.\n\n"
            "Evaluate each paper strictly based on mathematical novelty, theoretical validity, microstructure insight, alpha potential, "
            "and alignment with quantitative finance interests. Disregard simplistic backtests or trivial overfitted claims.\n"
            "Respond ONLY with a valid JSON object matching the required schema. Do not include markdown wraps like ```json or explanations outside the JSON."
        )

        user_content = f"""
Quant Team Research Interests:
{chr(10).join(f"- {i}" for i in interests)}

Key Evaluation Criteria:
{chr(10).join(f"- {c}" for c in criteria)}

Paper to Evaluate:
Title: {paper.get('title', 'Unknown')}
Authors: {', '.join(paper.get('authors', [])[:5])}
Published Date: {paper.get('published_date', 'Unknown')}
Category: {paper.get('category', 'Quantitative Finance')}
Abstract:
{paper.get('abstract', '')}

Evaluate this paper and output JSON with this exact structure:
{{
  "score": <integer from 1 to 100 representing overall quantitative depth, alpha novelty, and intellectual interest>,
  "hook": "<a punchy, fascinating 1-sentence hook explaining why a quantitative researcher must read this>",
  "breakthrough_summary": "<2-3 clear sentences explaining the mathematical innovation, market model, or statistical finding>",
  "takeaway": "<1-2 sentences explaining practical trading, pricing, risk management, or execution implications>",
  "concepts": ["<quant_concept1>", "<quant_concept2>", "<quant_concept3>", "<quant_concept4>", "<quant_concept5>"]
}}
"""
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

    def evaluate(self, paper: Dict[str, Any], filter_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Sends the paper abstract to GLM and parses the JSON evaluation.
        """
        active_filters = filter_settings or self.config.get("filters", {})
        messages = self._build_prompt(paper, active_filters)
        
        raw_output = self.glm_client.chat_completion(messages)
        return self._parse_response(raw_output)

    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        """Safely parses GLM output into a dictionary."""
        # Strip potential markdown code block markers
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
            # Validate essential fields
            return {
                "score": int(parsed.get("score", 70)),
                "hook": str(parsed.get("hook", "Fascinating new scientific paper.")),
                "breakthrough_summary": str(parsed.get("breakthrough_summary", "")),
                "takeaway": str(parsed.get("takeaway", "")),
                "concepts": [str(c) for c in parsed.get("concepts", []) if isinstance(c, str)],
                "model_name": self.glm_client.model
            }
        except json.JSONDecodeError:
            # Fallback regex extraction if model returned conversational preface
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return {
                        "score": int(parsed.get("score", 70)),
                        "hook": str(parsed.get("hook", "Fascinating new scientific paper.")),
                        "breakthrough_summary": str(parsed.get("breakthrough_summary", "")),
                        "takeaway": str(parsed.get("takeaway", "")),
                        "concepts": [str(c) for c in parsed.get("concepts", []) if isinstance(c, str)],
                        "model_name": self.glm_client.model
                    }
                except Exception:
                    pass

            return {
                "score": 70,
                "hook": "Intriguing study with notable findings.",
                "breakthrough_summary": clean_text[:250],
                "takeaway": "Worth reviewing for relevant domain insights.",
                "concepts": ["research", "science"],
                "model_name": self.glm_client.model
            }
