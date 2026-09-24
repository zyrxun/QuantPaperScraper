"""
Client for Google Gemini Free API Tier (gemini-1.5-flash / gemini-2.0-flash).
Uses Google AI Studio free API key (no credit card required).
Endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
"""

import os
import json
import ssl
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional

def get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        return ssl.create_default_context()
    except Exception:
        return ssl._create_unverified_context()

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10 and not self.api_key.startswith("your_"))

    def evaluate_paper(self, paper: Dict[str, Any], query: str = "") -> Optional[Dict[str, Any]]:
        """Calls Gemini free API to evaluate a paper abstract into structured JSON."""
        if not self.is_configured():
            return None

        prompt = f"""You are a quantitative research director evaluating a research paper.
Topic Query: {query or 'Quantitative Finance'}

Paper to Evaluate:
Title: {paper.get('title', '')}
Category: {paper.get('category', '')}
Abstract: {paper.get('abstract', '')}

Evaluate strictly for quantitative finance relevance, mathematical novelty, and empirical depth.
Respond ONLY with a JSON object with this exact structure:
{{
  "is_quant_finance": <true or false>,
  "score": <integer from 1 to 100>,
  "hook": "<a punchy 1-sentence hook>",
  "breakthrough_summary": "<2-3 sentence mathematical & methodology summary>",
  "takeaway": "<1-2 sentence practical trading/execution implication>",
  "concepts": ["<concept1>", "<concept2>", "<concept3>"]
}}
"""

        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        ctx = get_ssl_context()

        try:
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                parsed = json.loads(text)
                parsed["model_name"] = self.model
                return parsed
        except Exception as e:
            print(f"[GeminiClient] Note: Gemini API call failed ({e}). Falling back to local evaluator.")
            return None