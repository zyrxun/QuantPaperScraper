"""
PDF text extraction, BPE tokenization, and concept extraction.
Tokenizes full PDFs to record token counts, section footprints, and graph keywords.
"""

import os
import re
from typing import Dict, Any, List, Tuple
from collections import Counter

# Try importing tiktoken, fallback to BPE estimation if unavailable
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Try importing pypdf, fallback to basic text extraction
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

class PDFTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self.encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                try:
                    self.encoder = tiktoken.get_encoding("gpt2")
                except Exception:
                    self.encoder = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts readable text from all pages of a PDF."""
        if not os.path.exists(pdf_path):
            return ""

        if PYPDF_AVAILABLE:
            try:
                reader = pypdf.PdfReader(pdf_path)
                full_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
                return "\n\n".join(full_text)
            except Exception as e:
                print(f"[PDFTokenizer] pypdf error on {pdf_path}: {e}")

        # Fallback basic extraction for unreadable or non-pypdf environments
        try:
            with open(pdf_path, "rb") as f:
                raw_bytes = f.read()
            # Extract ascii text streams
            ascii_text = re.findall(rb"[\x20-\x7E\s]{4,}", raw_bytes)
            return "\n".join(t.decode("latin1", errors="ignore") for t in ascii_text)
        except Exception:
            return ""

    def count_tokens(self, text: str) -> int:
        """Counts exact tokens using tiktoken (cl100k_base) or fallback word estimation."""
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Standard heuristic: 1 token ~= 4 characters or ~0.75 words
        return int(len(text.split()) * 1.33)

    def extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        """Extracts prominent domain-specific keywords and scientific terms for graph linking."""
        stopwords = {
            "the", "of", "and", "in", "to", "a", "is", "that", "for", "it", "as", "was",
            "with", "be", "by", "on", "not", "he", "i", "this", "are", "or", "an", "they",
            "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
            "we", "him", "been", "has", "when", "who", "will", "more", "no", "if", "out",
            "so", "said", "what", "up", "its", "about", "into", "than", "them", "can",
            "only", "other", "new", "some", "could", "time", "these", "two", "may", "then",
            "do", "first", "any", "my", "now", "such", "like", "our", "over", "man", "me",
            "even", "most", "made", "after", "also", "did", "many", "before", "must", "through",
            "back", "years", "where", "much", "your", "way", "well", "down", "should", "because",
            "each", "just", "those", "people", "mr", "how", "too", "little", "state", "good",
            "very", "make", "world", "still", "own", "see", "men", "work", "long", "get", "here",
            "between", "both", "life", "being", "under", "never", "day", "same", "another", "know",
            "while", "last", "might", "us", "great", "old", "year", "off", "come", "since",
            "against", "go", "came", "right", "used", "take", "three", "himself", "few", "house",
            "use", "during", "without", "again", "place", "american", "around", "however", "home",
            "small", "found", "mrs", "thought", "went", "say", "part", "once", "general", "high",
            "upon", "school", "every", "don", "does", "got", "united", "left", "number", "course",
            "war", "until", "always", "away", "something", "fact", "water", "though", "public",
            "less", "et", "al", "fig", "figure", "table", "section", "paper", "approach", "method",
            "methods", "results", "study", "using", "based", "proposed", "model", "models", "data", "show",
            "from", "into", "onto", "within", "among", "across", "along", "behind", "beyond", "during",
            "towards", "theorem", "treatment", "exact", "sample", "samples", "assumption", "assumptions",
            "given", "value", "values", "second", "third", "case", "cases", "analysis", "article",
            "present", "presents", "show", "shows", "provide", "provides", "investigate", "test",
            "defined", "proof", "lemma", "proposition", "corollary", "definition", "equation", "formula"
        }

        # Find words of length 4 to 25
        words = re.findall(r"\b[a-zA-Z]{4,25}\b", text.lower())
        filtered = [w for w in words if w not in stopwords]
        
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(top_k)]

    def tokenize_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extracts full text, chunks sections, counts tokens, and extracts key terms.
        """
        full_text = self.extract_text_from_pdf(pdf_path)
        total_tokens = self.count_tokens(full_text)

        # Estimate section token counts
        sections = {
            "Abstract": "",
            "Introduction": "",
            "Methods/Architecture": "",
            "Results/Discussion": ""
        }

        lower_text = full_text.lower()
        intro_pos = lower_text.find("introduction")
        method_pos = lower_text.find("method") if "method" in lower_text else lower_text.find("approach")
        results_pos = lower_text.find("result") if "result" in lower_text else lower_text.find("experiment")
        conclusion_pos = lower_text.find("conclusion") if "conclusion" in lower_text else lower_text.find("discussion")

        # Rough segmenting
        if intro_pos != -1:
            sections["Abstract"] = full_text[:intro_pos]
            if method_pos != -1 and method_pos > intro_pos:
                sections["Introduction"] = full_text[intro_pos:method_pos]
                if results_pos != -1 and results_pos > method_pos:
                    sections["Methods/Architecture"] = full_text[method_pos:results_pos]
                    sections["Results/Discussion"] = full_text[results_pos:conclusion_pos if conclusion_pos > results_pos else len(full_text)]

        section_breakdown = {
            sec: self.count_tokens(txt) for sec, txt in sections.items() if txt
        }

        keywords = self.extract_keywords(full_text, top_k=8)

        return {
            "total_tokens": total_tokens,
            "section_breakdown": section_breakdown,
            "extracted_keywords": keywords
        }
