"""
Domain Concept & Keyword Extractor for Quantitative Finance and Machine Learning.
Extracts clean, normalized, domain-relevant concepts from paper titles, abstracts, and categories.
"""

import re
from typing import List, Optional, Set

# High-salience quant finance & machine learning taxonomies and pattern matchers
QUANT_TAXONOMY = [
    # Market Microstructure & Trading
    ("market microstructure", r"\b(market microstructure|microstructure)\b"),
    ("limit order book", r"\b(limit order books?|lob|order books? dynamics|order books?)\b"),
    ("high-frequency trading", r"\b(high[- ]frequency trading|hft|ultra[- ]fast trading)\b"),
    ("algorithmic trading", r"\b(algorithmic trading|automated trading|algo trading)\b"),
    ("liquidity provision", r"\b(liquidity provision|market making|market maker|liquidity provider)\b"),
    ("price discovery", r"\b(price discovery|information incorporation)\b"),
    ("order flow toxicity", r"\b(order flow|order flow toxicity|vpin|informed trading)\b"),
    ("market impact", r"\b(market impact|slippage|permanent impact|temporary impact)\b"),
    ("optimal execution", r"\b(optimal execution|optimal liquidation|almgren[- ]chriss|execution algorithm|twap|vwap)\b"),
    ("transaction costs", r"\b(transaction costs?|tca|cost analysis)\b"),
    ("bid-ask spread", r"\b(bid[- ]ask spread|spread estimation|quoted spread)\b"),
    ("frequent batch auctions", r"\b(frequent batch auctions|batch auction|latency arbitrage)\b"),
    ("foreign exchange", r"\b(foreign exchange|fx market|currencies|exchange rates?)\b"),
    ("intraday dynamics", r"\b(intraday data|tick[- ]by[- ]tick|trade direction|order flow imbalance)\b"),
    ("electronic markets", r"\b(electronic communications? networks?|ecn|crossing networks?|dark pools?)\b"),

    # Volatility & Derivatives
    ("stochastic volatility", r"\b(stochastic volatility|rough volatility|heston|sabr|local volatility)\b"),
    ("option pricing", r"\b(option pricing|options pricing|derivative pricing|contingent claim)\b"),
    ("volatility surface", r"\b(volatility surface|volatility smile|implied volatility)\b"),
    ("jump diffusion", r"\b(jump[- ]diffusion|jump process|merton model)\b"),
    ("risk-neutral pricing", r"\b(risk[- ]neutral pricing|risk[- ]neutral measure|equivalent martingale)\b"),
    ("deep hedging", r"\b(deep hedging|neural hedging|hedging strategies?)\b"),
    ("american options", r"\b(american options?|optimal stopping|early exercise)\b"),
    ("exotic derivatives", r"\b(exotic derivatives?|barrier options?|asian options?)\b"),
    ("stochastic calculus", r"\b(malliavin calculus|stochastic calculus|ito calculus|martingale)\b"),
    ("monte carlo simulation", r"\b(monte carlo|quasi[- ]monte carlo|variance reduction)\b"),
    ("pde numerical methods", r"\b(partial differential equations?|finite difference|pde methods?)\b"),

    # Statistical Arbitrage & Factors
    ("statistical arbitrage", r"\b(statistical arbitrage|stat arb|pairs trading)\b"),
    ("mean reversion", r"\b(mean reversion|mean[- ]reverting|ou process|ornstein[- ]uhlenbeck)\b"),
    ("cointegration", r"\b(cointegration|cointegrated|engle[- ]granger|johansen)\b"),
    ("alpha factor design", r"\b(alpha factors?|alpha signals?|quantitative signals?)\b"),
    ("momentum strategies", r"\b(cross[- ]sectional momentum|time[- ]series momentum|trend following)\b"),
    ("multi-factor models", r"\b(multi[- ]factor models?|factor investing|fama[- ]french|carhart)\b"),
    ("asset pricing anomalies", r"\b(cross[- ]sectional|asset pricing|pricing anomal(y|ies))\b"),

    # Portfolio & Risk Management
    ("portfolio optimization", r"\b(portfolio optimization|asset allocation|portfolio selection)\b"),
    ("mean-variance framework", r"\b(mean[- ]variance|markowitz|efficient frontier)\b"),
    ("risk parity", r"\b(risk parity|equal risk contribution)\b"),
    ("black-litterman", r"\b(black[- ]litterman|bayesian portfolio)\b"),
    ("covariance estimation", r"\b(covariance estimation|shrinkage estimation|ledoit[- ]wolf|random matrix theory)\b"),
    ("hierarchical risk parity", r"\b(hierarchical risk parity|hrp|machine learning portfolio)\b"),
    ("value at risk & cvar", r"\b(value at risk|var|expected shortfall|cvar|tail risk)\b"),
    ("drawdown control", r"\b(drawdown|maximum drawdown|downside risk)\b"),
    ("credit risk modeling", r"\b(credit risk|default risk|cds|structural default)\b"),

    # Machine Learning & AI
    ("deep reinforcement learning", r"\b(deep reinforcement learning|reinforcement learning|deep rl|q[- ]learning|ppo|actor[- ]critic)\b"),
    ("deep learning", r"\b(deep learning|deep neural network|dnn)\b"),
    ("neural networks", r"\b(neural networks?|artificial neural network|mlp)\b"),
    ("transformers & llms", r"\b(transformers?|attention mechanism|large language models?|llms?|gpt)\b"),
    ("temporal networks & lstm", r"\b(recurrent neural network|rnn|lstm|gru|temporal convolution)\b"),
    ("graph neural networks", r"\b(graph neural networks?|gnn|graph convolutional)\b"),
    ("gaussian processes", r"\b(gaussian processes?|gaussian process regression|kernel methods?)\b"),
    ("tree-based ensembles", r"\b(gradient boosting|xgboost|lightgbm|random forests?)\b"),
    ("dimension reduction", r"\b(principal component analysis|pca|autoencoders?|tsne|umap)\b"),
    ("supervised learning", r"\b(supervised learning|scikit[- ]learn|classification|regression)\b"),
    ("distributed machine learning", r"\b(distributed machine learning|tensorflow|pytorch|large[- ]scale machine learning)\b"),

    # Financial Econometrics & Time Series
    ("financial econometrics", r"\b(financial econometrics|econometric analysis|empirical finance)\b"),
    ("garch & volatility models", r"\b(garch|egarch|gjr[- ]garch|volatility modeling)\b"),
    ("state-space & kalman filter", r"\b(kalman filter|state[- ]space|filter algorithms?)\b"),
    ("regime switching", r"\b(regime switching|hidden markov|markov switching)\b"),
    ("high-frequency data", r"\b(high[- ]frequency data|financial high frequency|realized volatility)\b"),
    ("vector autoregression", r"\b(vector autoregression|var model|granger causality)\b"),

    # Crypto & Decentralized Finance
    ("cryptocurrency markets", r"\b(cryptocurrenc(y|ies)|bitcoin|ethereum|crypto assets?)\b"),
    ("decentralized finance", r"\b(decentralized finance|defi|automated market maker|amm|uniswap)\b"),
]

# Category code mappings
CATEGORY_CONCEPTS = {
    "q-fin.tr": ["market microstructure", "optimal execution"],
    "q-fin.pr": ["option pricing", "stochastic calculus"],
    "q-fin.pm": ["portfolio optimization", "asset allocation"],
    "q-fin.st": ["statistical finance", "financial econometrics"],
    "q-fin.cp": ["computational finance", "monte carlo simulation"],
    "q-fin.mf": ["mathematical finance", "risk-neutral pricing"],
    "q-fin.rm": ["risk management", "value at risk & cvar"],
    "cs.lg": ["machine learning", "deep learning"],
    "cs.ai": ["artificial intelligence"],
    "stat.ml": ["machine learning", "statistical learning"],
}

STOPWORDS = {
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
    "methods", "results", "study", "using", "based", "proposed", "model", "models", "data",
    "show", "from", "onto", "within", "among", "across", "along", "behind", "beyond",
    "towards", "theorem", "treatment", "exact", "sample", "assumption", "assumptions",
    "given", "value", "values", "second", "third", "case", "cases", "analysis", "article",
    "present", "presents", "show", "shows", "provide", "provides", "investigate", "test",
    "system", "large", "scale", "learn", "learning", "rise", "direction", "issues", "applications",
    "markets", "financial", "empirical", "framework", "performance", "novel", "via"
}


def extract_domain_concepts(
    title: str,
    abstract: Optional[str] = None,
    category: Optional[str] = None,
    max_concepts: int = 5
) -> List[str]:
    """
    Extracts high-salience, normalized domain concepts from paper metadata.
    Balances specific taxonomy matching with distinctive n-gram phrase extraction.
    """
    text = f"{title or ''} {abstract or ''} {category or ''}".lower()
    matched_concepts: List[str] = []
    seen: Set[str] = set()

    # 1. Check taxonomy pattern matches
    for concept_name, pattern in QUANT_TAXONOMY:
        if re.search(pattern, text):
            if concept_name not in seen:
                matched_concepts.append(concept_name)
                seen.add(concept_name)
                if len(matched_concepts) >= max_concepts:
                    break

    # 2. Check category mapping if still room
    if len(matched_concepts) < max_concepts and category:
        cat_lower = category.lower().strip()
        for key, cat_concepts in CATEGORY_CONCEPTS.items():
            if key in cat_lower:
                for c in cat_concepts:
                    if c not in seen:
                        matched_concepts.append(c)
                        seen.add(c)
                        if len(matched_concepts) >= max_concepts:
                            break
            if len(matched_concepts) >= max_concepts:
                break

    # 3. Fallback: extract prominent bigrams from title if taxonomy matched few
    if len(matched_concepts) < 2 and title:
        title_words = re.findall(r"\b[a-zA-Z]{3,20}\b", title.lower())
        filtered_words = [w for w in title_words if w not in STOPWORDS]
        for i in range(len(filtered_words) - 1):
            phrase = f"{filtered_words[i]} {filtered_words[i+1]}"
            if phrase not in seen:
                matched_concepts.append(phrase)
                seen.add(phrase)
                if len(matched_concepts) >= max_concepts:
                    break

    # 4. Default guarantee if empty
    if not matched_concepts:
        matched_concepts = ["quantitative finance", "empirical finance"]

    return matched_concepts[:max_concepts]
