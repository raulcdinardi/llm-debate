"""
Modular reward heuristics for summarization.

Each function is independent and can be composed via RewardConfig.
Training uses one reward, evaluation can compute multiple metrics.
"""

from dataclasses import dataclass, field


def compression_reward(summary: str, article: str, _highlights: str) -> float:
    """Reward based on compression ratio. Shorter summaries score higher.
    
    Score = min(1.0, target_ratio / actual_ratio) where:
    - target_ratio = 0.1 (10:1 compression target)
    - actual_ratio = len(summary) / len(article)
    
    Returns 0-1 range. Degenerate empty summaries get 0.
    """
    if len(summary) == 0 or len(article) == 0:
        return 0.0
    
    actual_ratio = len(summary) / len(article)
    target_ratio = 0.1  # aim for 10:1 compression
    
    if actual_ratio < target_ratio:
        # Too short might be degenerate, penalize slightly
        return max(0.0, actual_ratio / target_ratio)
    
    return min(1.0, target_ratio / actual_ratio)


def rouge_reward(summary: str, _article: str, highlights: str) -> float:
    """ROUGE-L F1 score against reference highlights.
    
    Requires rouge_score package. Returns 0 if package unavailable.
    """
    if len(summary) == 0 or len(highlights) == 0:
        return 0.0
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        if not getattr(rouge_reward, "_warned", False):
            print("[WARNING] rouge_score package not installed, rouge_reward returns 0")
            rouge_reward._warned = True
        return 0.0
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(highlights, summary)
    return scores["rougeL"].fmeasure


def tfidf_similarity(summary: str, article: str, _highlights: str) -> float:
    """TF-IDF cosine similarity between summary and article.
    
    Measures how well summary captures important terms from article.
    Requires scikit-learn. Returns 0 if unavailable.
    """
    if len(summary) == 0 or len(article) == 0:
        return 0.0
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        if not getattr(tfidf_similarity, "_warned", False):
            print("[WARNING] scikit-learn not installed, tfidf_similarity returns 0")
            tfidf_similarity._warned = True
        return 0.0
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([article, summary])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(similarity)


def embedding_similarity(summary: str, article: str, _highlights: str) -> float:
    """Embedding cosine similarity using sentence-transformers.
    
    Semantic similarity between summary and article embeddings.
    Requires sentence-transformers. Returns 0 if unavailable.
    """
    if len(summary) == 0 or len(article) == 0:
        return 0.0
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        if not getattr(embedding_similarity, "_warned", False):
            print("[WARNING] sentence-transformers not installed, embedding_similarity returns 0")
            embedding_similarity._warned = True
        return 0.0
    
    # Cache model to avoid reloading
    if not hasattr(embedding_similarity, "_model"):
        embedding_similarity._model = SentenceTransformer("all-MiniLM-L6-v2")
    
    model = embedding_similarity._model
    embeddings = model.encode([article, summary])
    sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return float(sim)


def length_penalty(summary: str, _article: str, _highlights: str, min_chars: int = 50, max_chars: int = 500) -> float:
    """Penalize summaries outside acceptable length range.
    
    Returns 1.0 if within range, decays linearly outside.
    """
    n = len(summary)
    if n < min_chars:
        return n / min_chars if min_chars > 0 else 0.0
    if n > max_chars:
        return max(0.0, 1.0 - (n - max_chars) / max_chars)
    return 1.0


def fluency_heuristic(summary: str, _article: str, _highlights: str) -> float:
    """Simple fluency proxy: sentence structure.
    
    Checks for reasonable sentence count and length.
    Not a true fluency measure, just a sanity check.
    """
    if len(summary) == 0:
        return 0.0
    
    # Count sentences (rough heuristic)
    sentences = [s.strip() for s in summary.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    
    if len(sentences) == 0:
        return 0.0
    
    # Ideal: 2-5 sentences, avg 15-30 words each
    sentence_count_score = min(1.0, len(sentences) / 2) if len(sentences) < 2 else (1.0 if len(sentences) <= 5 else max(0.5, 1.0 - (len(sentences) - 5) / 10))
    
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    avg_length_score = 1.0 if 10 <= avg_words <= 40 else max(0.0, 1.0 - abs(avg_words - 25) / 25)
    
    return (sentence_count_score + avg_length_score) / 2


# Registry of available reward functions
REWARD_FUNCTIONS = {
    "compression": compression_reward,
    "rouge": rouge_reward,
    "tfidf": tfidf_similarity,
    "embedding": embedding_similarity,
    "length_penalty": length_penalty,
    "fluency": fluency_heuristic,
}


@dataclass
class RewardConfig:
    """Configuration for composing multiple reward heuristics.
    
    Supports arbitrary linear combinations: reward = sum(weight_i * metric_i) / sum(weight_i)
    """
    
    weights: dict[str, float] = field(default_factory=lambda: {"compression": 1.0})
    
    def compute(self, summary: str, article: str, highlights: str) -> tuple[float, dict[str, float]]:
        """Compute weighted reward and individual metric scores.
        
        Returns:
            (total_reward, {metric_name: score})
        """
        scores = {}
        total = 0.0
        weight_sum = 0.0
        
        for name, weight in self.weights.items():
            fn = REWARD_FUNCTIONS[name]
            score = fn(summary, article, highlights)
            scores[name] = score
            total += weight * score
            weight_sum += weight
        
        if weight_sum > 0:
            total /= weight_sum
        
        return total, scores
    
    @classmethod
    def from_string(cls, spec: str) -> "RewardConfig":
        """Parse a reward spec string like 'compression:0.5,rouge:0.3,tfidf:0.2'.
        
        If no weights specified (e.g., 'compression'), assumes weight=1.0.
        """
        weights = {}
        for part in spec.split(","):
            part = part.strip()
            if ":" in part:
                name, weight = part.split(":", 1)
                weights[name.strip()] = float(weight.strip())
            else:
                weights[part] = 1.0
        return cls(weights=weights)
