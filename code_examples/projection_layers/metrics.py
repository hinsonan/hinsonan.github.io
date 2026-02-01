"""Fast simplified metrics for image captioning."""
import math
import re
from collections import Counter
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    count = Counter(tokens)
    total = len(tokens) or 1
    return {token: cnt / total for token, cnt in count.items()}


def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    n_docs = len(documents) or 1
    idf = {}
    all_tokens = set(token for doc in documents for token in doc)
    for token in all_tokens:
        doc_count = sum(1 for doc in documents if token in doc)
        idf[token] = math.log(n_docs / (doc_count + 1e-10))
    return idf


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    keys = set(vec1.keys()) | set(vec2.keys())
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_cider_fast(predictions: List[str], references: List[List[str]]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    if not predictions:
        return 0.0

    pred_tokens = [_tokenize(p) for p in predictions]
    ref_tokens = [
        _tokenize(r)
        for ref_list in references
        for r in ref_list
    ]
    idf = _compute_idf(pred_tokens + ref_tokens)

    scores = []
    for pred, ref_list in zip(pred_tokens, references):
        pred_tf = _compute_tf(pred)
        pred_tfidf = {token: tf * idf.get(token, 0) for token, tf in pred_tf.items()}

        ref_scores = []
        for ref in ref_list:
            ref_toks = _tokenize(ref)
            ref_tf = _compute_tf(ref_toks)
            ref_tfidf = {token: tf * idf.get(token, 0) for token, tf in ref_tf.items()}
            ref_scores.append(_cosine_similarity(pred_tfidf, ref_tfidf))

        scores.append(sum(ref_scores) / len(ref_scores))

    return (sum(scores) / len(scores)) * 10


def _stem_word(word: str) -> str:
    for suffix in ("ing", "ly", "ed", "ies", "ied", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def compute_meteor_fast(predictions: List[str], references: List[List[str]]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    if not predictions:
        return 0.0

    scores = []
    for pred, ref_list in zip(predictions, references):
        pred_tokens = [_stem_word(t) for t in _tokenize(pred)]
        pred_set = set(pred_tokens)
        best_score = 0.0

        for ref in ref_list:
            ref_tokens = [_stem_word(t) for t in _tokenize(ref)]
            ref_set = set(ref_tokens)
            matches = len(pred_set & ref_set)
            if matches == 0:
                continue

            precision = matches / len(pred_tokens) if pred_tokens else 0.0
            recall = matches / len(ref_tokens) if ref_tokens else 0.0
            if precision + recall == 0:
                continue

            f_mean = (10 * precision * recall) / (9 * precision + recall)
            frag_penalty = 0.5 * (matches / len(pred_tokens)) if pred_tokens else 0.0
            score = f_mean * (1 - frag_penalty)
            best_score = max(best_score, score)

        scores.append(best_score)

    return sum(scores) / len(scores)


class MetricsTracker:
    """Accumulate predictions and references for validation metrics."""

    def __init__(self) -> None:
        self.predictions: List[str] = []
        self.references: List[List[str]] = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, predictions: List[str], references: List[List[str]], loss: float = 0.0) -> None:
        self.predictions.extend(predictions)
        self.references.extend(references)
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {"cider": 0.0, "meteor": 0.0, "loss": 0.0}

        cider = compute_cider_fast(self.predictions, self.references)
        meteor = compute_meteor_fast(self.predictions, self.references)
        avg_loss = self.total_loss / self.num_batches if self.num_batches else 0.0

        return {
            "cider": round(cider, 4),
            "meteor": round(meteor, 4),
            "loss": round(avg_loss, 4),
        }
