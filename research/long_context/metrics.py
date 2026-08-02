"""
Metrics for evaluating long-context understanding in small models.

Covers:
- Needle-in-a-Haystack accuracy (position-dependent)
- Entity recall (information preservation under compression)
- Compression-recall tradeoff curve (AUC)
- Position bias analysis (primacy/recency effects)
"""

import re
from typing import Dict, List, Tuple


def needle_accuracy(response: str, needle: str) -> float:
    """Check if the model's response contains the target needle.

    Args:
        response: Model-generated text.
        needle: The fact we inserted into the context.

    Returns:
        1.0 if needle found in response, 0.0 otherwise.
    """
    return 1.0 if needle.lower() in response.lower() else 0.0


def entity_recall(original: str, compressed: str) -> float:
    """Fraction of named entities from the original that survive compression.

    Entities are extracted using simple heuristics:
    capitalized words, proper names, known entity types.

    Args:
        original: Original uncompressed text.
        compressed: Text after compression.

    Returns:
        Fraction of entities from original that appear in compressed.
    """
    orig_entities = _extract_entities(original)
    if not orig_entities:
        return 1.0
    comp_entities = _extract_entities(compressed)
    retained = sum(1 for e in orig_entities if e.lower() in compressed.lower())
    return retained / len(orig_entities)


def _extract_entities(text: str) -> List[str]:
    """Extract named entities from text using pattern matching."""
    if not text:
        return []

    entities = set()
    # Extract capitalized words/phrases (potential proper nouns)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        word = match.group(1)
        # Filter out common false positives
        if word.lower() not in {'the', 'this', 'that', 'these', 'those',
                                  'when', 'where', 'what', 'why', 'how',
                                  'which', 'there', 'their', 'they',
                                  'then', 'than', 'also', 'very', 'just',
                                  'first', 'second', 'next', 'last',
                                  'some', 'each', 'every', 'both',
                                  'here', 'there', 'one', 'two'}:
            entities.add(word)

    return list(entities)


def compute_position_recall(results: List[Dict]) -> Dict:
    """Compute recall broken down by position in context.

    Args:
        results: List of {"position": float (0-1), "correct": bool}.

    Returns:
        Dict with overall, front_half, back_half recall.
    """
    if not results:
        return {"overall": 0.0, "front_half": 0.0, "back_half": 0.0}

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    front = [r for r in results if r.get("position", 0) <= 0.5]
    back = [r for r in results if r.get("position", 0) > 0.5]

    return {
        "overall": correct / total,
        "front_half": sum(1 for r in front if r["correct"]) / max(len(front), 1),
        "back_half": sum(1 for r in back if r["correct"]) / max(len(back), 1),
    }


def compute_position_bias(results: List[Dict]) -> Dict:
    """Measure primacy/recency bias from position-dependent accuracy.

    Positive bias_score = primacy bias (front performs better).
    Negative bias_score = recency bias (back performs better).

    Args:
        results: List of {"position": float (0-1), "correct": bool}.

    Returns:
        Dict with front_accuracy, back_accuracy, bias_score.
    """
    if not results:
        return {"front_accuracy": 0.0, "back_accuracy": 0.0, "bias_score": 0.0}

    front = [r for r in results if r.get("position", 0) <= 0.5]
    back = [r for r in results if r.get("position", 0) > 0.5]
    front_acc = sum(1 for r in front if r["correct"]) / max(len(front), 1)
    back_acc = sum(1 for r in back if r["correct"]) / max(len(back), 1)

    return {
        "front_accuracy": front_acc,
        "back_accuracy": back_acc,
        "bias_score": front_acc - back_acc,
    }


def compression_tradeoff(results: List[Dict]) -> Dict:
    """Compute the compression-recall tradeoff curve.

    Args:
        results: List of {"compression_ratio": float (0-1), "recall": float (0-1)}.

    Returns:
        Dict with auc (area under curve), tradeoff_points.
    """
    if not results:
        return {"auc": 0.0, "points": []}

    sorted_results = sorted(results, key=lambda x: x["compression_ratio"])
    auc = 0.0
    for i in range(1, len(sorted_results)):
        x_diff = sorted_results[i]["compression_ratio"] - sorted_results[i-1]["compression_ratio"]
        y_avg = (sorted_results[i]["recall"] + sorted_results[i-1]["recall"]) / 2
        auc += x_diff * y_avg

    return {
        "auc": round(auc, 4),
        "points": [(r["compression_ratio"], r["recall"]) for r in sorted_results],
    }
