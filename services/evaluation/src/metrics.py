"""Metrics computation for LLM evaluation."""

import math
from typing import List, Union, Dict, Optional


# =============================================================================
# Exact Match
# =============================================================================

def exact_match(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
    ignore_case: bool = False,
    strip: bool = False,
) -> float:
    """Compute exact match accuracy.

    Args:
        predictions: Single prediction string or list.
        references: Single reference string or list.
        ignore_case: If True, compare lowercased.
        strip: If True, strip whitespace.

    Returns:
        Accuracy score (0.0 to 1.0).
    """
    if isinstance(predictions, str) and isinstance(references, str):
        predictions = [predictions]
        references = [references]

    if len(predictions) != len(references):
        if len(predictions) == 0:
            return 0.0
        raise ValueError("predictions and references must have same length")

    if not predictions:
        return 1.0  # vacuous truth

    correct = 0
    for p, r in zip(predictions, references):
        if strip:
            p = p.strip()
            r = r.strip()
        if ignore_case:
            p = p.lower()
            r = r.lower()
        if p == r:
            correct += 1

    return correct / len(predictions)


# =============================================================================
# F1 Score
# =============================================================================

def f1_score(predictions: Union[str, List[str]], references: Union[str, List[str]]) -> float:
    """Compute token-level F1 score.

    Args:
        predictions: Single prediction string or list.
        references: Single reference string or list.

    Returns:
        F1 score (0.0 to 1.0).
    """
    if isinstance(predictions, str) and isinstance(references, str):
        predictions = [predictions]
        references = [references]

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    if not predictions:
        return 1.0

    total_f1 = 0.0
    for p, r in zip(predictions, references):
        p_tokens = set(p.lower().split())
        r_tokens = set(r.lower().split())

        if not p_tokens and not r_tokens:
            total_f1 += 1.0
            continue
        if not p_tokens or not r_tokens:
            total_f1 += 0.0
            continue

        intersection = p_tokens & r_tokens
        precision = len(intersection) / len(p_tokens)
        recall = len(intersection) / len(r_tokens)

        if precision + recall == 0:
            total_f1 += 0.0
        else:
            total_f1 += 2 * precision * recall / (precision + recall)

    return total_f1 / len(predictions)


# =============================================================================
# Pass@K
# =============================================================================

def pass_at_k(c: int, n: int, k: int) -> float:
    """Compute pass@k metric.

    pass@k = 1 - C(n-c, k) / C(n, k)
    where c is number of correct samples, n is total samples, k is passes.

    Args:
        c: Number of correct samples.
        n: Total number of samples.
        k: Number of passes (k in pass@k).

    Returns:
        pass@k score (0.0 to 1.0).
    """
    if c < 0 or n < 1 or k < 1:
        raise ValueError("c >= 0, n >= 1, k >= 1 required")
    if c > n:
        raise ValueError("c cannot exceed n")
    if k > n:
        k = n  # Clamp

    if c == 0:
        return 0.0
    if c == n:
        return 1.0

    try:
        import math
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)
    except (AttributeError, OverflowError):
        # Fallback for Python < 3.8 or large numbers
        return c / n


# =============================================================================
# ROUGE (simplified unigram-based)
# =============================================================================

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def rouge_1(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1 (unigram overlap).

    Returns:
        Dict with precision, recall, f1.
    """
    p_tokens = _tokenize(prediction)
    r_tokens = _tokenize(reference)

    if not p_tokens or not r_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p_bag = _bag(p_tokens)
    r_bag = _bag(r_tokens)

    overlap = sum(min(p_bag.get(t, 0), r_bag.get(t, 0)) for t in set(p_tokens) | set(r_tokens))

    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L (longest common subsequence).

    Returns:
        Dict with precision, recall, f1.
    """
    p_tokens = _tokenize(prediction)
    r_tokens = _tokenize(reference)

    if not p_tokens or not r_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = _lcs_length(p_tokens, r_tokens)

    precision = lcs_len / len(p_tokens)
    recall = lcs_len / len(r_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _bag(tokens: List[str]) -> Dict[str, int]:
    result = {}
    for t in tokens:
        result[t] = result.get(t, 0) + 1
    return result


def _lcs_length(a: List[str], b: List[str]) -> int:
    """DP-based LCS length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# =============================================================================
# Aggregation
# =============================================================================

def average_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across multiple runs.

    Args:
        results: List of metric dicts.

    Returns:
        Dict with averaged metrics.
    """
    if not results:
        return {}
    keys = results[0].keys()
    avg = {}
    for key in keys:
        values = [r[key] for r in results if key in r]
        if values:
            avg[key] = sum(values) / len(values)
    return avg


def confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Compute mean and confidence interval.

    Args:
        values: List of metric values.
        confidence: Confidence level (default 0.95).

    Returns:
        Dict with mean, std, ci_95.
    """
    import statistics
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_95": 0.0}

    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    # 95% CI using normal approximation
    z = 1.96
    ci = z * std / (n ** 0.5)

    return {"mean": mean, "std": std, "ci_95": ci}
