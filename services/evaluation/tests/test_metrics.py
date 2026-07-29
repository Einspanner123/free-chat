"""Tests for metrics computation."""

import os
import sys
import math

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestAccuracy:
    def test_exact_match(self):
        from metrics import exact_match
        assert exact_match("Hello", "Hello") == 1.0
        assert exact_match("Hello", "World") == 0.0

    def test_exact_match_case_insensitive(self):
        from metrics import exact_match
        assert exact_match("hello", "HELLO", ignore_case=True) == 1.0

    def test_exact_match_strip(self):
        from metrics import exact_match
        assert exact_match("  hello  ", "hello") == 0.0  # not exact due to spaces
        assert exact_match("  hello  ", "hello", strip=True) == 1.0

    def test_exact_match_list(self):
        from metrics import exact_match
        predictions = ["A", "B", "C"]
        references = ["A", "B", "D"]
        assert exact_match(predictions, references) == 2 / 3

    def test_exact_match_empty(self):
        from metrics import exact_match
        assert exact_match([], []) == 1.0  # vacuous truth
        assert exact_match([], ["A"]) == 0.0


class TestF1Score:
    def test_perfect_f1(self):
        from metrics import f1_score
        pred = "The cat sat on the mat"
        ref = "The cat sat on the mat"
        assert f1_score(pred, ref) == 1.0

    def test_no_overlap(self):
        from metrics import f1_score
        pred = "abc def"
        ref = "ghi jkl"
        assert f1_score(pred, ref) == 0.0

    def test_partial_f1(self):
        from metrics import f1_score
        pred = "The cat sat"
        ref = "The dog ran"
        # shared: {"the"} → precision=1/3, recall=1/3 → f1=1/3
        expected = 2 * (1/3 * 1/3) / (1/3 + 1/3)
        assert abs(f1_score(pred, ref) - expected) < 1e-6

    def test_f1_list(self):
        from metrics import f1_score
        preds = ["A B", "C D"]
        refs = ["A B", "E F"]
        score = f1_score(preds, refs)
        assert 0 < score < 1

    def test_f1_empty_prediction(self):
        from metrics import f1_score
        assert f1_score("", "hello world") == 0.0

    def test_f1_empty_reference(self):
        from metrics import f1_score
        score = f1_score("hello", "")
        assert score == 0.0


class TestPassAtK:
    def test_pass_at_1(self):
        from metrics import pass_at_k
        # 1 correct out of 1 sample
        assert pass_at_k(1, 1, 1) == 1.0
        # 0 correct out of 1 sample
        assert pass_at_k(0, 1, 1) == 0.0

    def test_pass_at_k_multiple(self):
        from metrics import pass_at_k
        # n=3 samples, c=2 correct, k=1
        result = pass_at_k(2, 3, 1)
        expected = 1.0 - math.comb(3-2, 1) / math.comb(3, 1) if hasattr(math, 'comb') else 2/3
        # fallback: c/n = 2/3
        assert result > 0

    def test_pass_at_k_all_correct(self):
        from metrics import pass_at_k
        assert pass_at_k(5, 5, 3) == 1.0

    def test_pass_at_k_none_correct(self):
        from metrics import pass_at_k
        assert pass_at_k(0, 5, 3) == 0.0

    def test_pass_at_k_larger_k(self):
        from metrics import pass_at_k
        # c=2, n=5, k=5 → always pass
        assert pass_at_k(2, 5, 5) == 1.0

    def test_pass_at_k_invalid_args(self):
        from metrics import pass_at_k
        with pytest.raises(ValueError):
            pass_at_k(-1, 5, 1)
        with pytest.raises(ValueError):
            pass_at_k(6, 5, 1)  # c > n


class TestROUGE:
    def test_rouge_1_perfect(self):
        from metrics import rouge_1
        pred = "the cat sat on the mat"
        ref = "the cat sat on the mat"
        result = rouge_1(pred, ref)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_rouge_1_no_overlap(self):
        from metrics import rouge_1
        result = rouge_1("abc def", "ghi jkl")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_rouge_1_partial(self):
        from metrics import rouge_1
        pred = "the cat"
        ref = "the dog"
        result = rouge_1(pred, ref)
        assert result["precision"] == 0.5  # 1/2
        assert result["recall"] == 0.5  # 1/2

    def test_rouge_l_perfect(self):
        from metrics import rouge_l
        pred = "the cat sat"
        ref = "the cat sat"
        result = rouge_l(pred, ref)
        assert result["f1"] == 1.0

    def test_rouge_l_no_overlap(self):
        from metrics import rouge_l
        result = rouge_l("abc", "xyz")
        assert result["f1"] == 0.0


class TestMetricAggregation:
    def test_average_metrics(self):
        from metrics import average_metrics
        results = [
            {"accuracy": 0.8, "f1": 0.7},
            {"accuracy": 0.6, "f1": 0.5},
        ]
        avg = average_metrics(results)
        assert avg["accuracy"] == 0.7
        assert avg["f1"] == 0.6

    def test_average_empty(self):
        from metrics import average_metrics
        assert average_metrics([]) == {}

    def test_confidence_interval(self):
        from metrics import confidence_interval
        values = [0.8, 0.7, 0.9, 0.85, 0.75]
        ci = confidence_interval(values)
        assert "mean" in ci
        assert "ci_95" in ci
        assert "std" in ci
        assert abs(ci["mean"] - 0.8) < 0.05
