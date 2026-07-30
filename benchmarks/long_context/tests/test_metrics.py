"""Tests for long-context recall metrics."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


class TestNeedleHaystack:
    def test_needle_correct(self):
        from metrics import needle_accuracy
        # Model response contains the needle
        assert needle_accuracy("The secret password is 42.", "password is 42") == 1.0

    def test_needle_wrong(self):
        from metrics import needle_accuracy
        assert needle_accuracy("I don't know.", "password is 42") == 0.0

    def test_needle_partial(self):
        from metrics import needle_accuracy
        # Partial match: "password" appears but not "42"
        assert needle_accuracy("The password is secret.", "password is 42") == 0.0

    def test_needle_multi_position(self):
        from metrics import compute_position_recall
        results = [
            {"position": 0.1, "correct": True},
            {"position": 0.3, "correct": True},
            {"position": 0.5, "correct": False},
            {"position": 0.7, "correct": True},
            {"position": 0.9, "correct": False},
        ]
        recall = compute_position_recall(results)
        assert recall["overall"] == 0.6  # 3/5
        assert 0.0 <= recall["front_half"] <= 1.0
        assert 0.0 <= recall["back_half"] <= 1.0


class TestEntityRecall:
    def test_entity_recall_perfect(self):
        from metrics import entity_recall
        original = "Alice went to Paris with Bob. They visited the Eiffel Tower."
        compressed = "Alice went to Paris with Bob. They visited the Eiffel Tower."
        assert entity_recall(original, compressed) == 1.0

    def test_entity_recall_partial(self):
        from metrics import entity_recall
        original = "Alice went to Paris with Bob. They visited the Eiffel Tower."
        compressed = "Someone went to Paris with Bob."
        # Entities: Alice, Paris, Bob, Eiffel Tower
        # Retained: Paris, Bob
        recall = entity_recall(original, compressed)
        assert 0.4 < recall < 0.6

    def test_entity_recall_none(self):
        from metrics import entity_recall
        original = "Alice and Bob in Paris."
        compressed = "Nothing about them."
        assert entity_recall(original, compressed) == 0.0

    def test_entity_recall_empty(self):
        from metrics import entity_recall
        assert entity_recall("", "") == 1.0


class TestCompressionTradeoff:
    def test_tradeoff_curve(self):
        from metrics import compression_tradeoff
        results = [
            {"compression_ratio": 0.0, "recall": 1.0},
            {"compression_ratio": 0.5, "recall": 0.9},
            {"compression_ratio": 0.8, "recall": 0.7},
            {"compression_ratio": 0.95, "recall": 0.3},
        ]
        curve = compression_tradeoff(results)
        assert curve["auc"] > 0  # Area under curve
        assert abs(curve["auc"]) <= 1.0

    def test_tradeoff_empty(self):
        from metrics import compression_tradeoff
        assert compression_tradeoff([])["auc"] == 0.0


class TestPositionBias:
    def test_position_bias_compute(self):
        from metrics import compute_position_bias
        # Simulate: earlier positions have higher accuracy (primacy effect)
        results = [
            {"position": 0.1, "correct": True},
            {"position": 0.2, "correct": True},
            {"position": 0.8, "correct": False},
            {"position": 0.9, "correct": False},
        ]
        bias = compute_position_bias(results)
        assert bias["front_accuracy"] > bias["back_accuracy"]
        assert bias["front_accuracy"] >= 0
        assert bias["bias_score"] > 0  # positive = primacy bias

    def test_no_bias(self):
        from metrics import compute_position_bias
        results = [
            {"position": 0.1, "correct": True},
            {"position": 0.9, "correct": True},
        ]
        bias = compute_position_bias(results)
        assert bias["bias_score"] == 0.0


class TestBenchmarkRunner:
    def test_config_defaults(self):
        from run import LongContextBenchConfig
        cfg = LongContextBenchConfig()
        assert 0 < cfg.context_length <= 32768
        assert cfg.num_needles >= 1

    def test_generate_context(self):
        from run import generate_and_insert
        context, needles = generate_and_insert(length=1024, num_needles=3, seed=42)
        assert len(context) >= 1000
        assert len(needles) == 3
        for n in needles:
            assert n["needle"] in context
