"""
Smoke tests: end-to-end without GPU.

Tests the data pipeline end-to-end without requiring model training:
- Load data -> process -> format -> verify output structure
"""

import json
import os
import sys
import tempfile
import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestDataPipelineSmoke:
    """End-to-end data pipeline: load -> validate -> format -> statistics."""

    def test_full_data_pipeline_from_sharegpt(self):
        """Load ShareGPT JSON, process, compute stats, format for training."""
        from data_processor import DataProcessor

        # Create a realistic ShareGPT dataset
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "What is Python?"},
                    {"from": "gpt", "value": "Python is a programming language."},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Explain AI."},
                    {"from": "gpt", "value": "AI is a field of study."},
                    {"from": "human", "value": "Give an example."},
                    {"from": "gpt", "value": "Machine learning is an example of AI."},
                ]
            },
        ]

        dp = DataProcessor()

        # 1. Load
        processed = dp.load_sharegpt(data)
        assert len(processed) == 2

        # 2. Split
        train, eval_data = dp.split(processed, eval_ratio=0.5, seed=42)
        assert len(train) + len(eval_data) == 2

        # 3. Statistics
        stats = dp.compute_statistics(processed)
        assert stats["num_examples"] == 2
        assert stats["total_turns"] == 6  # 2 + 4 messages
        assert stats["role_distribution"]["user"] == 3
        assert stats["role_distribution"]["assistant"] == 3

        # 4. Format for training
        for entry in processed:
            messages = entry["messages"]
            formatted = dp.format_for_training(messages)
            assert "input" in formatted
            assert "output" in formatted

    def test_full_data_pipeline_from_file(self):
        """Load from JSON file -> process -> export to JSONL."""
        from data_processor import DataProcessor

        data = [
            {"instruction": "What is Python?", "input": "", "output": "A language."},
            {"instruction": "What is AI?", "input": "", "output": "A field."},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name

        try:
            dp = DataProcessor()
            processed = dp.load_file(fname, format="alpaca")
            assert len(processed) == 2

            stats = dp.compute_statistics(processed)
            assert stats["num_examples"] == 2
            assert stats["avg_input_tokens"] > 0
            assert stats["avg_output_tokens"] > 0
        finally:
            os.unlink(fname)

    def test_all_formats_roundtrip(self):
        """Load, save, reload in different formats."""
        from data_processor import DataProcessor

        dp = DataProcessor()

        # ShareGPT -> internal format
        sharegpt = dp.load_sharegpt([
            {"conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
            ]}
        ])
        assert len(sharegpt) == 1

        # Alpaca -> internal format
        alpaca = dp.load_alpaca([
            {"instruction": "Q1", "input": "", "output": "A1"}
        ])
        assert len(alpaca) == 1

        # ChatML -> internal format
        chatml = dp.load_chatml([
            {"messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ]}
        ])
        assert len(chatml) == 1

        # All formats should produce identical message structure
        for dataset in [sharegpt, alpaca, chatml]:
            entry = dataset[0]
            assert "messages" in entry
            assert len(entry["messages"]) == 2
            assert entry["messages"][0]["role"] in ("user", "system")
            assert entry["messages"][1]["role"] == "assistant"

    def test_dataset_split_reproducibility(self):
        """Split should produce identical results with same seed."""
        from data_processor import DataProcessor

        dp = DataProcessor()
        data = [{"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(100)]

        train1, eval1 = dp.split(data, eval_ratio=0.1, seed=123)
        train2, eval2 = dp.split(data, eval_ratio=0.1, seed=123)

        assert len(train1) == len(train2)
        assert len(eval1) == len(eval2)
        for e1, e2 in zip(eval1, eval2):
            assert e1["messages"][0]["content"] == e2["messages"][0]["content"]

    def test_format_preserves_message_structure(self):
        """Template conversion preserves all required fields."""
        from data_processor import DataProcessor
        dp = DataProcessor()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        # Format with fallback template (no tokenizer)
        result = dp.apply_template(messages)
        assert "Hello" in result
        assert "Hi there" in result
        assert "<|im_start|>assistant" in result
