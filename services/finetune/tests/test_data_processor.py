"""
Tests for data processing module.

Covers data loading, format conversion (ShareGPT, Alpaca, ChatML),
multi-turn conversations, token counting, data splitting, and template application.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup_path():
    import sys
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestDataFormats:
    """不同数据格式的加载与转换."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_sharegpt_format_single_turn(self):
        """ShareGPT 格式：单轮对话."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "What is Python?"},
                    {"from": "gpt", "value": "Python is a programming language."}
                ]
            }
        ]
        dp = self.get_processor()
        result = dp.load_sharegpt(data)
        assert len(result) == 1
        entry = result[0]
        assert len(entry["messages"]) == 2
        assert entry["messages"][0]["role"] == "user"
        assert entry["messages"][0]["content"] == "What is Python?"
        assert entry["messages"][1]["role"] == "assistant"

    def test_sharegpt_format_multi_turn(self):
        """ShareGPT 格式：多轮对话."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            }
        ]
        dp = self.get_processor()
        result = dp.load_sharegpt(data)
        assert len(result) == 1
        assert len(result[0]["messages"]) == 4

    def test_sharegpt_with_system_prompt(self):
        """ShareGPT 格式：带 system prompt."""
        data = [
            {
                "system": "You are a helpful assistant.",
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"}
                ]
            }
        ]
        dp = self.get_processor()
        result = dp.load_sharegpt(data)
        assert len(result[0]["messages"]) == 3
        assert result[0]["messages"][0]["role"] == "system"

    def test_sharegpt_multiple_entries(self):
        """ShareGPT 格式：多条数据."""
        data = [
            {"conversations": [{"from": "human", "value": "Q1"}, {"from": "gpt", "value": "A1"}]},
            {"conversations": [{"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}]},
            {"conversations": [{"from": "human", "value": "Q3"}, {"from": "gpt", "value": "A3"}]},
        ]
        dp = self.get_processor()
        result = dp.load_sharegpt(data)
        assert len(result) == 3

    def test_alpaca_format(self):
        """Alpaca 格式."""
        data = [
            {"instruction": "Explain gravity.", "input": "", "output": "Gravity is a force..."}
        ]
        dp = self.get_processor()
        result = dp.load_alpaca(data)
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"
        assert "gravity" in result[0]["messages"][0]["content"].lower()

    def test_alpaca_format_with_input(self):
        """Alpaca 格式：带 input 字段."""
        data = [
            {"instruction": "Summarize", "input": "Long text here...", "output": "Summary."}
        ]
        dp = self.get_processor()
        result = dp.load_alpaca(data)
        assert len(result) == 1
        # instruction + input 都应包含在 user message 中
        user_msg = result[0]["messages"][0]["content"]
        assert "Summarize" in user_msg
        assert "Long text" in user_msg

    def test_chatml_format(self):
        """ChatML 格式."""
        data = [
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"}
                ]
            }
        ]
        dp = self.get_processor()
        result = dp.load_chatml(data)
        assert len(result) == 1
        assert len(result[0]["messages"]) == 3

    def test_chatml_raw_format(self):
        """ChatML 原生格式（与内部格式一致，直接通过）. """
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        ]
        dp = self.get_processor()
        result = dp.load_chatml(data)
        assert result[0]["messages"][0]["content"] == "Hello"


class TestDataValidation:
    """数据验证."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_empty_data(self):
        dp = self.get_processor()
        assert dp.load_sharegpt([]) == []
        assert dp.load_alpaca([]) == []

    def test_invalid_sharegpt_role(self):
        """不支持的 role 应过滤或报错."""
        data = [
            {"conversations": [
                {"from": "unknown_role", "value": "test"}
            ]}
        ]
        dp = self.get_processor()
        with pytest.raises(ValueError, match="role|from"):
            dp.load_sharegpt(data)

    def test_missing_value_field(self):
        """缺少 value 字段应报错."""
        data = [
            {"conversations": [
                {"from": "human", "content": "test"}
            ]}
        ]
        dp = self.get_processor()
        with pytest.raises(ValueError, match="value"):
            dp.load_sharegpt(data)

    def test_missing_conversations(self):
        """缺少 conversations 字段应报错."""
        data = [{"id": "123"}]
        dp = self.get_processor()
        with pytest.raises(ValueError, match="conversations"):
            dp.load_sharegpt(data)

    def test_odd_number_of_turns(self):
        """对话应以 assistant 结尾，否则丢弃最后一条."""
        data = [
            {"conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
                {"from": "human", "value": "Q2"},  # 没有对应的 assistant 回复
            ]}
        ]
        dp = self.get_processor()
        result = dp.load_sharegpt(data)
        # 应截断最后一条 human 消息
        assert len(result[0]["messages"]) == 2  # 只保留 Q1/A1

    def test_alpaca_no_output(self):
        """Alpaca 格式缺少 output 应丢弃."""
        data = [{"instruction": "test", "input": "", "output": ""}]
        dp = self.get_processor()
        result = dp.load_alpaca(data)
        assert len(result) == 0

    def test_empty_messages_in_entry(self):
        """消息内容为空的条目应跳过."""
        data = [
            {"conversations": [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""}
            ]}
        ]
        dp = self.get_processor()
        result = dp.load_sharegpt(data)
        assert len(result) == 0


class TestDataSplitting:
    """数据切分."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_train_eval_split_default(self):
        data = [{"messages": [{"role": "user", "content": f"test_{i}"}]} for i in range(100)]
        dp = self.get_processor()
        train, eval_data = dp.split(data, eval_ratio=0.1)
        assert len(train) == 90
        assert len(eval_data) == 10

    def test_train_eval_split_custom_ratio(self):
        data = [{"messages": [{"role": "user", "content": f"test_{i}"}]} for i in range(100)]
        dp = self.get_processor()
        train, eval_data = dp.split(data, eval_ratio=0.2)
        assert len(train) == 80
        assert len(eval_data) == 20

    def test_train_eval_split_no_eval(self):
        data = [{"messages": [{"role": "user", "content": "test"}]} for i in range(10)]
        dp = self.get_processor()
        train, eval_data = dp.split(data, eval_ratio=0.0)
        assert len(train) == 10
        assert len(eval_data) == 0

    def test_train_eval_split_seed_reproducibility(self):
        data = [{"messages": [{"role": "user", "content": f"test_{i}"}]} for i in range(100)]
        dp = self.get_processor()
        train1, eval1 = dp.split(data, eval_ratio=0.1, seed=42)
        train2, eval2 = dp.split(data, eval_ratio=0.1, seed=42)
        assert [e["messages"][0]["content"] for e in eval1] == [e["messages"][0]["content"] for e in eval2]

    def test_split_different_seeds_different(self):
        data = [{"messages": [{"role": "user", "content": f"test_{i}"}]} for i in range(100)]
        dp = self.get_processor()
        train1, eval1 = dp.split(data, eval_ratio=0.1, seed=42)
        train2, eval2 = dp.split(data, eval_ratio=0.1, seed=99)
        assert eval1 != eval2  # 大概率不同，但理论上可能相同

    def test_small_dataset_split(self):
        """小数据集切分."""
        data = [{"messages": [{"role": "user", "content": "test"}]} for i in range(2)]
        dp = self.get_processor()
        train, eval_data = dp.split(data, eval_ratio=0.5)
        assert len(train) >= 1
        assert len(eval_data) >= 0


class TestTemplateConversion:
    """模板转换."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_apply_chat_template_with_tokenizer(self):
        """使用 tokenizer 的 chat_template."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"

        messages = [{"role": "user", "content": "Hi"}]
        dp = self.get_processor()
        result = dp.apply_template(messages, tokenizer=mock_tokenizer)
        assert result == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_apply_chat_template_fallback(self):
        """无 tokenizer 时使用内置模板."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        dp = self.get_processor()
        result = dp.apply_template(messages)
        assert "<|im_start|>system" in result
        assert "Be helpful" in result
        assert "<|im_start|>assistant" in result

    def test_format_for_training(self):
        """训练格式：input 包含完整上下文（system + user），label 只包含 assistant 部分."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        dp = self.get_processor()
        formatted = dp.format_for_training(messages)
        assert "input" in formatted
        assert "output" in formatted or "labels" in formatted

    def test_format_for_training_multi_turn(self):
        """多轮对话训练格式."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        dp = self.get_processor()
        formatted = dp.format_for_training(messages)
        assert formatted is not None

    def test_format_for_training_system_prompt(self):
        """带 system prompt 的训练格式."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        dp = self.get_processor()
        formatted = dp.format_for_training(messages)
        assert formatted is not None
        # system prompt 应在 input 中
        assert "Be concise" in formatted.get("input", "")


class TestDataLoadingFromFiles:
    """从文件加载数据."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_load_json_file(self):
        """加载 JSON 文件（ShareGPT 格式）. """
        data = [
            {"conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"}
            ]}
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            fname = f.name
        dp = self.get_processor()
        result = dp.load_file(fname, format="sharegpt")
        os.unlink(fname)
        assert len(result) == 1

    def test_load_jsonl_file(self):
        """加载 JSONL 文件."""
        lines = [
            json.dumps({"conversations": [{"from": "human", "value": "Q1"}, {"from": "gpt", "value": "A1"}]}),
            json.dumps({"conversations": [{"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}]}),
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for line in lines:
                f.write(line + "\n")
            fname = f.name
        dp = self.get_processor()
        result = dp.load_file(fname, format="sharegpt")
        os.unlink(fname)
        assert len(result) == 2

    def test_auto_detect_format_from_filename(self):
        """根据文件名自动检测格式."""
        data = [
            {"instruction": "test", "input": "", "output": "response"}
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            fname = f.name
        dp = self.get_processor()
        # 显式指定 alpaca 格式
        result = dp.load_file(fname, format="alpaca")
        os.unlink(fname)
        assert len(result) == 1

    def test_file_not_found(self):
        dp = self.get_processor()
        with pytest.raises(FileNotFoundError):
            dp.load_file("/nonexistent/file.json", format="sharegpt")

    def test_invalid_format(self):
        dp = self.get_processor()
        with pytest.raises(ValueError, match="format"):
            dp.load_file("test.json", format="unknown_format")


class TestDatasetStatistics:
    """数据集统计."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_basic_statistics(self):
        data = [
            {"messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there!"}
            ]},
            {"messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm fine, thanks!"}
            ]}
        ]
        dp = self.get_processor()
        stats = dp.compute_statistics(data)
        assert stats["num_examples"] == 2
        assert stats["avg_messages_per_example"] == 2.0
        assert stats["total_turns"] == 4

    def test_statistics_with_tokenizer(self):
        """带 tokenizer 时统计 token 数."""
        data = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]}
        ]
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 102, 103]

        dp = self.get_processor()
        stats = dp.compute_statistics(data, tokenizer=mock_tokenizer)
        assert stats["avg_input_tokens"] > 0
        assert stats["avg_output_tokens"] > 0

    def test_statistics_empty_dataset(self):
        dp = self.get_processor()
        stats = dp.compute_statistics([])
        assert stats["num_examples"] == 0

    def test_statistics_role_distribution(self):
        data = [
            {"messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]}
        ]
        dp = self.get_processor()
        stats = dp.compute_statistics(data)
        assert stats.get("role_distribution", {}).get("system", 0) == 1
        assert stats["role_distribution"]["user"] == 1
        assert stats["role_distribution"]["assistant"] == 1


class TestDataSampling:
    """数据采样."""

    def get_processor(self):
        from data_processor import DataProcessor
        return DataProcessor()

    def test_sample_first_n(self):
        data = [{"messages": [{"role": "user", "content": f"test_{i}"}]} for i in range(100)]
        dp = self.get_processor()
        sampled = dp.sample(data, n=10)
        assert len(sampled) == 10

    def test_sample_more_than_available(self):
        data = [{"messages": [{"role": "user", "content": "test"}]} for i in range(5)]
        dp = self.get_processor()
        sampled = dp.sample(data, n=10)
        assert len(sampled) == 5

    def test_sample_reproducible(self):
        data = [{"messages": [{"role": "user", "content": f"test_{i}"}]} for i in range(100)]
        dp = self.get_processor()
        s1 = dp.sample(data, n=10, seed=42)
        s2 = dp.sample(data, n=10, seed=42)
        assert [e["messages"][0]["content"] for e in s1] == [e["messages"][0]["content"] for e in s2]

    def test_sample_zero(self):
        data = [{"messages": [{"role": "user", "content": "test"}]} for i in range(10)]
        dp = self.get_processor()
        sampled = dp.sample(data, n=0)
        assert len(sampled) == 0
