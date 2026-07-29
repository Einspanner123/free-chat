"""Tests for preference data loading and construction."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestPreferenceFormats:
    """偏好数据格式加载."""

    def get_loader(self):
        from preference_data import PreferenceDataLoader
        return PreferenceDataLoader()

    def _pair(self, chosen="great", rejected="bad"):
        return {
            "chosen": [{"role": "user", "content": "How are you?"},
                       {"role": "assistant", "content": chosen}],
            "rejected": [{"role": "user", "content": "How are you?"},
                         {"role": "assistant", "content": rejected}],
        }

    def test_standard_pair_format(self):
        """标准的 chosen/rejected pair 格式."""
        data = [self._pair()]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result) == 1
        pair = result[0]
        assert "chosen" in pair
        assert "rejected" in pair
        assert len(pair["chosen"]) == 2
        assert len(pair["rejected"]) == 2
        assert pair["chosen"][-1]["content"] == "great"

    def test_multiple_pairs(self):
        data = [self._pair(chosen="a", rejected="b") for _ in range(5)]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result) == 5

    def test_pair_with_system_prompt(self):
        """带 system prompt 的偏好对."""
        data = [{
            "system": "Be helpful.",
            "chosen": [{"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "Hello!"}],
            "rejected": [{"role": "user", "content": "Hi"},
                         {"role": "assistant", "content": "Go away."}],
        }]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result) == 1
        pair = result[0]
        # system prompt should be prepended to both
        assert pair["chosen"][0]["role"] == "system"
        assert pair["rejected"][0]["role"] == "system"

    def test_missing_chosen_raises(self):
        data = [{"rejected": [{"role": "user", "content": "hi"}]}]
        loader = self.get_loader()
        with pytest.raises(ValueError, match="chosen"):
            loader.load_standard(data)

    def test_missing_rejected_raises(self):
        data = [{"chosen": [{"role": "user", "content": "hi"}]}]
        loader = self.get_loader()
        with pytest.raises(ValueError, match="rejected"):
            loader.load_standard(data)

    def test_empty_messages_skipped(self):
        data = [{"chosen": [], "rejected": []}]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result) == 0

    def test_trailing_user_message_trimmed(self):
        """chosen/rejected 必须以 assistant 结尾，否则截断."""
        data = [{
            "chosen": [{"role": "user", "content": "Q1"},
                       {"role": "assistant", "content": "A1"},
                       {"role": "user", "content": "Q2"}],
            "rejected": [{"role": "user", "content": "Q1"},
                         {"role": "assistant", "content": "A1_bad"}],
        }]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result[0]["chosen"]) == 2  # Q2 trimmed
        assert result[0]["chosen"][-1]["role"] == "assistant"


class TestPreferenceValidation:
    """偏好数据验证."""

    def get_loader(self):
        from preference_data import PreferenceDataLoader
        return PreferenceDataLoader()

    def test_chosen_rejected_same_prompt(self):
        """chosen 和 rejected 应该有相同的 user message 前缀."""
        data = [{
            "chosen": [{"role": "user", "content": "Same Q"},
                       {"role": "assistant", "content": "Good answer"}],
            "rejected": [{"role": "user", "content": "Same Q"},
                         {"role": "assistant", "content": "Bad answer"}],
        }]
        loader = self.get_loader()
        result = loader.load_standard(data)
        # Should validate that prompts match
        pair = result[0]
        assert pair["chosen"][0]["content"] == pair["rejected"][0]["content"]

    def test_different_prompts_warn(self):
        """chosen 和 rejected 的 user message 不同时应发出警告或修复."""
        data = [{
            "chosen": [{"role": "user", "content": "Q1"},
                       {"role": "assistant", "content": "A_good"}],
            "rejected": [{"role": "user", "content": "Q2"},
                         {"role": "assistant", "content": "A_bad"}],
        }]
        loader = self.get_loader()
        result = loader.load_standard(data)
        # 如果 prompts 不同，应丢弃该条目
        assert len(result) == 0

    def test_chosen_equal_to_rejected(self):
        """chosen == rejected 时跳过."""
        data = [{
            "chosen": [{"role": "user", "content": "Q"},
                       {"role": "assistant", "content": "Same answer"}],
            "rejected": [{"role": "user", "content": "Q"},
                         {"role": "assistant", "content": "Same answer"}],
        }]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result) == 0

    def test_multi_turn_preference(self):
        """多轮偏好：整个对话作为 prompt，最后一条 assistant 消息比较."""
        data = [{
            "chosen": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."},
                {"role": "user", "content": "Tell me more"},
                {"role": "assistant", "content": "Sure, AI can..."},
            ],
            "rejected": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."},
                {"role": "user", "content": "Tell me more"},
                {"role": "assistant", "content": "I don't know."},
            ],
        }]
        loader = self.get_loader()
        result = loader.load_standard(data)
        assert len(result) == 1
        # prompt = 前3条消息 (user/assistant/user), chosen/rejected = 最后一条
        pair = result[0]
        assert len(pair["chosen"]) == 4
        assert len(pair["rejected"]) == 4


class TestPreferenceDataLoading:
    """从文件加载偏好数据."""

    def get_loader(self):
        from preference_data import PreferenceDataLoader
        return PreferenceDataLoader()

    def test_load_json_file(self):
        loader = self.get_loader()
        data = [{
            "chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Good"}],
            "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Bad"}],
        }]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            fname = f.name
        result = loader.load_file(fname)
        os.unlink(fname)
        assert len(result) == 1

    def test_load_jsonl_file(self):
        loader = self.get_loader()
        entries = [
            {"chosen": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "Good1"}],
             "rejected": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "Bad1"}]},
            {"chosen": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "Good2"}],
             "rejected": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "Bad2"}]},
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            fname = f.name
        result = loader.load_file(fname)
        os.unlink(fname)
        assert len(result) == 2

    def test_file_not_found(self):
        loader = self.get_loader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent.json")

    def test_empty_file_returns_empty(self):
        loader = self.get_loader()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            fname = f.name
        result = loader.load_file(fname)
        os.unlink(fname)
        assert result == []


class TestPreferenceStatistics:
    """偏好数据统计."""

    def get_loader(self):
        from preference_data import PreferenceDataLoader
        return PreferenceDataLoader()

    def test_basic_statistics(self):
        data = [
            {"chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Good"}],
             "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Bad"}]},
        ]
        loader = self.get_loader()
        stats = loader.compute_statistics(data)
        assert stats["num_pairs"] == 1
        assert stats["total_messages"] == 4

    def test_statistics_with_token_counts(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 102, 103]

        data = [
            {"chosen": [{"role": "user", "content": "Hello"},
                       {"role": "assistant", "content": "World"}],
             "rejected": [{"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Bad"}]},
        ]
        loader = self.get_loader()
        stats = loader.compute_statistics(data, tokenizer=mock_tokenizer)
        assert stats["avg_chosen_length"] > 0
        assert stats["avg_rejected_length"] > 0
