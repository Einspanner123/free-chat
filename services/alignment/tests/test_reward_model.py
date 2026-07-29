"""Tests for reward model training.

Depends on conftest.py for ML dependency mocks.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestRewardModelArchitecture:
    def test_reward_model_init(self):
        from reward_model import RewardModel, RewardModelConfig
        cfg = RewardModelConfig(base_model="test-model")
        model = RewardModel(cfg)
        assert model.config.base_model == "test-model"
        assert model.reward_head is not None
        model.close()

    def test_reward_head_dimensions(self):
        from reward_model import RewardModel, RewardModelConfig
        cfg = RewardModelConfig(base_model="test-model", hidden_size=4096)
        model = RewardModel(cfg)
        assert hasattr(model, "reward_head")
        model.close()

    def test_reward_model_custom_config(self):
        from reward_model import RewardModel, RewardModelConfig
        cfg = RewardModelConfig(base_model="test-model", hidden_size=1024, dropout=0.2)
        model = RewardModel(cfg)
        assert model.config.hidden_size == 1024
        assert model.config.dropout == 0.2
        model.close()


class TestRewardModelTraining:
    def test_train_on_preference_data(self):
        from reward_model import RewardModelTrainer
        trainer = RewardModelTrainer(base_model="test-model")
        result = trainer.train([
            {"chosen": "Good response", "rejected": "Bad response"},
        ])
        assert "accuracy" in result
        trainer.close()

    def test_train_empty_data_raises(self):
        from reward_model import RewardModelTrainer
        trainer = RewardModelTrainer(base_model="test-model")
        with pytest.raises(ValueError, match="data"):
            trainer.train([])
        trainer.close()


class TestRewardModelInference:
    def test_score_pair(self):
        from reward_model import RewardModel, RewardModelConfig
        cfg = RewardModelConfig(base_model="test-model")
        model = RewardModel(cfg)
        cs, rs = model.score_pair("Good answer", "Bad answer")
        assert cs is not None
        assert rs is not None
        model.close()

    def test_score_single_text(self):
        from reward_model import RewardModel, RewardModelConfig
        cfg = RewardModelConfig(base_model="test-model")
        model = RewardModel(cfg)
        score = model.score("Hello world")
        assert isinstance(score, (int, float))
        model.close()

    def test_preference_accuracy(self):
        from reward_model import RewardModel, RewardModelConfig
        cfg = RewardModelConfig(base_model="test-model")
        model = RewardModel(cfg)
        s1 = model.score("Good")
        s2 = model.score("Bad")
        assert isinstance(s1, (int, float))
        assert isinstance(s2, (int, float))
        model.close()
