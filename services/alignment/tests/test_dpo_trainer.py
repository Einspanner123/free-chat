"""Tests for DPO trainer.

Depends on conftest.py for ML dependency mocks.
"""

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


class TestDPOTrainerInit:
    def test_init_basic(self):
        from dpo_trainer import DPOTrainer
        trainer = DPOTrainer(base_model="test-model")
        assert trainer.base_model == "test-model"
        assert trainer.model is not None
        trainer.close()

    def test_init_with_config(self):
        from dpo_trainer import DPOTrainer
        from config import DPOPipelineConfig, DPOConfig
        cfg = DPOPipelineConfig(base_model="custom-model", dpo=DPOConfig(beta=0.3))
        trainer = DPOTrainer(config=cfg)
        assert trainer.config.base_model == "custom-model"
        assert trainer.config.dpo.beta == 0.3
        trainer.close()

    def test_init_reference_model(self):
        from dpo_trainer import DPOTrainer
        trainer = DPOTrainer(base_model="test-model")
        assert hasattr(trainer, "reference_model")
        trainer.close()


class TestDPOTrainerTraining:
    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_train_with_preference_data(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        with patch('dpo_trainer.DPOTrainer._create_trl_trainer') as mc:
            mt = MagicMock()
            mt.train.return_value = MagicMock()
            mt.train.return_value.metrics = {"train_loss": 0.5}
            mc.return_value = mt
            trainer = DPOTrainer(base_model="test-model")
            result = trainer.train([
                {"chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Good"}],
                 "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Bad"}]},
            ])
            mc.assert_called_once()

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_train_with_eval_data(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        with patch('dpo_trainer.DPOTrainer._create_trl_trainer') as mc:
            mc.return_value = MagicMock()
            trainer = DPOTrainer(base_model="test-model")
            trainer.train(
                [{"chosen": [], "rejected": []}],
                eval_data=[{"chosen": [], "rejected": []}],
            )
            mc.assert_called_once()

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_train_empty_data_raises(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        trainer = DPOTrainer(base_model="test-model")
        with pytest.raises(ValueError, match="data"):
            trainer.train([])
        trainer.close()

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_train_beta_parameter_passed(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        from config import DPOPipelineConfig, DPOConfig
        cfg = DPOPipelineConfig(base_model="test-model", dpo=DPOConfig(beta=0.5, num_epochs=2))
        with patch('dpo_trainer.DPOTrainer._create_trl_trainer') as mc:
            mc.return_value = MagicMock()
            trainer = DPOTrainer(config=cfg)
            trainer.train([{"chosen": [], "rejected": []}])
            assert mc.called

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_training_metrics_returned(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        with patch('dpo_trainer.DPOTrainer._create_trl_trainer') as mc:
            mt = MagicMock()
            mt.train.return_value.metrics = {"train_loss": 0.3}
            mc.return_value = mt
            trainer = DPOTrainer(base_model="test-model")
            result = trainer.train([{"chosen": [], "rejected": []}])
            assert "train_loss" in result or "metrics" in result


class TestDPOTrainerSaveLoad:
    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_save_policy(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        mock_trainer = MagicMock()
        trainer = DPOTrainer(base_model="test-model")
        trainer.trainer = mock_trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            mock_trainer.save_model.assert_called_once()

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_save_creates_directory(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        mock_trainer = MagicMock()
        trainer = DPOTrainer(base_model="test-model")
        trainer.trainer = mock_trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, "nested", "policy")
            trainer.save(deep_path)
            assert os.path.exists(deep_path)


class TestDPOTrainerEdgeCases:
    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_close_releases_resources(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        trainer = DPOTrainer(base_model="test-model")
        trainer.close()
        assert trainer._closed

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_context_manager(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        with DPOTrainer(base_model="test-model") as trainer:
            assert not trainer._closed
        assert trainer._closed

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_label_smoothing_in_loss(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        from config import DPOPipelineConfig, DPOConfig
        cfg = DPOPipelineConfig(base_model="test-model", dpo=DPOConfig(label_smoothing=0.1))
        trainer = DPOTrainer(config=cfg)
        assert trainer.config.dpo.label_smoothing == 0.1
        trainer.close()

    @patch('dpo_trainer.AutoModelForCausalLM')
    @patch('dpo_trainer.AutoTokenizer')
    def test_ipo_loss_variant(self, mock_at, mock_am):
        from dpo_trainer import DPOTrainer
        from config import DPOPipelineConfig, DPOConfig
        cfg = DPOPipelineConfig(base_model="test-model", dpo=DPOConfig(loss_type="ipo"))
        trainer = DPOTrainer(config=cfg)
        assert trainer.config.dpo.loss_type == "ipo"
        trainer.close()
