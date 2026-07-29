"""Tests for DPO alignment configuration."""

import os
import sys
import tempfile
import json

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestDPOConfig:
    def test_default_values(self):
        from config import DPOConfig
        cfg = DPOConfig()
        assert cfg.beta == 0.1  # DPO temperature parameter
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 5e-6
        assert cfg.max_length == 2048
        assert cfg.max_prompt_length == 1024

    def test_custom_values(self):
        from config import DPOConfig
        cfg = DPOConfig(beta=0.5, num_epochs=5, learning_rate=1e-5)
        assert cfg.beta == 0.5
        assert cfg.num_epochs == 5
        assert cfg.learning_rate == 1e-5

    def test_beta_must_be_positive(self):
        from config import DPOConfig
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(beta=0)
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(beta=-0.1)

    def test_reference_model_config(self):
        from config import DPOConfig
        cfg = DPOConfig(reference_model="Qwen/Qwen3-0.6B")
        assert cfg.reference_model == "Qwen/Qwen3-0.6B"

    def test_reference_model_defaults_to_base(self):
        """未指定时 reference_model = None，训练时使用 base model 的 frozen copy."""
        from config import DPOConfig
        cfg = DPOConfig()
        assert cfg.reference_model is None

    def test_dpo_loss_type(self):
        from config import DPOConfig
        cfg = DPOConfig(loss_type="sigmoid")
        assert cfg.loss_type == "sigmoid"
        cfg2 = DPOConfig(loss_type="ipo")
        assert cfg2.loss_type == "ipo"
        with pytest.raises(ValueError):
            DPOConfig(loss_type="invalid")

    def test_to_dict_roundtrip(self):
        from config import DPOConfig
        cfg = DPOConfig(beta=0.2, num_epochs=4)
        d = cfg.to_dict()
        loaded = DPOConfig.from_dict(d)
        assert loaded.beta == 0.2
        assert loaded.num_epochs == 4


class TestDPOPipelineConfig:
    def test_pipeline_config_defaults(self):
        from config import DPOPipelineConfig
        cfg = DPOPipelineConfig()
        assert cfg.dpo.beta == 0.1
        assert cfg.base_model is not None

    def test_pipeline_config_custom(self):
        from config import DPOPipelineConfig, DPOConfig
        cfg = DPOPipelineConfig(
            base_model="Qwen/Qwen3-0.6B",
            dpo=DPOConfig(beta=0.3),
        )
        assert cfg.dpo.beta == 0.3

    def test_to_dict(self):
        from config import DPOPipelineConfig
        cfg = DPOPipelineConfig()
        d = cfg.to_dict()
        assert "dpo" in d
        assert "base_model" in d

    def test_label_smoothing(self):
        from config import DPOConfig
        cfg = DPOConfig(label_smoothing=0.1)
        assert cfg.label_smoothing == 0.1
        cfg2 = DPOConfig()
        assert cfg2.label_smoothing == 0.0  # default: no smoothing
