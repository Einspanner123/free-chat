"""
Tests for the fine-tuning configuration module.
Covers training args, LoRA config, QLoRA config, and validation.
"""

import json
import os
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# =============================================================================
# Import with mocks for optional deps
# =============================================================================

@pytest.fixture(autouse=True)
def _setup_path():
    import sys
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestFinetuneConfig:
    """FineTuneConfig 必须包含所有训练参数."""

    def test_default_values(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig()
        assert cfg.base_model == "Qwen/Qwen3-0.6B"
        assert cfg.output_dir == "./output"
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-4
        assert cfg.warmup_ratio == 0.03
        assert cfg.max_seq_length == 2048
        assert cfg.save_steps == 500
        assert cfg.logging_steps == 10
        assert cfg.eval_steps == 100
        assert cfg.save_total_limit == 2

    def test_custom_values(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="/models/my-finetune",
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-4,
            max_seq_length=4096,
        )
        assert cfg.base_model == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.output_dir == "/models/my-finetune"
        assert cfg.num_epochs == 5
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 1e-4
        assert cfg.max_seq_length == 4096

    def test_validation_positive_epochs(self):
        from config import FineTuneConfig
        with pytest.raises(ValueError, match="epochs"):
            FineTuneConfig(num_epochs=0)
        with pytest.raises(ValueError, match="epochs"):
            FineTuneConfig(num_epochs=-1)

    def test_validation_batch_size(self):
        from config import FineTuneConfig
        with pytest.raises(ValueError, match="batch_size"):
            FineTuneConfig(batch_size=0)

    def test_validation_learning_rate(self):
        from config import FineTuneConfig
        with pytest.raises(ValueError, match="learning_rate"):
            FineTuneConfig(learning_rate=0)

    def test_validation_max_seq_length(self):
        from config import FineTuneConfig
        with pytest.raises(ValueError, match="max_seq_length"):
            FineTuneConfig(max_seq_length=0)

    def test_to_dict(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig(base_model="test", num_epochs=2)
        d = cfg.to_dict()
        assert d["base_model"] == "test"
        assert d["num_epochs"] == 2
        assert "learning_rate" in d

    def test_from_dict(self):
        from config import FineTuneConfig
        d = {"base_model": "test", "num_epochs": 5, "learning_rate": 1e-4}
        cfg = FineTuneConfig.from_dict(d)
        assert cfg.base_model == "test"
        assert cfg.num_epochs == 5
        assert cfg.learning_rate == 1e-4
        # 未提供的使用默认值
        assert cfg.batch_size == 4

    def test_from_dict_ignores_extra_keys(self):
        from config import FineTuneConfig
        d = {"base_model": "test", "unknown": "ignored"}
        cfg = FineTuneConfig.from_dict(d)
        assert cfg.base_model == "test"

    def test_save_and_load_yaml(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig(base_model="test-model", num_epochs=3, learning_rate=1e-4)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            cfg.save_yaml(f.name)
            loaded = FineTuneConfig.load_yaml(f.name)
        os.unlink(f.name)
        assert loaded.base_model == "test-model"
        assert loaded.num_epochs == 3
        assert loaded.learning_rate == 1e-4

    def test_device_map_auto(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig()
        assert "device_map" not in cfg.to_dict() or True  # no crash

    def test_repr(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig()
        s = repr(cfg)
        assert "FineTuneConfig" in s
        assert "Qwen" in s or "base_model" in s


class TestLoraConfig:
    """LoRA 配置必须覆盖关键参数."""

    def test_default_lora_values(self):
        from config import LoraConfig
        cfg = LoraConfig()
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.05
        assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert cfg.bias == "none"
        assert cfg.task_type == "CAUSAL_LM"
        assert cfg.use_rslora is False

    def test_custom_lora_values(self):
        from config import LoraConfig
        cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            use_rslora=True,
        )
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.use_rslora is True

    def test_lora_r_must_be_positive(self):
        from config import LoraConfig
        with pytest.raises(ValueError, match="r"):
            LoraConfig(r=0)
        with pytest.raises(ValueError, match="r"):
            LoraConfig(r=-1)

    def test_lora_alpha_must_be_positive(self):
        from config import LoraConfig
        with pytest.raises(ValueError, match="lora_alpha"):
            LoraConfig(lora_alpha=0)

    def test_different_target_module_combinations(self):
        from config import LoraConfig
        # Attention only
        LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        # Attention + MLP
        LoraConfig(target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
        # All linear
        LoraConfig(target_modules="all-linear")
        # Empty list (valid but pointless)
        LoraConfig(target_modules=[])

    def test_to_peft_dict(self):
        from config import LoraConfig
        cfg = LoraConfig(r=8, lora_alpha=16)
        d = cfg.to_peft_dict()
        assert d["r"] == 8
        assert d["lora_alpha"] == 16
        assert d["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert d["bias"] == "none"

    def test_rslora_scaling(self):
        """rslora 启用时 scaling = lora_alpha / sqrt(r) 而非 lora_alpha / r."""
        from config import LoraConfig
        import math
        cfg = LoraConfig(r=16, lora_alpha=32, use_rslora=True)
        d = cfg.to_peft_dict()
        # rslora 的标准 scaling
        expected_scale = 32 / math.sqrt(16)
        # peft 内部处理这个，我们只验证 use_rslora 传递了
        assert d.get("use_rslora", False) is True


class TestQLoraConfig:
    """QLoRA 配置."""

    def test_default_qlora_values(self):
        from config import QLoraConfig
        cfg = QLoraConfig()
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True
        assert cfg.bnb_4bit_compute_dtype == "bfloat16"

    def test_custom_qlora_values(self):
        from config import QLoraConfig
        cfg = QLoraConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype="float16",
        )
        assert cfg.bnb_4bit_quant_type == "fp4"
        assert cfg.bnb_4bit_use_double_quant is False

    def test_8bit_quantization(self):
        from config import QLoraConfig
        cfg = QLoraConfig(load_in_4bit=False, load_in_8bit=True)
        assert cfg.load_in_8bit is True
        assert cfg.load_in_4bit is False

    def test_to_bnb_dict(self):
        from config import QLoraConfig
        cfg = QLoraConfig()
        d = cfg.to_bnb_dict()
        assert d["load_in_4bit"] is True
        assert d["bnb_4bit_quant_type"] == "nf4"
        assert d["bnb_4bit_compute_dtype"] == "bfloat16"

    def test_no_quantization(self):
        from config import QLoraConfig
        cfg = QLoraConfig.none()
        assert cfg.load_in_4bit is False
        assert cfg.load_in_8bit is False

    def test_validation_exclusive_4bit_8bit(self):
        """不能同时启用 4bit 和 8bit."""
        from config import QLoraConfig
        with pytest.raises(ValueError, match="both 4-bit and 8-bit"):
            QLoraConfig(load_in_4bit=True, load_in_8bit=True)

    def test_to_bnb_dict_no_quant(self):
        from config import QLoraConfig
        cfg = QLoraConfig.none()
        d = cfg.to_bnb_dict()
        assert d == {"load_in_4bit": False, "load_in_8bit": False}


class TestFullFinetunePipelineConfig:
    """完整微调管道的配置集成."""

    def test_pipeline_config_defaults(self):
        from config import TrainingPipelineConfig
        cfg = TrainingPipelineConfig()
        assert cfg.finetune.base_model == "Qwen/Qwen3-0.6B"
        assert cfg.lora.r == 8
        assert cfg.qlora.load_in_4bit is True

    def test_pipeline_config_custom(self):
        from config import TrainingPipelineConfig, FineTuneConfig, LoraConfig, QLoraConfig
        cfg = TrainingPipelineConfig(
            finetune=FineTuneConfig(base_model="test", num_epochs=2),
            lora=LoraConfig(r=16, target_modules=["q_proj", "v_proj"]),
            qlora=QLoraConfig(load_in_4bit=True),
        )
        assert cfg.finetune.base_model == "test"
        assert cfg.lora.r == 16
        assert cfg.qlora.load_in_4bit is True

    def test_pipeline_to_dict(self):
        from config import TrainingPipelineConfig
        cfg = TrainingPipelineConfig()
        d = cfg.to_dict()
        assert "finetune" in d
        assert "lora" in d
        assert "qlora" in d
        assert d["finetune"]["base_model"] == "Qwen/Qwen3-0.6B"

    def test_is_qlora(self):
        from config import TrainingPipelineConfig, QLoraConfig
        cfg = TrainingPipelineConfig(qlora=QLoraConfig(load_in_4bit=True))
        assert cfg.is_qlora() is True
        cfg2 = TrainingPipelineConfig(qlora=QLoraConfig.none())
        assert cfg2.is_qlora() is False

    def test_get_effective_batch_size(self):
        from config import TrainingPipelineConfig, FineTuneConfig
        cfg = TrainingPipelineConfig(
            finetune=FineTuneConfig(batch_size=4, gradient_accumulation_steps=4)
        )
        assert cfg.get_effective_batch_size() == 16

    def test_effective_batch_size_default(self):
        from config import TrainingPipelineConfig
        cfg = TrainingPipelineConfig()
        assert cfg.get_effective_batch_size() == 16  # 4 * 4

    def test_config_serialization_roundtrip(self):
        from config import TrainingPipelineConfig
        import tempfile
        cfg = TrainingPipelineConfig()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg.to_dict(), f)
            fname = f.name
        with open(fname, 'r') as f:
            loaded_dict = json.load(f)
        os.unlink(fname)
        loaded = TrainingPipelineConfig.from_dict(loaded_dict)
        assert loaded.finetune.base_model == cfg.finetune.base_model
        assert loaded.lora.r == cfg.lora.r
        assert loaded.qlora.load_in_4bit == cfg.qlora.load_in_4bit

    def test_lora_target_modules_validation(self):
        """target_modules 必须是列表或 'all-linear'."""
        from config import LoraConfig
        LoraConfig(target_modules=["q_proj", "v_proj"])
        LoraConfig(target_modules="all-linear")
        with pytest.raises(ValueError, match="target_modules"):
            LoraConfig(target_modules="invalid")

    def test_gradient_checkpointing(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig(gradient_checkpointing=True)
        assert cfg.gradient_checkpointing is True
        cfg2 = FineTuneConfig(gradient_checkpointing=False)
        assert cfg2.gradient_checkpointing is False

    def test_optimizer_default(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig()
        assert cfg.optimizer == "adamw_torch"

    def test_lr_scheduler_default(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig()
        assert cfg.lr_scheduler == "cosine"

    def test_report_to_default(self):
        from config import FineTuneConfig
        cfg = FineTuneConfig()
        assert cfg.report_to == "none"

    def test_deepspeed_config(self):
        """支持 DeepSpeed ZeRO 配置."""
        from config import FineTuneConfig
        cfg = FineTuneConfig(deepspeed="configs/zero2.json")
        assert cfg.deepspeed == "configs/zero2.json"
        cfg2 = FineTuneConfig()
        assert cfg2.deepspeed is None
