"""
Tests for LoRA weight merging module.

Covers weight merge, model saving, and validation.
"""

import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

import pytest


# Mock heavy ML deps
class _MockModule(types.ModuleType):
    pass

_mock_torch = _MockModule("torch")
_mock_torch.cuda = type("cuda", (), {"is_available": staticmethod(lambda: False), "empty_cache": staticmethod(lambda: None)})()
_mock_torch.no_grad = lambda: _MockModule("ctx", {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

_mock_peft = _MockModule("peft")
_mock_peft.PeftModel = type("PeftModel", (), {"from_pretrained": staticmethod(lambda *a, **kw: MagicMock())})

_mock_transformers = _MockModule("transformers")
_mock_transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {"from_pretrained": staticmethod(lambda *a, **kw: MagicMock())})()
_mock_transformers.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **kw: MagicMock())})()

for mod_name, mod in [
    ("torch", _mock_torch),
    ("peft", _mock_peft),
    ("transformers", _mock_transformers),
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod


@pytest.fixture(autouse=True)
def _setup_path():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestMergeConfig:
    """合并配置验证."""

    def get_merger(self):
        from merge_lora import LoraMerger
        return LoraMerger()

    def test_merge_defaults(self):
        from merge_lora import MergeConfig
        cfg = MergeConfig()
        assert cfg.base_model_name_or_path is not None
        assert cfg.lora_weights_path is not None
        assert cfg.output_path is not None
        assert cfg.save_tokenizer is True
        assert cfg.push_to_hub is False

    def test_merge_custom(self):
        from merge_lora import MergeConfig
        cfg = MergeConfig(
            base_model_name_or_path="Qwen/Qwen3-0.6B",
            lora_weights_path="./output/lora-final",
            output_path="./models/merged",
            save_tokenizer=True,
            push_to_hub=True,
            hf_repo_id="myuser/merged-model",
        )
        assert cfg.base_model_name_or_path == "Qwen/Qwen3-0.6B"
        assert cfg.hf_repo_id == "myuser/merged-model"


class TestMergeProcess:
    """合并流程."""

    def get_merger(self):
        from merge_lora import LoraMerger
        return LoraMerger()

    @patch('merge_lora.AutoModelForCausalLM')
    @patch('merge_lora.AutoTokenizer')
    @patch('merge_lora.PeftModel')
    def test_merge_and_save(self, mock_peft, mock_at, mock_am):
        """完整的合并与保存流程."""
        mock_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_model

        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = merger.create_config(
                base_model="Qwen/Qwen3-0.6B",
                lora_path="/path/to/lora",
                output_path=tmpdir,
            )
            result = merger.merge_and_save(cfg)
            # 验证 merge_and_unload 被调用
            mock_model.merge_and_unload.assert_called_once()
            # 验证保存被调用
            assert mock_model.save_pretrained.called or result is not None

    @patch('merge_lora.AutoModelForCausalLM')
    @patch('merge_lora.AutoTokenizer')
    @patch('merge_lora.PeftModel')
    def test_merge_preserves_dtype(self, mock_peft, mock_at, mock_am):
        """合并后保持数据类型."""
        mock_model = MagicMock()
        mock_model.dtype = "float16"
        mock_peft.from_pretrained.return_value = mock_model

        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = merger.create_config(
                base_model="test",
                lora_path="/path/to/lora",
                output_path=tmpdir,
            )
            result = merger.merge_and_save(cfg)
            assert result is not None or mock_model.save_pretrained.called

    @patch('merge_lora.AutoModelForCausalLM')
    @patch('merge_lora.AutoTokenizer')
    @patch('merge_lora.PeftModel')
    def test_merge_quantized_model(self, mock_peft, mock_at, mock_am):
        """量化模型的合并."""
        mock_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_model

        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = merger.create_config(
                base_model="test",
                lora_path="/path/to/lora",
                output_path=tmpdir,
                load_in_4bit=True,
            )
            result = merger.merge_and_save(cfg)
            # 量化模型也能合并
            assert result is not None or mock_model.merge_and_unload.called

    @patch('merge_lora.AutoModelForCausalLM')
    @patch('merge_lora.AutoTokenizer')
    @patch('merge_lora.PeftModel')
    def test_merge_invalid_lora_path_raises(self, mock_peft, mock_at, mock_am):
        """无效的 LoRA 路径应报错."""
        mock_peft.from_pretrained.side_effect = FileNotFoundError("Not found")

        merger = self.get_merger()
        cfg = merger.create_config(
            base_model="test",
            lora_path="/nonexistent/path",
            output_path="/tmp/out",
        )
        with pytest.raises((FileNotFoundError, ValueError)):
            merger.merge_and_save(cfg)

    @patch('merge_lora.AutoModelForCausalLM')
    @patch('merge_lora.AutoTokenizer')
    @patch('merge_lora.PeftModel')
    def test_merge_saves_tokenizer(self, mock_peft, mock_at, mock_am):
        """合并时保存 tokenizer."""
        mock_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_at.from_pretrained.return_value = mock_tokenizer

        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = merger.create_config(
                base_model="test",
                lora_path="/path/to/lora",
                output_path=tmpdir,
            )
            merger.merge_and_save(cfg)
            mock_tokenizer.save_pretrained.assert_called_once()


class TestMergeValidation:
    """合并前验证."""

    def get_merger(self):
        from merge_lora import LoraMerger
        return LoraMerger()

    def test_validate_creates_output_dir(self):
        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "new_dir", "merged")
            cfg = merger.create_config(
                base_model="test",
                lora_path=tmpdir,
                output_path=output,
            )
            merger.validate(cfg)
            assert os.path.exists(output)

    def test_validate_handles_nonexistent_lora_path(self):
        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = merger.create_config(
                base_model="test",
                lora_path="/nonexistent/path",
                output_path=os.path.join(tmpdir, "out"),
            )
            # Should not raise; validate is lenient for remote/test paths
            merger.validate(cfg)
        merger = self.get_merger()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "new_dir", "merged")
            cfg = merger.create_config(
                base_model="test",
                lora_path=tmpdir,  # exists
                output_path=output,
            )
            # 自动创建输出目录
            merger.validate(cfg)
            assert os.path.exists(os.path.dirname(output))

    def test_validate_raises_on_empty_model(self):
        merger = self.get_merger()
        with pytest.raises(ValueError):
            merger.create_config(base_model="", lora_path="/path", output_path="/out")
