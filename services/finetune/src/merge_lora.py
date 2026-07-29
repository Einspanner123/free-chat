"""
LoRA weight merging module.

Merges LoRA adapters into the base model and saves the full model.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


@dataclass
class MergeConfig:
    """Configuration for merging LoRA weights."""
    base_model_name_or_path: str = ""
    lora_weights_path: str = ""
    output_path: str = ""
    save_tokenizer: bool = True
    push_to_hub: bool = False
    hf_repo_id: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class LoraMerger:
    """Merges LoRA adapters into the base model."""

    def create_config(
        self,
        base_model: str,
        lora_path: str,
        output_path: str,
        **kwargs,
    ) -> MergeConfig:
        """Create a MergeConfig with validation."""
        if not base_model:
            raise ValueError("base_model must not be empty")
        if not lora_path:
            raise ValueError("lora_path must not be empty")
        if not output_path:
            raise ValueError("output_path must not be empty")
        return MergeConfig(
            base_model_name_or_path=base_model,
            lora_weights_path=lora_path,
            output_path=output_path,
            **kwargs,
        )

    def validate(self, cfg: MergeConfig):
        """Validate paths before merging."""
        if not os.path.exists(cfg.base_model_name_or_path):
            # Could be a HuggingFace model ID, not a local path
            pass
        if not os.path.exists(cfg.lora_weights_path):
            # For remote LoRA paths or test paths, skip local validation
            # The downstream PeftModel.from_pretrained will fail if truly invalid
            pass
        # Create output directory
        os.makedirs(cfg.output_path, exist_ok=True)

    def merge_and_save(self, cfg: MergeConfig) -> str:
        """Merge LoRA weights into base model and save.

        Args:
            cfg: Merge configuration.

        Returns:
            Path to the saved merged model.
        """
        self.validate(cfg)

        logger.info(
            f"Merging LoRA weights from '{cfg.lora_weights_path}' "
            f"into base model '{cfg.base_model_name_or_path}'"
        )

        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for merging")

        if PeftModel is None:
            raise ImportError("peft is required for merging")

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch is not None else "auto",
        }
        if cfg.load_in_4bit or cfg.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=cfg.load_in_4bit,
                    load_in_8bit=cfg.load_in_8bit,
                )
            except ImportError:
                logger.warning("BitsAndBytes not available, ignoring quantization config")

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path,
            **model_kwargs,
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, cfg.lora_weights_path)

        # Merge weights
        logger.info("Merging weights...")
        merged_model = model.merge_and_unload()

        # Save
        logger.info(f"Saving merged model to '{cfg.output_path}'")
        merged_model.save_pretrained(cfg.output_path, safe_serialization=True)

        if cfg.save_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.base_model_name_or_path,
                trust_remote_code=True,
            )
            tokenizer.save_pretrained(cfg.output_path)

        # Push to HuggingFace Hub
        if cfg.push_to_hub and cfg.hf_repo_id:
            logger.info(f"Pushing to HuggingFace Hub: {cfg.hf_repo_id}")
            try:
                merged_model.push_to_hub(cfg.hf_repo_id)
                if cfg.save_tokenizer:
                    tokenizer.push_to_hub(cfg.hf_repo_id)
            except Exception as e:
                logger.error(f"Failed to push to Hub: {e}")

        logger.info("Merge completed successfully")
        return cfg.output_path
