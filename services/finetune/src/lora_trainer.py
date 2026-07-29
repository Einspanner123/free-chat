"""
LoRA/QLoRA fine-tuning trainer.

Provides a high-level API for parameter-efficient fine-tuning
using the PEFT library and HuggingFace Transformers.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from loguru import logger

# [Optional] Imports that may not be available in all environments
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None
    TorchDataset = object

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = object

try:
    from peft import LoraConfig as PefLoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
except ImportError:
    PefLoraConfig = object
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PeftModel = None

try:
    from trl import SFTTrainer
except ImportError:
    SFTTrainer = None

from config import TrainingPipelineConfig, FineTuneConfig, LoraConfig, QLoraConfig


class LoraTrainer:
    """High-level LoRA/QLoRA fine-tuning trainer."""

    def __init__(
        self,
        base_model: str,
        config: Optional[TrainingPipelineConfig] = None,
    ):
        self.base_model = base_model
        self.config = config or TrainingPipelineConfig(
            finetune=FineTuneConfig(base_model=base_model)
        )
        self._closed = False
        self.model = None
        self.tokenizer = None
        self.trainer = None

        self._load_model()

    def _load_model(self):
        """Load base model and apply LoRA/QLoRA configuration."""
        finetune_cfg = self.config.finetune
        lora_cfg = self.config.lora
        qlora_cfg = self.config.qlora

        logger.info(
            f"Loading model '{finetune_cfg.base_model}' "
            f"(QLoRA={self.config.is_qlora()})"
        )

        # Build model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch is not None else "auto",
        }

        # Apply quantization for QLoRA
        if self.config.is_qlora():
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(**qlora_cfg.to_bnb_dict())
                model_kwargs["quantization_config"] = bnb_config
            except ImportError:
                logger.warning("BitsAndBytes not available, skipping quantization config")

        # Load model
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required for LoraTrainer")
        self.model = AutoModelForCausalLM.from_pretrained(
            finetune_cfg.base_model,
            **model_kwargs,
        )

        # Load tokenizer
        if AutoTokenizer is None:
            raise ImportError("transformers is required for LoraTrainer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            finetune_cfg.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model for k-bit training if QLoRA
        if self.config.is_qlora() and prepare_model_for_kbit_training is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        if PefLoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for LoraTrainer")
        peft_config = PefLoraConfig(**lora_cfg.to_peft_dict())
        self.model = get_peft_model(self.model, peft_config)

        # Enable gradient checkpointing if configured
        if finetune_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model loaded and LoRA applied successfully")

    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run fine-tuning.

        Args:
            train_data: List of training entries with "messages" key.
            eval_data: Optional list of evaluation entries.
            resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            Training metrics dict.
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        if SFTTrainer is None:
            raise ImportError("trl is required for SFTTrainer")

        finetune_cfg = self.config.finetune

        # Build training arguments
        training_args = TrainingArguments(
            output_dir=finetune_cfg.output_dir,
            num_train_epochs=finetune_cfg.num_epochs,
            per_device_train_batch_size=finetune_cfg.batch_size,
            gradient_accumulation_steps=finetune_cfg.gradient_accumulation_steps,
            learning_rate=finetune_cfg.learning_rate,
            warmup_ratio=finetune_cfg.warmup_ratio,
            logging_steps=finetune_cfg.logging_steps,
            save_steps=finetune_cfg.save_steps,
            eval_steps=finetune_cfg.eval_steps,
            save_total_limit=finetune_cfg.save_total_limit,
            report_to=finetune_cfg.report_to,
            optim=finetune_cfg.optimizer,
            lr_scheduler_type=finetune_cfg.lr_scheduler,
            fp16=torch.cuda.is_available() if torch is not None else False,
            bf16=False,
            gradient_checkpointing=finetune_cfg.gradient_checkpointing,
            deepspeed=finetune_cfg.deepspeed,
            remove_unused_columns=False,
        )

        # Build SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            max_seq_length=finetune_cfg.max_seq_length,
            dataset_text_field="messages",
        )

        # Train
        logger.info("Starting training...")
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted, saving checkpoint...")
            self.trainer.save_model()
            return {"interrupted": True}

        # Save final model
        self.trainer.save_model()

        metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
        logger.info(f"Training completed: {metrics}")
        return metrics

    def save(self, path: str):
        """Save LoRA weights.

        Args:
            path: Output directory path.
        """
        os.makedirs(path, exist_ok=True)
        if self.trainer is not None:
            self.trainer.save_model(path)
        elif self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        logger.info(f"LoRA weights saved to {path}")

    def load(self, path: str):
        """Load pre-trained LoRA weights.

        Args:
            path: Path to LoRA weights directory.
        """
        if PeftModel is None:
            raise ImportError("peft is required to load LoRA weights")
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA weights loaded from {path}")

    def close(self):
        """Release resources."""
        self._closed = True
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Trainer resources released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
