"""
DPO (Direct Preference Optimization) trainer.

Implements the DPO algorithm for aligning LLMs with human preferences.
Supports sigmoid (standard DPO), IPO, and KTO pair loss variants.
"""

import os
from typing import List, Dict, Optional, Any

from loguru import logger

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = object

try:
    from peft import LoraConfig as PefLoraConfig, get_peft_model
except ImportError:
    PefLoraConfig = None
    get_peft_model = None

try:
    from trl import DPOTrainer as TRLDPOTrainer
except ImportError:
    TRLDPOTrainer = None

from config import DPOPipelineConfig, DPOConfig


class DPOTrainer:
    """DPO alignment trainer.

    The DPO loss:
        L_DPO = -E[log σ(β(log πθ(y_w|x) - log πref(y_w|x)
                           - (log πθ(y_l|x) - log πref(y_l|x))))]

    where y_w is the preferred response, y_l is the dispreferred response,
    πθ is the policy model, πref is the reference model, and β controls
    how much we focus on the preference margin.

    With label smoothing ε:
        L = (1-ε) * L_DPO(chosen) - ε * L_DPO(rejected)
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        config: Optional[DPOPipelineConfig] = None,
    ):
        if config is None:
            config = DPOPipelineConfig(base_model=base_model or "Qwen/Qwen3-0.6B")
        self.config = config
        self.base_model = config.base_model
        self._closed = False
        self.model = None
        self.reference_model = None
        self.tokenizer = None
        self.trainer = None

        self._load_models()

    def _load_models(self):
        """Load policy model, reference model, and tokenizer."""
        dpo_cfg = self.config.dpo

        logger.info(
            f"Loading DPO models: policy={self.base_model}, "
            f"reference={dpo_cfg.reference_model or 'policy (frozen copy)'}"
        )

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch is not None else "auto",
        }

        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required for DPOTrainer")

        # Load policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs,
        )

        # Load or create reference model
        ref_model_path = dpo_cfg.reference_model or self.base_model
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            **model_kwargs,
        )

        # Load tokenizer
        if AutoTokenizer is None:
            raise ImportError("transformers is required for DPOTrainer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("DPO models loaded successfully")

    def _create_trl_trainer(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ) -> Any:
        """Create the TRL DPOTrainer instance."""
        if TRLDPOTrainer is None:
            raise ImportError("trl is required for DPOTrainer")

        dpo_cfg = self.config.dpo

        training_args = TrainingArguments(
            output_dir="./dpo_output",
            num_train_epochs=dpo_cfg.num_epochs,
            per_device_train_batch_size=dpo_cfg.batch_size,
            learning_rate=dpo_cfg.learning_rate,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=False,
        )

        self.trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=self.reference_model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            beta=dpo_cfg.beta,
            max_length=dpo_cfg.max_length,
            max_prompt_length=dpo_cfg.max_prompt_length,
            loss_type=dpo_cfg.loss_type,
            label_smoothing=dpo_cfg.label_smoothing,
        )
        return self.trainer

    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Run DPO training.

        Args:
            train_data: List of preference pairs with "chosen" and "rejected".
            eval_data: Optional list of evaluation pairs.

        Returns:
            Training metrics dict.
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        logger.info(f"Starting DPO training with {len(train_data)} pairs")

        trainer = self._create_trl_trainer(train_data, eval_data)

        try:
            train_result = trainer.train()
        except KeyboardInterrupt:
            logger.warning("DPO training interrupted")
            return {"interrupted": True}

        metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
        logger.info(f"DPO training completed: {metrics}")
        return metrics

    def save(self, path: str):
        """Save the policy model.

        Args:
            path: Output directory path.
        """
        os.makedirs(path, exist_ok=True)
        if self.trainer is not None:
            self.trainer.save_model(path)
        elif self.model is not None:
            self.model.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
        logger.info(f"DPO policy saved to {path}")

    def close(self):
        """Release resources."""
        self._closed = True
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DPO trainer resources released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
