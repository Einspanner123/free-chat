"""
Command-line entry point for the fine-tuning pipeline.

Usage:
    python finetune_cli.py --config config.yaml --data train.jsonl
    python finetune_cli.py --merge --base_model Qwen/Qwen3-0.6B --lora_path ./output/lora-final
"""

import argparse
import json
import os
import sys

from loguru import logger

from config import TrainingPipelineConfig, FineTuneConfig, LoraConfig, QLoraConfig
from data_processor import DataProcessor
from lora_trainer import LoraTrainer
from merge_lora import LoraMerger


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA Fine-tuning Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # train
    train_parser = subparsers.add_parser("train", help="Run fine-tuning")
    train_parser.add_argument("--config", type=str, help="Path to config YAML/JSON")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training data")
    train_parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    train_parser.add_argument("--format", type=str, default="sharegpt",
                              choices=["sharegpt", "alpaca", "chatml"])
    train_parser.add_argument("--base-model", type=str, help="Base model name or path")
    train_parser.add_argument("--output-dir", type=str, default="./output")
    train_parser.add_argument("--num-epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--lora-r", type=int, default=8)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--qlora", action="store_true", help="Use QLoRA")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA weights")
    merge_parser.add_argument("--base-model", type=str, required=True)
    merge_parser.add_argument("--lora-path", type=str, required=True)
    merge_parser.add_argument("--output-path", type=str, required=True)
    merge_parser.add_argument("--push-to-hub", action="store_true")
    merge_parser.add_argument("--repo-id", type=str, help="HF Hub repo ID")

    return parser.parse_args()


def cmd_train(args):
    # Load or build config
    if args.config:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            cfg = TrainingPipelineConfig(
                finetune=FineTuneConfig.load_yaml(args.config)
            )
        else:
            with open(args.config) as f:
                cfg = TrainingPipelineConfig.from_dict(json.load(f))
    else:
        finetune_cfg = FineTuneConfig(
            base_model=args.base_model or "Qwen/Qwen3-0.6B",
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha)
        qlora_cfg = QLoraConfig(load_in_4bit=args.qlora) if args.qlora else QLoraConfig.none()
        cfg = TrainingPipelineConfig(finetune=finetune_cfg, lora=lora_cfg, qlora=qlora_cfg)

    logger.info(f"Training config: {cfg.to_dict()}")

    # Load data
    dp = DataProcessor()
    train_data = dp.load_file(args.data, format=args.format)
    logger.info(f"Loaded {len(train_data)} training examples")

    eval_data = None
    if args.eval_data:
        eval_data = dp.load_file(args.eval_data, format=args.format)
        logger.info(f"Loaded {len(eval_data)} evaluation examples")

    # Stats
    stats = dp.compute_statistics(train_data)
    logger.info(f"Training data stats: {stats}")

    # Train
    with LoraTrainer(
        base_model=cfg.finetune.base_model,
        config=cfg,
    ) as trainer:
        metrics = trainer.train(
            train_data,
            eval_data=eval_data,
            resume_from_checkpoint=args.resume,
        )
        logger.info(f"Training metrics: {metrics}")
        trainer.save(os.path.join(cfg.finetune.output_dir, "lora-final"))


def cmd_merge(args):
    merger = LoraMerger()
    cfg = merger.create_config(
        base_model=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hf_repo_id=args.repo_id,
    )
    output = merger.merge_and_save(cfg)
    logger.info(f"Merged model saved to {output}")


def main():
    args = parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "merge":
        cmd_merge(args)
    else:
        print("Usage: python finetune_cli.py [train|merge] --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
