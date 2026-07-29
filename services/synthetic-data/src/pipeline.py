"""Full synthetic data pipeline: generate → filter → augment → export."""

import json
import os
import random
from typing import List, Dict, Any, Optional

from loguru import logger

from config import SynthConfig, FilterConfig
from generator import GeneratorFactory, SelfInstructGenerator
from filter import QualityFilter
from augmenter import DataAugmenter


_SEED_EXAMPLES = {
    "general": [
        {"instruction": "What is the capital of France?", "output": "Paris."},
        {"instruction": "Explain photosynthesis.", "output": "Photosynthesis is..."},
        {"instruction": "Write a haiku about nature.", "output": "Green leaves sway..."},
    ],
    "tech": [
        {"instruction": "What is a linked list?", "output": "A linked list is..."},
        {"instruction": "Explain TCP/IP.", "output": "TCP/IP is..."},
        {"instruction": "Write a function to reverse a string.", "output": "def reverse(s): return s[::-1]"},
    ],
    "science": [
        {"instruction": "What is the water cycle?", "output": "The water cycle..."},
        {"instruction": "Explain Newton's laws.", "output": "Newton's laws..."},
    ],
}


class SynthPipeline:
    """End-to-end synthetic data pipeline."""

    def __init__(self, config: Optional[SynthConfig] = None):
        self.config = config or SynthConfig()
        self._llm = None
        self._generator = None
        self._filter = None
        self._augmenter = None

    def set_llm(self, llm):
        self._llm = llm

    def get_seed_examples(self, domain: str = "general") -> List[Dict]:
        """Get seed examples for a domain."""
        return _SEED_EXAMPLES.get(domain, _SEED_EXAMPLES["general"])

    def generate(self, num: int = 10, seed_topic: str = "general") -> List[Dict]:
        """Generate synthetic data.

        Args:
            num: Number of examples to generate.
            seed_topic: Topic/domain for generation.

        Returns:
            Generated dataset.
        """
        if self._llm is None:
            logger.warning("No LLM set, using seed examples only")
            return self.get_seed_examples(domain=seed_topic)

        dataset = []
        for strategy in self.config.strategies:
            generator = GeneratorFactory.create(strategy, llm=self._llm)
            gen_per_strategy = num // len(self.config.strategies)
            try:
                result = generator.generate_dataset(num=gen_per_strategy, seed_topic=seed_topic)
                dataset.extend(result)
            except Exception as e:
                logger.error(f"Strategy '{strategy}' failed: {e}")

        return dataset[:num]

    def generate_with_filter(
        self,
        num: int = 10,
        seed_topic: str = "general",
        filter_config: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate and filter data."""
        dataset = self.generate(num, seed_topic)
        fc = FilterConfig(**(filter_config or {}))
        qf = QualityFilter(min_length=fc.min_length, max_length=fc.max_length)
        filtered = qf.filter(dataset, deduplicate=fc.deduplicate, remove_html=fc.remove_html)
        logger.info(f"Filtered from {len(dataset)} to {len(filtered)}")
        return filtered

    def generate_with_augmentation(
        self,
        num: int = 10,
        seed_topic: str = "general",
        augment_factor: int = 2,
    ) -> List[Dict]:
        """Generate, filter, and augment data."""
        dataset = self.generate_with_filter(num, seed_topic)
        aug = DataAugmenter(translator=self._llm)
        augmented = aug.augment_dataset(dataset, factor=augment_factor)
        return augmented

    def run(
        self,
        num_generate: int = 100,
        seed_topic: str = "general",
        filter_config: Optional[Dict] = None,
        augment_factor: int = 1,
    ) -> Dict[str, Any]:
        """Run the full pipeline: generate → filter → augment.

        Args:
            num_generate: Number of examples to generate.
            seed_topic: Topic/domain.
            filter_config: Filter configuration overrides.
            augment_factor: Data augmentation factor.

        Returns:
            Dict with dataset and stats.
        """
        generated = self.generate(num_generate, seed_topic)
        stats = {"generated": len(generated)}

        fc = FilterConfig(**(filter_config or {}))
        qf = QualityFilter(min_length=fc.min_length, max_length=fc.max_length)
        filtered = qf.filter(generated, deduplicate=fc.deduplicate, remove_html=fc.remove_html)
        stats["after_filter"] = len(filtered)
        stats["filter_dropped"] = len(generated) - len(filtered)

        if augment_factor > 1:
            aug = DataAugmenter(translator=self._llm)
            filtered = aug.augment_dataset(filtered, factor=augment_factor)
        stats["after_augment"] = len(filtered)

        return {"dataset": filtered, "stats": stats}

    def compute_stats(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not dataset:
            return {"num_examples": 0}

        total_inst_len = 0
        total_out_len = 0
        for item in dataset:
            inst = item.get("instruction", item.get("input", ""))
            out = item.get("output", "")
            total_inst_len += len(inst)
            total_out_len += len(out)

        n = len(dataset)
        return {
            "num_examples": n,
            "avg_instruction_len": total_inst_len // n,
            "avg_output_len": total_out_len // n,
        }

    def save_dataset(self, dataset: List[Dict], path: str):
        """Save dataset to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    def export_dataset(self, dataset: List[Dict], path: str, format: str = "jsonl"):
        """Export dataset in specified format.

        Args:
            dataset: Dataset to export.
            path: Output path.
            format: "json" or "jsonl".
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if format == "jsonl":
            with open(path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
