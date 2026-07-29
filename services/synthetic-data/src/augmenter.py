"""Data augmentation techniques."""

import random
import re
from typing import List, Dict, Optional

from loguru import logger


class DataAugmenter:
    """Data augmentation for synthetic datasets.

    Supports: synonym replacement, random insertion/swap/deletion,
    and back-translation augmentation.
    """

    def __init__(self, translator=None):
        self.translator = translator

    # ------------------------------------------------------------------
    # EDA: Easy Data Augmentation techniques
    # ------------------------------------------------------------------

    def synonym_replacement(self, text: str, p: float = 0.1) -> str:
        """Replace words with synonyms (simplified: uses word shuffling)."""
        words = text.split()
        if len(words) < 3:
            return text
        new_words = list(words)
        n_replace = max(1, int(len(words) * p))
        for _ in range(n_replace):
            idx = random.randint(0, len(new_words) - 1)
            # Simple synonym: replace with a random word from the same text
            candidates = [w for w in words if w != new_words[idx]]
            if candidates:
                new_words[idx] = random.choice(candidates)
        return " ".join(new_words)

    def random_insertion(self, text: str, p: float = 0.1) -> str:
        """Randomly insert synonyms of random words."""
        words = text.split()
        if len(words) < 3:
            return text
        n_insert = max(1, int(len(words) * p))
        for _ in range(n_insert):
            idx = random.randint(0, len(words) - 1)
            # Insert a random word from the text
            new_word = random.choice(words)
            words.insert(idx, new_word)
        return " ".join(words)

    def random_swap(self, text: str, n_swaps: int = 2) -> str:
        """Randomly swap two words in the text."""
        words = text.split()
        if len(words) < 2:
            return text
        n_swaps = min(n_swaps, len(words) // 2)
        for _ in range(n_swaps):
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) < 3:
            return text
        new_words = [w for w in words if random.random() > p]
        if not new_words:
            new_words = [random.choice(words)]
        return " ".join(new_words)

    # ------------------------------------------------------------------
    # Back-translation augmentation
    # ------------------------------------------------------------------

    def back_translation_augment(self, text: str, target_lang: str = "fr") -> str:
        """Augment text via back-translation if translator is available."""
        if self.translator is None:
            return self.synonym_replacement(text)
        prompt = f"Translate to {target_lang}:\n{text}"
        fwd = self.translator.generate([{"role": "user", "content": prompt}])
        prompt_back = f"Translate back to English:\n{fwd.chunk}"
        bwd = self.translator.generate([{"role": "user", "content": prompt_back}])
        return bwd.chunk

    # ------------------------------------------------------------------
    # Record-level augmentation
    # ------------------------------------------------------------------

    def augment_record(self, record: Dict) -> List[Dict]:
        """Augment a single record using multiple techniques.

        Args:
            record: Dict with instruction and output.

        Returns:
            List of augmented records (including original).
        """
        results = [record]  # Keep original
        instruction = record.get("instruction", record.get("input", ""))
        output = record.get("output", "")

        if not instruction:
            return results

        # Synonym replacement on instruction
        aug_inst = self.synonym_replacement(instruction)
        if aug_inst != instruction:
            results.append({"instruction": aug_inst, "output": output})

        # Random swap on output
        aug_out = self.random_swap(output)
        if aug_out != output:
            results.append({"instruction": instruction, "output": aug_out})

        # Both augmented
        if aug_inst != instruction and aug_out != output:
            results.append({"instruction": aug_inst, "output": aug_out})

        return results

    def augment_dataset(self, dataset: List[Dict], factor: int = 2) -> List[Dict]:
        """Augment an entire dataset.

        Args:
            dataset: List of records.
            factor: Target multiplication factor.

        Returns:
            Augmented dataset.
        """
        augmented = []
        for record in dataset:
            augmented.extend(self.augment_record(record))
        logger.info(f"Augmented dataset from {len(dataset)} to {len(augmented)} records")
        return augmented[:len(dataset) * factor]
