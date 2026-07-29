"""Data generators: Self-Instruct, Evol-Question, Back-Translation."""

import re
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from loguru import logger


class BaseGenerator(ABC):
    def __init__(self, llm):
        self._llm = llm

    @abstractmethod
    def generate_dataset(self, **kwargs) -> List[Dict]:
        ...


class SelfInstructGenerator(BaseGenerator):
    """Self-Instruct: generate {instruction, output} pairs from seed topics."""

    def __init__(self, llm, temperature: float = 0.8):
        super().__init__(llm)
        self.temperature = temperature

    def generate_tasks(self, num: int = 5, seed_topic: str = "general") -> List[str]:
        prompt = f"""Generate {num} diverse {seed_topic} questions or tasks. 
Each on a new line starting with a number and a period.
Examples:
1. What is machine learning?
2. Explain the concept of recursion.
3. Write a Python function to sort a list."""
        response = self._llm.generate([{"role": "user", "content": prompt}])
        tasks = self._parse_tasks(response.chunk)
        return tasks[:num]

    def generate_response(self, instruction: str) -> Dict[str, str]:
        prompt = f"Instruction: {instruction}\n\nProvide a detailed, accurate response:"
        response = self._llm.generate([{"role": "user", "content": prompt}])
        return {"instruction": instruction, "output": response.chunk}

    def generate_dataset(self, num: int = 10, seed_topic: str = "general") -> List[Dict]:
        tasks = self.generate_tasks(num, seed_topic)
        dataset = []
        for task in tasks:
            try:
                item = self.generate_response(task)
                dataset.append(item)
            except Exception as e:
                logger.warning(f"Failed to generate for '{task}': {e}")
        logger.info(f"SelfInstruct: generated {len(dataset)} examples")
        return dataset

    @staticmethod
    def _parse_tasks(text: str) -> List[str]:
        lines = text.strip().split("\n")
        tasks = []
        for line in lines:
            line = line.strip()
            # Remove leading numbers like "1. " or "1) "
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if line and len(line) > 10:
                tasks.append(line)
        return tasks


class EvolQuestionGenerator(BaseGenerator):
    """Evol-Question: evolve simple questions into more complex ones."""

    def evolve_deepen(self, question: str) -> str:
        prompt = f"""Rewrite the following question to make it more specific and require deeper understanding.
Add constraints or ask for step-by-step reasoning.
Original: {question}
Evolved:"""
        response = self._llm.generate([{"role": "user", "content": prompt}])
        return self._clean_evolved(response.chunk, question)

    def evolve_breaden(self, question: str) -> str:
        prompt = f"""Rewrite the following question to cover a broader scope or compare multiple concepts.
Original: {question}
Evolved:"""
        response = self._llm.generate([{"role": "user", "content": prompt}])
        return self._clean_evolved(response.chunk, question)

    def generate_dataset(self, seed_questions: List[str], **kwargs) -> List[Dict]:
        dataset = []
        for q in seed_questions:
            for evolve_fn in [self.evolve_deepen, self.evolve_breaden]:
                try:
                    evolved = evolve_fn(q)
                    if evolved and evolved != q:
                        dataset.append({"instruction": evolved, "output": ""})
                except Exception as e:
                    logger.warning(f"Evolve failed for '{q}': {e}")
        logger.info(f"EvolQuestion: generated {len(dataset)} examples")
        return dataset

    @staticmethod
    def _clean_evolved(text: str, original: str) -> str:
        text = text.strip().strip('"\'')
        return text if len(text) > len(original) * 0.5 else original


class BackTranslationGenerator(BaseGenerator):
    """Back-Translation: paraphrase via round-trip translation."""

    def paraphrase(self, text: str) -> str:
        prompt = f"Paraphrase the following text:\n{text}"
        response = self._llm.generate([{"role": "user", "content": prompt}])
        return response.chunk

    def back_translate(self, text: str, source_lang: str = "English", bridge_lang: str = "French") -> str:
        prompt_forward = f"Translate from {source_lang} to {bridge_lang}:\n{text}"
        forward = self._llm.generate([{"role": "user", "content": prompt_forward}])
        prompt_back = f"Translate from {bridge_lang} to {source_lang}:\n{forward.chunk}"
        backward = self._llm.generate([{"role": "user", "content": prompt_back}])
        return backward.chunk

    def generate_dataset(self, seed_texts: List[str], **kwargs) -> List[Dict]:
        dataset = []
        for text in seed_texts:
            try:
                paraphrased = self.paraphrase(text)
                dataset.append({"instruction": text, "output": paraphrased})
            except Exception as e:
                logger.warning(f"Back-translation failed: {e}")
        return dataset


class GeneratorFactory:
    @staticmethod
    def create(strategy: str, llm, **kwargs) -> BaseGenerator:
        if strategy == "self_instruct":
            return SelfInstructGenerator(llm=llm, **kwargs)
        elif strategy == "evol_question":
            return EvolQuestionGenerator(llm=llm)
        elif strategy == "back_translation":
            return BackTranslationGenerator(llm=llm)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
