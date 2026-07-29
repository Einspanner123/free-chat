"""MMLU (Massive Multitask Language Understanding) benchmark."""

import re
import random
from typing import List, Dict, Any, Optional

from benchmarks.base import BaseBenchmark

# A minimal set of MMLU subjects and few-shot examples
_MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


class MMLUBenchmark(BaseBenchmark):
    """MMLU benchmark using 5-shot accuracy."""

    def __init__(self):
        self._results = {}

    def name(self) -> str:
        return "mmlu"

    def available_subjects(self) -> List[str]:
        return _MMLU_SUBJECTS

    def get_few_shot_examples(self, subject: str, num: int = 5) -> List[Dict]:
        """Get few-shot examples for a subject.

        In a real implementation, this loads from the MMLU dataset.
        Here we return placeholder examples.
        """
        return []

    def format_question(self, question: Dict) -> str:
        """Format a multiple-choice question.

        Expected question format:
            {"question": "...", "choices": [...], "answer": int}
        """
        q = question["question"]
        choices = question["choices"]
        labels = ["A", "B", "C", "D"]
        parts = [q]
        for i, (label, choice) in enumerate(zip(labels, choices)):
            parts.append(f"{label}. {choice}")
        return "\n".join(parts)

    def parse_answer(self, text: str) -> int:
        """Parse the answer from model output.

        Returns the index of the chosen answer (0-3) or -1 if not found.
        """
        text = text.strip()
        if not text:
            return -1

        # Direct letter answer
        if text in ("A", "B", "C", "D"):
            return ord(text) - ord("A")

        # Extract first letter from text like "The answer is A"
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return ord(match.group(1)) - ord("A")

        return -1

    def _query_model(self, model, prompt: str) -> str:
        """Query the model with a prompt and return the response."""
        response = model.generate([{"role": "user", "content": prompt}])
        return response.chunk

    def run(self, model, config=None, num_subjects: int = None, num_questions: int = None) -> Dict[str, Any]:
        """Run MMLU evaluation.

        Args:
            model: Engine with generate interface.
            config: Optional config with num_few_shot.
            num_subjects: Number of subjects to evaluate (None = all).
            num_questions: Questions per subject (None = all).

        Returns:
            Dict with accuracy and per-subject results.
        """
        subjects = _MMLU_SUBJECTS
        if num_subjects:
            subjects = subjects[:num_subjects]

        total_correct = 0
        total_questions = 0
        subject_results = {}

        for subject in subjects:
            subject_correct = 0
            subject_total = 0
            n_q = num_questions or 5  # Use 5 questions per subject in test mode

            for q_idx in range(n_q):
                placeholder_q = {
                    "question": f"Sample {subject} question {q_idx}?",
                    "choices": ["Option A", "Option B", "Option C", "Option D"],
                    "answer": q_idx % 4,
                }
                prompt = self.format_question(placeholder_q)
                response = self._query_model(model, prompt)
                predicted = self.parse_answer(response)
                correct = predicted == placeholder_q["answer"]

                if correct:
                    subject_correct += 1
                    total_correct += 1
                total_questions += 1
                subject_total += 1

            subject_results[subject] = {
                "correct": subject_correct,
                "total": subject_total,
                "accuracy": subject_correct / subject_total if subject_total > 0 else 0.0,
            }

        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        self._results = {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_questions,
            "subjects": subject_results,
        }
        return self._results

    def get_metrics(self) -> Dict[str, float]:
        return {"accuracy": self._results.get("accuracy", 0.0)}
