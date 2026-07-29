"""C-Eval (Chinese Evaluation) benchmark."""

import re
from typing import List, Dict, Any, Optional

from benchmarks.base import BaseBenchmark

_CEVAL_SUBJECTS = [
    "high_school_mathematics", "high_school_physics", "high_school_chemistry",
    "high_school_biology", "high_school_chinese", "high_school_history",
    "high_school_geography", "high_school_politics", "college_mathematics",
    "college_physics", "college_chemistry", "college_biology",
    "college_computer_science", "college_medicine", "college_law",
    "college_economics", "college_psychology", "professional_medicine",
    "professional_accounting", "professional_law",
]


class CEvalBenchmark(BaseBenchmark):
    """C-Eval (Chinese Evaluation) benchmark."""

    def __init__(self):
        self._results = {}

    def name(self) -> str:
        return "ceval"

    def available_subjects(self) -> List[str]:
        return _CEVAL_SUBJECTS

    def format_question(self, question: Dict) -> str:
        """Format a Chinese multiple-choice question."""
        q = question["question"]
        choices = question["choices"]
        labels = ["A", "B", "C", "D"]
        parts = [q]
        for label, choice in zip(labels, choices):
            parts.append(f"{label}. {choice}")
        return "\n".join(parts)

    def parse_answer(self, text: str) -> int:
        """Parse the answer from model output. Supports Chinese text."""
        text = text.strip()
        if not text:
            return -1
        if text in ("A", "B", "C", "D"):
            return ord(text) - ord("A")

        # Chinese: "答案是C" or "C选项正确"
        match = re.search(r'[ABCD]', text)
        if match:
            return ord(match.group(0)) - ord("A")

        return -1

    def _query_model(self, model, prompt: str) -> str:
        response = model.generate([{"role": "user", "content": prompt}])
        return response.chunk

    def run(self, model, config=None, num_subjects: int = None, num_questions: int = None) -> Dict[str, Any]:
        subjects = _CEVAL_SUBJECTS
        if num_subjects:
            subjects = subjects[:num_subjects]

        total_correct = 0
        total_questions = 0
        subject_results = {}

        for subject in subjects:
            subject_correct = 0
            subject_total = 0
            n_q = num_questions or 5

            for q_idx in range(n_q):
                placeholder_q = {
                    "question": f"以下哪个关于{subject}的说法是正确的？",
                    "choices": ["选项A", "选项B", "选项C", "选项D"],
                    "answer": q_idx % 4,
                }
                prompt = self.format_question(placeholder_q)
                response = self._query_model(model, prompt)
                predicted = self.parse_answer(response)
                if predicted == placeholder_q["answer"]:
                    subject_correct += 1
                    total_correct += 1
                total_questions += 1
                subject_total += 1

            subject_results[subject] = {
                "correct": subject_correct,
                "total": subject_total,
                "accuracy": subject_correct / subject_total if subject_total > 0 else 0.0,
            }

        overall = total_correct / total_questions if total_questions > 0 else 0.0
        self._results = {"accuracy": overall, "correct": total_correct, "total": total_questions, "subjects": subject_results}
        return self._results

    def get_metrics(self) -> Dict[str, float]:
        return {"accuracy": self._results.get("accuracy", 0.0)}
