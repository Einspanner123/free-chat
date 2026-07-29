"""GSM8K (Grade School Math) benchmark."""

import re
from typing import List, Dict, Any, Optional

from benchmarks.base import BaseBenchmark


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K math reasoning benchmark."""

    def __init__(self):
        self._results = {}

    def name(self) -> str:
        return "gsm8k"

    def load_problems(self, num: Optional[int] = None) -> List[Dict]:
        """Load GSM8K problems.

        In production, loads from the dataset. Here returns placeholders.
        """
        problems = [
            {"question": f"Sample math problem {i}. What is the answer?", "answer": str(i * 2 + 1)}
            for i in range(20)
        ]
        if num:
            problems = problems[:num]
        return problems

    def format_problem(self, problem: Dict) -> str:
        """Format a math problem into a prompt."""
        return f"Problem: {problem['question']}\nLet's think step by step.\n"

    def extract_answer(self, text: str) -> str:
        """Extract the numeric answer from model output."""
        # Look for "answer is X" pattern
        match = re.search(r'(?:answer|answer is|the answer is)\s*:?\s*([\d]+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Look for last number in text
        numbers = re.findall(r'[\d]+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return ""

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Check if the predicted answer matches expected."""
        # Clean both
        pred = re.sub(r'[^0-9.]', '', predicted)
        exp = re.sub(r'[^0-9.]', '', expected)
        if not pred or not exp:
            return pred == exp
        return abs(float(pred) - float(exp)) < 1e-6

    def _query_model(self, model, prompt: str) -> str:
        response = model.generate([{"role": "user", "content": prompt}])
        return response.chunk

    def run(self, model, config=None, num_problems: int = None) -> Dict[str, Any]:
        problems = self.load_problems(num_problems)

        correct = 0
        total = 0
        problem_results = []

        for problem in problems:
            prompt = self.format_problem(problem)
            response = self._query_model(model, prompt)
            predicted = self.extract_answer(response)
            expected = problem["answer"]
            is_correct = self.check_answer(predicted, expected)

            if is_correct:
                correct += 1
            total += 1

            problem_results.append({
                "correct": is_correct,
                "predicted": predicted,
                "expected": expected,
            })

        accuracy = correct / total if total > 0 else 0.0
        self._results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "problems": problem_results,
        }
        return self._results

    def get_metrics(self) -> Dict[str, float]:
        return {"accuracy": self._results.get("accuracy", 0.0)}
