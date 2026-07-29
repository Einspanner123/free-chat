"""HumanEval code generation benchmark."""

import ast
import sys
from typing import List, Dict, Any, Optional
from io import StringIO
from contextlib import redirect_stdout

from benchmarks.base import BaseBenchmark


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval benchmark for code generation."""

    def __init__(self):
        self._results = {}

    def name(self) -> str:
        return "humaneval"

    def get_problems(self) -> List[Dict]:
        """Get HumanEval problems.

        In production, loads from the dataset. Here returns placeholders.
        """
        return [
            {
                "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
                "entry_point": "add",
                "test": "assert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(-1, 1) == 0",
            },
            {
                "prompt": "def factorial(n):\n    \"\"\"Return n!\"\"\"\n",
                "entry_point": "factorial",
                "test": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6",
            },
        ]

    def format_prompt(self, problem: Dict) -> str:
        """Format a coding problem into a prompt."""
        prompt = problem["prompt"]
        return (
            f"Complete the following Python function:\n\n"
            f"{prompt}\n\n"
            f"Return only the function definition, no explanations."
        )

    def check_solution(self, problem: Dict, solution: str) -> bool:
        """Check if the solution passes the test cases.

        Args:
            problem: Dict with entry_point and test fields.
            solution: The generated code.

        Returns:
            True if all tests pass, False otherwise.
        """
        if not solution or not solution.strip():
            return False

        # Try to parse the solution first
        try:
            ast.parse(solution)
        except SyntaxError:
            return False

        # Create a namespace and execute
        namespace = {}
        try:
            exec(solution, namespace)
            if problem["entry_point"] not in namespace:
                return False

            # Run tests
            test_code = problem.get("test", "")
            if not test_code:
                return False

            # Capture stdout to avoid noise
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                exec(test_code, namespace)
                return True
            finally:
                sys.stdout = old_stdout

        except Exception:
            return False

    def _query_model(self, model, prompt: str) -> str:
        response = model.generate([{"role": "user", "content": prompt}])
        return response.chunk

    def run(self, model, config=None, num_problems: int = None) -> Dict[str, Any]:
        problems = self.get_problems()
        if num_problems:
            problems = problems[:num_problems]

        passed = 0
        total = 0
        problem_results = []

        for problem in problems:
            prompt = self.format_prompt(problem)
            solution = self._query_model(model, prompt)
            is_correct = self.check_solution(problem, solution)

            if is_correct:
                passed += 1
            total += 1

            problem_results.append({
                "entry_point": problem["entry_point"],
                "passed": is_correct,
            })

        pass_rate = passed / total if total > 0 else 0.0
        self._results = {
            "pass@1": pass_rate,
            "passed": passed,
            "total": total,
            "problems": problem_results,
        }
        return self._results

    def get_metrics(self) -> Dict[str, float]:
        return {"pass@1": self._results.get("pass@1", 0.0)}
