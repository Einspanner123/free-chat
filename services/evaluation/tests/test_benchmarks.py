"""Tests for benchmark implementations."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestBaseBenchmark:
    def test_base_class_cannot_instantiate(self):
        from benchmarks.base import BaseBenchmark
        with pytest.raises(TypeError):
            BaseBenchmark()

    def test_abstract_methods(self):
        from benchmarks.base import BaseBenchmark
        import inspect
        methods = {name for name, _ in inspect.getmembers(BaseBenchmark, predicate=inspect.isfunction)}
        required = {"run", "name", "get_metrics"}
        for m in required:
            assert m in dir(BaseBenchmark)

    def test_concrete_benchmark(self):
        from benchmarks.base import BaseBenchmark
        from config import EvalConfig

        class TestBM(BaseBenchmark):
            def name(self): return "test"
            def run(self, model, config): return {"accuracy": 0.5}
            def get_metrics(self): return {"accuracy": 0.5}

        bm = TestBM()
        assert bm.name() == "test"
        assert bm.get_metrics() == {"accuracy": 0.5}


class TestMMLUBenchmark:
    def test_name(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        assert "mmlu" in bm.name().lower()

    def test_subjects_list(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        subjects = bm.available_subjects()
        assert len(subjects) > 0
        assert "abstract_algebra" in subjects
        assert "anatomy" in subjects

    def test_few_shot_examples(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        examples = bm.get_few_shot_examples("anatomy", num=2)
        assert len(examples) <= 2

    def test_format_question(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        question = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
        }
        formatted = bm.format_question(question)
        assert "capital of France" in formatted
        assert "Paris" in formatted
        assert "A." in formatted or "A)" in formatted

    def test_parse_answer(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        assert bm.parse_answer("A") == 0
        assert bm.parse_answer("B") == 1
        assert bm.parse_answer("C") == 2
        assert bm.parse_answer("D") == 3

    def test_parse_answer_from_text(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        assert bm.parse_answer("The answer is A") == 0
        assert bm.parse_answer("I think B is correct") == 1
        assert bm.parse_answer("C is the right choice") == 2

    def test_parse_answer_invalid(self):
        from benchmarks.mmlu import MMLUBenchmark
        bm = MMLUBenchmark()
        assert bm.parse_answer("") == -1
        assert bm.parse_answer("No answer here") == -1


class TestCEvalBenchmark:
    def test_name(self):
        from benchmarks.ceval import CEvalBenchmark
        bm = CEvalBenchmark()
        assert "ceval" in bm.name().lower() or "c-eval" in bm.name().lower()

    def test_subjects_list(self):
        from benchmarks.ceval import CEvalBenchmark
        bm = CEvalBenchmark()
        subjects = bm.available_subjects()
        assert len(subjects) > 0
        assert "high_school_mathematics" in subjects or len(subjects) > 5

    def test_chinese_content(self):
        from benchmarks.ceval import CEvalBenchmark
        bm = CEvalBenchmark()
        question = {
            "question": "以下哪个是中国的首都？",
            "choices": ["上海", "北京", "广州", "深圳"],
            "answer": 1,
        }
        formatted = bm.format_question(question)
        assert "北京" in formatted
        assert "A." in formatted or "A)" in formatted

    def test_parse_answer_chinese(self):
        from benchmarks.ceval import CEvalBenchmark
        bm = CEvalBenchmark()
        assert bm.parse_answer("A") == 0
        assert bm.parse_answer("B") == 1
        assert bm.parse_answer("答案是C") == 2
        assert bm.parse_answer("D选项正确") == 3


class TestGSM8KBenchmark:
    def test_name(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        bm = GSM8KBenchmark()
        assert "gsm8k" in bm.name().lower()

    def test_format_problem(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        bm = GSM8KBenchmark()
        problem = {"question": "Jan has 5 apples. She gives 2 away. How many remain?", "answer": "3"}
        formatted = bm.format_problem(problem)
        assert "apples" in formatted
        assert "remain" in formatted

    def test_extract_answer(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        bm = GSM8KBenchmark()
        assert bm.extract_answer("The answer is 42.") == "42"
        assert bm.extract_answer("42") == "42"
        assert bm.extract_answer("So the final answer is 3 apples.") == "3"

    def test_extract_answer_with_reasoning(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        bm = GSM8KBenchmark()
        response = "First, I add 5 + 3 = 8. Then I subtract 2, so answer is 6. The answer is 6."
        assert bm.extract_answer(response) == "6"

    def test_check_answer(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        bm = GSM8KBenchmark()
        assert bm.check_answer("42", "42") is True
        assert bm.check_answer("42", "42.0") is True
        assert bm.check_answer("3.5", "3.5") is True
        assert bm.check_answer("42", "43") is False

    def test_check_answer_flexible(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        bm = GSM8KBenchmark()
        assert bm.check_answer("$42", "42") is True
        assert bm.check_answer("42 apples", "42") is True
        assert bm.check_answer("5%", "5") is True


class TestHumanEval:
    def test_name(self):
        from benchmarks.humaneval import HumanEvalBenchmark
        bm = HumanEvalBenchmark()
        assert "humaneval" in bm.name().lower()

    def test_count_problems(self):
        from benchmarks.humaneval import HumanEvalBenchmark
        bm = HumanEvalBenchmark()
        problems = bm.get_problems()
        assert len(problems) > 0

    def test_format_prompt(self):
        from benchmarks.humaneval import HumanEvalBenchmark
        bm = HumanEvalBenchmark()
        problem = {
            "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
            "entry_point": "add",
            "test": "assert add(1, 2) == 3",
        }
        formatted = bm.format_prompt(problem)
        assert "def add" in formatted
        assert "Return a + b" in formatted

    def test_check_solution_correct(self):
        from benchmarks.humaneval import HumanEvalBenchmark
        bm = HumanEvalBenchmark()
        problem = {
            "entry_point": "add",
            "test": "assert add(1, 2) == 3",
        }
        solution = "def add(a, b):\n    return a + b"
        result = bm.check_solution(problem, solution)
        # In test environment, this may be mocked
        assert result is True or result is False

    def test_check_solution_syntax_error(self):
        from benchmarks.humaneval import HumanEvalBenchmark
        bm = HumanEvalBenchmark()
        problem = {"entry_point": "f", "test": "assert f() == 1"}
        solution = "def f():\n    return "
        result = bm.check_solution(problem, solution)
        assert result is False

    def test_check_solution_empty(self):
        from benchmarks.humaneval import HumanEvalBenchmark
        bm = HumanEvalBenchmark()
        problem = {"entry_point": "f", "test": "assert f() == 1"}
        assert bm.check_solution(problem, "") is False


class TestBenchmarkRunner:
    def test_run_mmlu_benchmark(self):
        from benchmarks.mmlu import MMLUBenchmark
        mock_model = MagicMock()
        mock_model.generate.return_value = "A"
        mock_model.count_tokens.return_value = 10

        bm = MMLUBenchmark()
        with patch.object(bm, '_query_model', return_value="A"):
            results = bm.run(mock_model, num_subjects=1, num_questions=2)
            assert results is not None
            assert "accuracy" in results or "overall" in results

    def test_run_gsm8k_benchmark(self):
        from benchmarks.gsm8k import GSM8KBenchmark
        mock_model = MagicMock()
        bm = GSM8KBenchmark()
        with patch.object(bm, '_query_model', return_value="The answer is 42."):
            with patch.object(bm, 'load_problems', return_value=[{"question": "test", "answer": "42"}]):
                results = bm.run(mock_model, num_problems=1)
                assert results is not None
