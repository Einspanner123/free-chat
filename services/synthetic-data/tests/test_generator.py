import os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestSelfInstruct:
    def test_generate_tasks(self):
        from generator import SelfInstructGenerator
        mock_llm = MagicMock()
        mock_llm.generate.return_value.chunk = "1. What is Python?\n2. Explain AI."
        gen = SelfInstructGenerator(llm=mock_llm)
        tasks = gen.generate_tasks(num=5, seed_topic="programming")
        assert len(tasks) >= 1
        assert "Python" in str(tasks) or "AI" in str(tasks)

    def test_generate_response(self):
        from generator import SelfInstructGenerator
        mock_llm = MagicMock()
        mock_llm.generate.return_value.chunk = "A programming language."
        gen = SelfInstructGenerator(llm=mock_llm)
        data = gen.generate_response("What is Python?")
        assert data["instruction"] == "What is Python?"
        assert data["output"] == "A programming language."

    def test_generate_dataset(self):
        from generator import SelfInstructGenerator
        mock_llm = MagicMock()
        # Return valid task list for generate_tasks call, then responses for each
        mock_llm.generate.side_effect = [
            type('R', (), {'chunk': '1. What is Python?\n2. Explain AI.\n3. What is ML?'})(),
            type('R', (), {'chunk': 'Python is a language.'})(),
            type('R', (), {'chunk': 'AI is a field.'})(),
            type('R', (), {'chunk': 'ML is a subset.'})(),
        ]
        gen = SelfInstructGenerator(llm=mock_llm)
        dataset = gen.generate_dataset(num=3, seed_topic="general")
        assert len(dataset) == 3
        for item in dataset:
            assert "instruction" in item
            assert "output" in item

class TestEvolQuestion:
    def test_evolve_deepen(self):
        from generator import EvolQuestionGenerator
        mock_llm = MagicMock()
        mock_llm.generate.return_value.chunk = "Explain the mathematical foundations of neural networks including backpropagation."
        gen = EvolQuestionGenerator(llm=mock_llm)
        evolved = gen.evolve_deepen("What is a neural network?")
        assert len(evolved) > len("What is a neural network?")

    def test_evolve_breaden(self):
        from generator import EvolQuestionGenerator
        mock_llm = MagicMock()
        mock_llm.generate.return_value.chunk = "Compare and contrast neural networks with decision trees and SVMs."
        gen = EvolQuestionGenerator(llm=mock_llm)
        evolved = gen.evolve_breaden("What is a neural network?")
        assert evolved != "What is a neural network?"

    def test_generate_dataset(self):
        from generator import EvolQuestionGenerator
        mock_llm = MagicMock()
        mock_llm.generate.return_value.chunk = "Evolved much longer question that passes the length check."
        gen = EvolQuestionGenerator(llm=mock_llm)
        seeds = ["Q1", "Q2", "Q3"]
        dataset = gen.generate_dataset(seed_questions=seeds)
        # 2 evolve functions per seed * 3 seeds = 6
        assert len(dataset) == 6

class TestBackTranslation:
    def test_paraphrase(self):
        from generator import BackTranslationGenerator
        mock_llm = MagicMock()
        mock_llm.generate.return_value.chunk = "Paraphrased: What is the definition of Python?"
        gen = BackTranslationGenerator(llm=mock_llm)
        result = gen.paraphrase("What is Python?")
        assert result is not None

    def test_back_translate(self):
        from generator import BackTranslationGenerator
        mock_llm = MagicMock()
        # First: translate to French
        # Second: translate back to English
        mock_llm.generate.side_effect = [
            type('R', (), {'chunk': 'Qu\'est-ce que Python?'})(),
            type('R', (), {'chunk': 'What is Python?'})(),
        ]
        gen = BackTranslationGenerator(llm=mock_llm)
        result = gen.back_translate("What is Python?", source_lang="English", bridge_lang="French")
        assert result is not None

class TestGeneratorFactory:
    def test_factory_create(self):
        from generator import GeneratorFactory
        mock_llm = MagicMock()
        gen = GeneratorFactory.create("self_instruct", llm=mock_llm)
        assert gen is not None
        gen2 = GeneratorFactory.create("evol_question", llm=mock_llm)
        assert gen2 is not None
        gen3 = GeneratorFactory.create("back_translation", llm=mock_llm)
        assert gen3 is not None

    def test_factory_invalid(self):
        from generator import GeneratorFactory
        from unittest.mock import MagicMock
        with pytest.raises(ValueError):
            GeneratorFactory.create("invalid", llm=MagicMock())
