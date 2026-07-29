import os, sys, pytest
from unittest.mock import MagicMock
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestDataAugmenter:
    def test_synonym_replacement(self):
        from augmenter import DataAugmenter
        aug = DataAugmenter()
        text = "The quick brown fox"
        augmented = aug.synonym_replacement(text)
        assert augmented is not None
        assert len(augmented) > 0

    def test_random_insertion(self):
        from augmenter import DataAugmenter
        aug = DataAugmenter()
        text = "Hello world"
        augmented = aug.random_insertion(text)
        assert len(augmented) >= len(text)

    def test_random_swap(self):
        from augmenter import DataAugmenter
        aug = DataAugmenter()
        text = "The quick brown fox jumps"
        augmented = aug.random_swap(text)
        assert augmented is not None

    def test_random_deletion(self):
        from augmenter import DataAugmenter
        aug = DataAugmenter()
        text = "The quick brown fox jumps over the lazy dog"
        augmented = aug.random_deletion(text, p=0.3)
        assert len(augmented) > 0
        assert len(augmented.split()) <= len(text.split())

    def test_back_translation_augment(self):
        from augmenter import DataAugmenter
        mock_translator = MagicMock()
        mock_translator.generate.return_value.chunk = "Translated text."
        aug = DataAugmenter(translator=mock_translator)
        result = aug.back_translation_augment("Hello world", target_lang="fr")
        assert result is not None

    def test_augment_record(self):
        from augmenter import DataAugmenter
        aug = DataAugmenter()
        record = {"instruction": "What is Python?", "output": "A language."}
        augmented = aug.augment_record(record)
        assert len(augmented) >= 1
        for r in augmented:
            assert "instruction" in r
            assert "output" in r

    def test_augment_dataset(self):
        from augmenter import DataAugmenter
        aug = DataAugmenter()
        dataset = [
            {"instruction": f"Q{i}", "output": f"A{i}"}
            for i in range(5)
        ]
        augmented = aug.augment_dataset(dataset, factor=2)
        assert len(augmented) >= len(dataset)
