import os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestSynthPipeline:
    @pytest.fixture
    def pipeline(self):
        from pipeline import SynthPipeline
        from config import SynthConfig
        p = SynthPipeline(config=SynthConfig(max_generated=10))
        # Mock LLM
        p._llm = MagicMock()
        p._llm.generate.return_value.chunk = "Mock instruction.\nMock response."
        return p

    def test_generate(self, pipeline):
        dataset = pipeline.generate(num=5, seed_topic="general")
        assert len(dataset) >= 1
        for item in dataset:
            assert "instruction" in item or "messages" in item

    def test_generate_with_filter(self, pipeline):
        dataset = pipeline.generate_with_filter(
            num=10, seed_topic="tech",
            filter_config={"min_length": 5, "max_length": 200}
        )
        assert len(dataset) >= 0

    def test_generate_with_augmentation(self, pipeline):
        dataset = pipeline.generate_with_augmentation(
            num=5, seed_topic="science", augment_factor=2
        )
        assert len(dataset) >= 1

    def test_full_pipeline(self, pipeline):
        result = pipeline.run(
            num_generate=5,
            seed_topic="programming",
            filter_config={"min_length": 5},
            augment_factor=2,
        )
        assert "dataset" in result
        assert "stats" in result
        assert result["stats"]["generated"] >= 0
        assert result["stats"]["after_filter"] >= 0
        assert result["stats"]["after_augment"] >= 0

    def test_save_dataset(self, pipeline):
        import tempfile, json
        dataset = [{"instruction": "Q", "output": "A"}]
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            pipeline.save_dataset(dataset, f.name)
            with open(f.name) as f2:
                loaded = json.load(f2)
        os.unlink(f.name)
        assert len(loaded) == 1

    def test_export_formats(self, pipeline):
        import tempfile
        dataset = [{"instruction": "Q", "output": "A"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.jsonl")
            pipeline.export_dataset(dataset, path, format="jsonl")
            assert os.path.exists(path)
            path2 = os.path.join(tmp, "data.json")
            pipeline.export_dataset(dataset, path2, format="json")
            assert os.path.exists(path2)

    def test_dataset_statistics(self, pipeline):
        dataset = [
            {"instruction": "What is Python?", "output": "A language."},
            {"instruction": "Explain AI", "output": "AI is..."},
        ]
        stats = pipeline.compute_stats(dataset)
        assert stats["num_examples"] == 2
        assert stats["avg_instruction_len"] > 0
        assert stats["avg_output_len"] > 0

    def test_seed_examples(self, pipeline):
        seeds = pipeline.get_seed_examples(domain="general")
        assert len(seeds) > 0
        for s in seeds:
            assert "instruction" in s
            assert "output" in s
