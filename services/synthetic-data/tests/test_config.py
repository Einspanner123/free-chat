import os, sys, pytest
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestSynthConfig:
    def test_defaults(self):
        from config import SynthConfig, FilterConfig
        cfg = SynthConfig()
        assert cfg.num_seed_examples == 50
        assert cfg.max_generated == 10000
        assert cfg.temperature == 0.8
        fc = FilterConfig()
        assert fc.min_length == 10
        assert fc.max_length == 2048

    def test_custom(self):
        from config import SynthConfig
        cfg = SynthConfig(num_seed_examples=10, max_generated=500, temperature=0.3)
        assert cfg.num_seed_examples == 10
        assert cfg.max_generated == 500

    def test_generation_strategies(self):
        from config import SynthConfig
        cfg = SynthConfig(strategies=["self_instruct", "evol_question", "back_translation"])
        assert len(cfg.strategies) == 3

    def test_filter_config(self):
        from config import FilterConfig
        cfg = FilterConfig()
        assert cfg.min_length == 10
        assert cfg.max_length == 2048
        assert cfg.deduplicate is True
        assert cfg.remove_html is True
