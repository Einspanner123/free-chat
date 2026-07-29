import os, sys, pytest
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestPPOConfig:
    def test_defaults(self):
        from config import PPOConfig
        cfg = PPOConfig()
        assert cfg.learning_rate == 1e-5
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.kl_penalty == 0.05
        assert cfg.clip_range == 0.2
        assert cfg.vf_coef == 0.1
        assert cfg.num_epochs == 4
    def test_custom(self):
        from config import PPOConfig
        cfg = PPOConfig(learning_rate=5e-6, kl_penalty=0.1, clip_range=0.15)
        assert cfg.learning_rate == 5e-6
        assert cfg.kl_penalty == 0.1
class TestRLHFConfig:
    def test_defaults(self):
        from config import RLHFConfig
        cfg = RLHFConfig()
        assert cfg.base_model is not None
        assert cfg.reward_model is not None
