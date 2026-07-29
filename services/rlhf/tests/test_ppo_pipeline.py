import os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestRLHFPipeline:
    @pytest.fixture
    def pipeline(self):
        from ppo_pipeline import RLHFPipeline
        from config import RLHFConfig
        from ppo_trainer import PPOTrainer
        cfg = RLHFConfig(base_model="test", reward_model="test")
        p = RLHFPipeline(config=cfg)
        policy = MagicMock()
        policy.generate.return_value.chunk = "response"
        reward = MagicMock()
        reward.score.return_value = 0.5
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [101, 102]
        tokenizer.decode.return_value = "decoded"
        p.set_models(policy, reward, tokenizer)
        return p

    def test_init(self, pipeline):
        assert pipeline is not None
        assert pipeline.config is not None

    def test_collect_rollouts(self, pipeline):
        prompts = ["Q1", "Q2", "Q3"]
        rollouts = pipeline.collect_rollouts(prompts)
        assert len(rollouts) == 3
        for r in rollouts:
            assert "prompt" in r
            assert "response" in r
            assert "reward" in r

    def test_train_iteration(self, pipeline):
        prompts = ["P1", "P2"]
        metrics = pipeline.train_iteration(prompts)
        assert metrics is not None

    def test_full_train(self, pipeline):
        dataset = [{"prompt": f"Q{i}"} for i in range(5)]
        result = pipeline.train(dataset, num_iterations=2)
        assert "final_reward" in result or "metrics" in result

    def test_evaluate(self, pipeline):
        eval_data = [{"prompt": "test", "chosen": "good", "rejected": "bad"}]
        metrics = pipeline.evaluate(eval_data)
        assert metrics is not None

    def test_save_policy(self, pipeline):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.save_policy(tmp)
            assert os.path.exists(tmp)

    def test_end_to_end_mock(self):
        from ppo_pipeline import RLHFPipeline
        from config import RLHFConfig
        cfg = RLHFConfig(base_model="test", reward_model="test")

        p = RLHFPipeline(config=cfg)
        policy = MagicMock()
        policy.generate.return_value.chunk = "resp"
        reward = MagicMock()
        reward.score.return_value = 0.5
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [101]
        p.set_models(policy, reward, tokenizer)

        dataset = [{"prompt": "Q1"}, {"prompt": "Q2"}]
        result = p.train(dataset, num_iterations=1)
        assert result is not None
