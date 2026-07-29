import math, os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestPPOTrainer:
    def test_init(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(
            policy_model=MagicMock(),
            reward_model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert trainer is not None

    def test_compute_advantages(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        rewards = [0.5, 0.8, 0.3]
        values = [0.4, 0.7, 0.4]
        advantages, returns = trainer.compute_advantages(rewards, values, gamma=0.99, lam=0.95)
        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)

    def test_gae_computation(self):
        """GAE (Generalized Advantage Estimation) correctness."""
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.7, 0.9]
        advantages, returns = trainer.compute_advantages(rewards, values, gamma=0.9, lam=0.95)
        # With gamma=0.9, lam=0.95, the last advantage should be ~rewards[-1] - values[-1]
        assert abs(advantages[-1] - (rewards[-1] - values[-1])) < 0.01

    def test_kl_divergence(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        # Identical distributions → KL = 0
        log_probs_a = [-1.0, -2.0, -3.0]
        log_probs_b = [-1.0, -2.0, -3.0]
        kl = trainer.compute_kl_divergence(log_probs_a, log_probs_b)
        assert abs(kl) < 1e-6

    def test_kl_divergence_different(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        log_probs_a = [-1.0, -2.0, -3.0]
        log_probs_b = [-2.0, -3.0, -4.0]
        kl = trainer.compute_kl_divergence(log_probs_a, log_probs_b)
        assert kl > 0

    def test_ppo_loss(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        log_probs = [-1.0, -2.0, -3.0]
        old_log_probs = [-1.5, -2.5, -3.5]
        advantages = [1.0, 0.5, -0.5]
        loss = trainer.compute_ppo_loss(log_probs, old_log_probs, advantages)
        # Loss can be negative (PPO maximizes reward), just check it's finite
        assert isinstance(loss, float)
        assert not math.isnan(loss)
        assert not math.isinf(loss)

    def test_ppo_loss_clipping(self):
        """PPO loss clipping: ratio should be clipped to [1-clip, 1+clip]."""
        from ppo_trainer import PPOTrainer
        import math
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        # Very different log probs → ratio is extreme, will be clipped
        log_probs = [5.0, 5.0]
        old_log_probs = [-5.0, -5.0]
        advantages = [1.0, 1.0]
        loss = trainer.compute_ppo_loss(log_probs, old_log_probs, advantages, clip_range=0.2)
        # Clipped loss should be finite (not NaN/Inf)
        assert isinstance(loss, float)
        assert not math.isnan(loss)
        assert not math.isinf(loss)

    def test_kl_penalized_reward(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        reward = 1.0
        kl = 0.5
        penalized = trainer.apply_kl_penalty(reward, kl, kl_penalty=0.05)
        assert penalized < reward
        assert penalized > 0

    def test_kl_penalty_adaptive(self):
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
        target_kl = 0.1
        # KL is 60% above target → should increase
        current_kl = 0.16  # ratio = 1.6 > 1.5
        new_coef = trainer.adapt_kl_coefficient(0.05, current_kl, target_kl)
        assert new_coef > 0.05

        # KL is 60% below target → should decrease
        current_kl = 0.04  # ratio = 0.4 < 0.5
        new_coef = trainer.adapt_kl_coefficient(0.05, current_kl, target_kl)
        assert new_coef < 0.05

        # KL is within target range → should stay
        current_kl = 0.08  # ratio = 0.8, between 0.5 and 1.5
        new_coef = trainer.adapt_kl_coefficient(0.05, current_kl, target_kl)
        assert new_coef == 0.05

    def test_generate_rollout(self):
        from ppo_trainer import PPOTrainer
        mock_policy = MagicMock()
        mock_policy.generate.return_value.chunk = "Generated response."
        mock_reward = MagicMock()
        mock_reward.score.return_value = 0.8
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 102, 103]
        mock_tokenizer.decode.return_value = "Generated response."

        trainer = PPOTrainer(mock_policy, mock_reward, mock_tokenizer)
        rollout = trainer.generate_rollout("Test prompt")
        assert "prompt" in rollout
        assert "response" in rollout
        assert "reward" in rollout
        assert rollout["reward"] == 0.8

    def test_train_step(self):
        from ppo_trainer import PPOTrainer
        with patch('ppo_trainer.PPOTrainer.generate_rollout') as mock_rollout:
            mock_rollout.return_value = {
                "prompt": "test",
                "response": "output",
                "reward": 0.5,
                "log_probs": [-1.0, -2.0],
                "values": [0.4, 0.6],
            }
            trainer = PPOTrainer(MagicMock(), MagicMock(), MagicMock())
            metrics = trainer.train_step(["prompt1", "prompt2"])
            assert metrics is not None
