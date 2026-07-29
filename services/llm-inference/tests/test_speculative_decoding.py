import os, sys, pytest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

class TestSpeculativeDecoding:
    @pytest.fixture
    def draft(self):
        m = MagicMock()
        m.generate.return_value.chunk = "draft output"
        return m

    @pytest.fixture
    def target(self):
        m = MagicMock()
        m.generate.return_value.chunk = "target output"
        return m

    def test_init(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=5)
        assert sd.gamma == 5
        assert sd.draft_model == draft
        assert sd.target_model == target

    def test_draft_generation(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=3)
        tokens = ["the", "quick", "brown"]
        accepted = sd.verify(tokens, "the quick brown")
        assert len(accepted) > 0

    def test_rejection_sampling(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=4)
        q_values = [0.9, 0.8, 0.3, 0.1]
        p_values = [0.85, 0.75, 0.6, 0.5]
        accepted, n_accepted = sd.rejection_sampling(q_values, p_values)
        assert n_accepted >= 0
        assert n_accepted <= len(q_values)

    def test_speculate(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=3)
        # Mock rejection_sampling to return actual tokens
        with patch.object(sd, 'rejection_sampling') as mock_rs:
            mock_rs.return_value = ([True, True, False], 2)
            with patch.object(sd, '_estimate_draft_probs', return_value=[0.9]*3):
                with patch.object(sd, '_estimate_target_probs', return_value=[0.8]*3):
                    result = sd.speculate("Hello world")
                    assert result is not None

    def test_speedup_estimate(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=5)
        # With acceptance rate 0.8, expected speedup ≈ 1/(1-0.8+0.8/5) = 1/0.36 ≈ 2.78
        speedup = sd.estimate_speedup(acceptance_rate=0.8)
        assert abs(speedup - 2.78) < 0.1

    def test_theoretical_speedup(self):
        from optimization.speculative_decoding import SpeculativeDecoder
        # gamma=1 → no speedup (draft and target are None, not used for speedup calc)
        draft_mock = MagicMock()
        target_mock = MagicMock()
        sd = SpeculativeDecoder(draft_model=draft_mock, target_model=target_mock, gamma=1)
        assert sd.estimate_speedup(acceptance_rate=0.8) == 1.0

    def test_gamma_tuning(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=5)
        for gamma in [2, 3, 4, 5, 6]:
            sd.gamma = gamma
            assert sd.gamma == gamma

    def test_worst_case_speedup(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=5)
        # acceptance_rate = 0 → no tokens accepted
        # speedup = 1/(1-0+0/5) = 1/1 = 1.0
        speedup = sd.estimate_speedup(acceptance_rate=0.0)
        # With γ=5 and α=0, speedup <= 1.0 (no benefit, but no penalty either)
        assert speedup <= 1.0

    def test_best_case_speedup(self, draft, target):
        from optimization.speculative_decoding import SpeculativeDecoder
        sd = SpeculativeDecoder(draft_model=draft, target_model=target, gamma=5)
        speedup = sd.estimate_speedup(acceptance_rate=1.0)
        assert speedup > 1.0
