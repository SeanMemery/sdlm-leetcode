import torch
import pytest
from sdlm.stgs import STGS

class TestSTGS:
    @pytest.mark.parametrize("batch_size,seq_len,vocab_size", [
        (1, 10, 100),
        (4, 20, 50257),  # GPT-2 vocab size
        (8, 1, 1000)
    ])
    def test_forward_shape(self, device, batch_size, seq_len, vocab_size):
        stgs = STGS(vocab_size=vocab_size, device=device)
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        
        output_ids, one_hot, temperature, probs = stgs(logits)
        
        assert output_ids.shape == (batch_size, seq_len)
        assert one_hot.shape == (batch_size, seq_len, vocab_size)
        assert probs.shape == (batch_size, seq_len, vocab_size)
        assert temperature.numel() == 1

    def test_hard_sampling(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, stgs_hard=True, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device)
        
        output_ids, one_hot, _, _ = stgs(logits)
        
        # In hard mode, one_hot should be one-hot encoded
        assert torch.all(torch.sum(one_hot, dim=-1) == 1.0)
        assert torch.all(torch.sum(one_hot > 0, dim=-1) == 1)

    def test_soft_sampling(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, stgs_hard=False, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device)
        
        _, one_hot, _, _ = stgs(logits)
        
        # In soft mode, one_hot should be a probability distribution
        assert torch.allclose(torch.sum(one_hot, dim=-1), torch.ones(1, 5, device=device))
        assert torch.all(one_hot >= 0)
        assert torch.all(one_hot <= 1)

    def test_temperature_effect(self, device):
        vocab_size = 10
        logits = torch.tensor([[[1.0] + [0.0] * (vocab_size - 1)]], device=device)
        
        # High temperature should make the distribution more uniform
        stgs_high = STGS(vocab_size=vocab_size, init_temperature=10.0, device=device)
        _, one_hot_high, _, _ = stgs_high(logits)
        
        # Low temperature should make the distribution more peaked
        stgs_low = STGS(vocab_size=vocab_size, init_temperature=0.1, device=device)
        _, one_hot_low, _, _ = stgs_low(logits)
        
        # The max probability should be higher with lower temperature
        max_prob_high = torch.max(one_hot_high).item()
        max_prob_low = torch.max(one_hot_low).item()
        assert max_prob_low > max_prob_high

    def test_gradient_flow(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device, requires_grad=True)
        
        output_ids, one_hot, _, _ = stgs(logits)
        loss = one_hot.sum()
        loss.backward()
        
        # Check that gradients are flowing back to logits
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)
