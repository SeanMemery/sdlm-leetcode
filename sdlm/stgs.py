"""
Implementation of the Straight-Through Gumbel-Softmax operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor


class STGS(nn.Module):
    """
    Straight-Through Gumbel-Softmax operation that allows for differentiable
    sampling from a categorical distribution with parameterized temperature.
    
    Args:
        vocab_size: Size of the vocabulary
        stgs_hard: If True, uses hard (discrete) samples in forward pass
        init_temperature: Initial temperature for Gumbel-Softmax
        learnable_temperature: If True, makes temperature a learnable parameter
        conditioning_dim: Dimension of conditioning vector for adaptive temperature (0 for fixed temperature)
        eps: Small epsilon for numerical stability
        device: Device to run the computation on
    """
    def __init__(
        self,
        vocab_size: int,
        stgs_hard: bool = False,
        init_temperature: float = 1.0,
        learnable_temperature: bool = False,
        conditioning_dim: int = 0,
        eps: float = 1e-12,
        device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.stgs_hard = stgs_hard
        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        self.conditioning_dim = conditioning_dim
        self.eps = eps
        self.device = device

        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                self.temperature_param = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
            else:
                self.tau_fc = nn.Sequential(
                    nn.Linear(self.conditioning_dim, 1, bias=False),
                    nn.Softplus()
                ).to(device=device)

    def forward(self, x: Tensor, hidden_states: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with STGS sampling.
        
        Args:
            x: Input logits (batch_size, seq_len, vocab_size)
            hidden_states: Optional hidden states for conditional temperature
            
        Returns:
            Tuple of (sampled_tokens, sampled_probs, temperature)
        """
        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                eff_temperature = self.eps + 1. / (F.softplus(self.temperature_param) + 1.0/(self.eps+self.init_temperature))
            else:
                assert hidden_states is not None
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                last_hidden_state = hidden_states[-1][:,-1,:].reshape(batch_size, self.conditioning_dim)
                self.inv_tau0 = 1.0/(self.eps+self.init_temperature)
                eff_temperature = self.eps + 1. / (self.tau_fc(last_hidden_state)+self.inv_tau0).reshape(batch_size, -1, 1)
                eff_temperature = eff_temperature.repeat(1, seq_len, 1)
        else:
            eff_temperature = torch.tensor([self.init_temperature], device=self.device)

        # Gumbel-Softmax sampling
        u = torch.rand_like(x)*(0.999-self.eps)+self.eps
        gumbels = -torch.log( -torch.log(u))
        # batch_size x seq_len x vocab_size

        logits = (x + gumbels)
        y_soft = F.softmax(logits / eff_temperature, dim=-1)
        # batch_size x seq_len x vocab_size

        # Sampling from batched distribution y_soft:
        output_ids = torch.distributions.Categorical(probs=y_soft).sample()
        # batch_size x seq_len

        # Straight-through: use hard in forward, soft in backward
        if self.stgs_hard:
            y_hard = F.one_hot(output_ids, num_classes=self.vocab_size)
            # batch_size x seq_len x vocab_size
            # Type: half or full
            y_hard = y_hard.half() if x.dtype == torch.half else y_hard.float()
            # Straight-through trick: y_hard - y_soft.detach() + y_soft
            output_one_hot = y_hard - y_soft.detach() + y_soft
            # batch_size x seq_len x vocab_size
        else:
            output_one_hot = y_soft
            # batch_size x seq_len x vocab_size
        
        # Type: half or full
        output_one_hot = output_one_hot.half() if x.dtype == torch.half else output_one_hot.float()

        # Differentiable output ids:
        gathered_one_hot = torch.gather(output_one_hot, dim=-1, index=output_ids.unsqueeze(-1))
        diff_output_ids = output_ids.detach()-gathered_one_hot.detach()+gathered_one_hot
        # batch_size x seq_len
        
        return diff_output_ids, output_one_hot, eff_temperature, y_soft
