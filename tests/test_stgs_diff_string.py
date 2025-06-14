import torch
import pytest
from sdlm import STGSDiffString

class TestSTGSDiffString:
    def test_initialization(self, test_model_and_tokenizer, test_string, device):
        _, tokenizer = test_model_and_tokenizer
        diff_string = STGSDiffString(
            initial_string=test_string,
            tokenizer=tokenizer,
            device=device
        )
        
        # Test basic properties
        assert isinstance(diff_string.logits, torch.nn.Parameter)
        assert diff_string.logits.requires_grad
        assert diff_string.logits.device == device
        
        # Test string representation
        assert isinstance(str(diff_string), str)
        assert len(diff_string) > 0
    
    def test_forward_pass(self, test_model_and_tokenizer, test_string, device):
        _, tokenizer = test_model_and_tokenizer
        diff_string = STGSDiffString(
            initial_string=test_string,
            tokenizer=tokenizer,
            device=device
        )
        
        # Test forward pass
        input_ids, one_hot, decoded_string = diff_string()
        
        # Check output shapes and types
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(one_hot, torch.Tensor)
        assert isinstance(decoded_string, str)
        assert input_ids.shape[0] == 1  # Batch size 1
        assert one_hot.shape[0] == 1    # Batch size 1
        assert one_hot.shape[-1] == tokenizer.vocab_size
        
        # Test that we can decode back to something reasonable
        assert len(decoded_string) > 0
        
        # Test that one_hot is a valid probability distribution
        assert torch.allclose(one_hot.sum(dim=-1), torch.ones_like(one_hot.sum(dim=-1)))
        assert torch.all(one_hot >= 0)
    
    def test_hard_vs_soft(self, test_model_and_tokenizer, test_string, device):
        _, tokenizer = test_model_and_tokenizer
        
        # Test hard sampling
        hard_string = STGSDiffString(
            initial_string=test_string,
            tokenizer=tokenizer,
            hard=True,
            device=device
        )
        
        # Test soft sampling
        soft_string = STGSDiffString(
            initial_string=test_string,
            tokenizer=tokenizer,
            hard=False,
            device=device
        )
        
        # Get outputs
        _, hard_one_hot, _ = hard_string()
        _, soft_one_hot, _ = soft_string()
        
        # In hard mode, one_hot should be one-hot encoded
        assert torch.all(torch.sum(hard_one_hot > 0.99, dim=-1) == 1)
        
        # In soft mode, one_hot should be a probability distribution
        assert torch.all(soft_one_hot <= 1.0)
        assert torch.all(soft_one_hot >= 0.0)
        assert torch.allclose(
            torch.sum(soft_one_hot, dim=-1),
            torch.ones_like(torch.sum(soft_one_hot, dim=-1))
        )
    
    def test_gradient_flow(self, test_model_and_tokenizer, test_string, device):
        _, tokenizer = test_model_and_tokenizer
        diff_string = STGSDiffString(
            initial_string=test_string,
            tokenizer=tokenizer,
            device=device
        )
        
        # Forward pass
        input_ids, one_hot, _ = diff_string()
        
        # Compute loss and backpropagate
        loss = one_hot.sum()
        loss.backward()
        
        # Check that gradients are flowing back to logits
        assert diff_string.logits.grad is not None
        assert not torch.all(diff_string.logits.grad == 0)
    
    def test_to_device(self, test_model_and_tokenizer, test_string, device):
        _, tokenizer = test_model_and_tokenizer
        
        # Create on CPU first
        diff_string = STGSDiffString(
            initial_string=test_string,
            tokenizer=tokenizer,
            device='cpu'
        )
        
        # Move to target device
        diff_string = diff_string.to(device)
        
        # Check that all tensors are on the correct device
        assert diff_string.device == device
        assert diff_string.logits.device == device
        assert diff_string.input_ids.device == device
        
        # Test forward pass works
        input_ids, one_hot, _ = diff_string()
        assert input_ids.device == device
        assert one_hot.device == device
