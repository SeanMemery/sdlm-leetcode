import torch
import pytest
from transformers import AutoModelForCausalLM
from sdlm import STGSDiffModel

class TestSTGSDiffModel:
    def test_initialization(self, test_model_and_tokenizer, device):
        model, tokenizer = test_model_and_tokenizer
        
        # Test initialization with default parameters
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Check that model and tokenizer are set correctly
        assert diff_model.model is model
        assert diff_model.tokenizer is tokenizer
        assert diff_model.device == device
        
        # Check default parameters
        assert diff_model.stgs.init_temperature == 1.0
        assert diff_model.stgs.stgs_hard is False
    
    def test_forward_pass(self, test_model_and_tokenizer, device):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Create test input
        input_text = "This is a test"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Test forward pass
        outputs = diff_model(**inputs)
        
        # Check output keys
        expected_keys = {
            'logits', 'hidden_states', 'past_key_values',
            'sampled_diff_tokens', 'sampled_diff_one_hot',
            'temperature', 'stgs_logits'
        }
        assert all(key in outputs for key in expected_keys)
        
        # Check shapes
        batch_size, seq_len = inputs['input_ids'].shape
        assert outputs['logits'].shape == (batch_size, seq_len, tokenizer.vocab_size)
        assert outputs['sampled_diff_tokens'].shape == (batch_size, seq_len)
        assert outputs['sampled_diff_one_hot'].shape == (batch_size, seq_len, tokenizer.vocab_size)
    
    def test_generate_standard(self, test_model_and_tokenizer, device):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Test standard generation
        input_text = "This is a test"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate with standard sampling
        output_ids = diff_model.generate(
            input_ids=input_ids,
            max_length=20,
            use_bpttoken=False
        )
        
        # Check output shape and type
        assert isinstance(output_ids, torch.Tensor)
        assert output_ids.shape[0] == 1  # Batch size 1
        assert output_ids.shape[1] == 20  # Requested max_length
        
        # Check that input is preserved at the beginning
        input_len = input_ids.shape[1]
        assert torch.all(output_ids[0, :input_len] == input_ids[0])
    
    @pytest.mark.parametrize("use_bpttoken", [True, False])
    def test_generate_with_bpttoken(self, test_model_and_tokenizer, device, use_bpttoken):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Test generation with BPTToken
        input_text = "This is a test"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate with BPTToken
        output_ids = diff_model.generate(
            input_ids=input_ids,
            max_length=20,
            use_bpttoken=use_bpttoken
        )
        
        # Basic output validation
        assert isinstance(output_ids, torch.Tensor)
        assert output_ids.shape[0] == 1  # Batch size 1
        assert output_ids.shape[1] == 20  # Requested max_length
        
        # Check that input is preserved at the beginning
        input_len = input_ids.shape[1]
        assert torch.all(output_ids[0, :input_len] == input_ids[0])
    
    def test_gradient_flow(self, test_model_and_tokenizer, device):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Create test input
        input_text = "This is a test"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Forward pass
        outputs = diff_model(**inputs)
        
        # Compute loss and backpropagate
        loss = outputs['sampled_diff_one_hot'].sum()
        loss.backward()
        
        # Check that gradients are flowing
        for name, param in diff_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0), f"Zero gradients for parameter: {name}"
    
    def test_save_load_state_dict(self, test_model_and_tokenizer, device, tmp_path):
        model, tokenizer = test_model_and_tokenizer
        
        # Create and train model
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Save state dict
        state_dict = diff_model.state_dict()
        
        # Create new model
        new_diff_model = STGSDiffModel(
            model=AutoModelForCausalLM.from_config(model.config),
            tokenizer=tokenizer,
            device=device
        )
        
        # Load state dict
        new_diff_model.load_state_dict(state_dict)
        
        # Check that parameters match
        for (k1, v1), (k2, v2) in zip(
            diff_model.named_parameters(),
            new_diff_model.named_parameters()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2, atol=1e-6)
