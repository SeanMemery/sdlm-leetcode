import torch
import torch.nn.functional as F
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
    
    @pytest.mark.parametrize("stgs_hard", [True, False])
    @pytest.mark.parametrize("init_temperature", [0.1, 1.0, 2.0])
    @pytest.mark.parametrize("learnable_temperature", [True, False])
    @pytest.mark.parametrize("hidden_state_conditioning", [True, False])
    @pytest.mark.parametrize("with_ids", [True, False])
    def test_forward_pass(
        self,
        test_model_and_tokenizer,
        device,
        stgs_hard,
        init_temperature,
        learnable_temperature,
        hidden_state_conditioning,
        with_ids,
    ):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": stgs_hard,
                "init_temperature": init_temperature,
                "learnable_temperature": learnable_temperature,
                "hidden_state_conditioning": hidden_state_conditioning,
            },
        )
        
        # Create test input
        input_text = "This is a test"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_key = 'input_ids' if with_ids else 'input_one_hots'

        if not with_ids:
            inputs['input_one_hots'] = F.one_hot(inputs['input_ids'], num_classes=tokenizer.vocab_size).float()
            del inputs['input_ids']

        # Test forward pass
        outputs = diff_model(**inputs, return_dict=True)
        
        # Check output keys
        expected_keys = {
            'logits', 'hidden_states', 'past_key_values',
            'sampled_diff_tokens', 'sampled_diff_one_hot',
            'temperature', 'stgs_logits'
        }
        assert all(key in outputs for key in expected_keys)
        
        # Check shapes
        batch_size, seq_len = inputs[input_key].shape[:2]
        assert outputs['logits'].shape == (batch_size, seq_len, tokenizer.vocab_size)
        assert outputs['sampled_diff_tokens'].shape == (batch_size, seq_len)
        assert outputs['sampled_diff_one_hot'].shape == (batch_size, seq_len, tokenizer.vocab_size)
    
    @pytest.mark.parametrize("use_bpttoken", [True, False])
    @pytest.mark.parametrize("stgs_hard", [True, False])
    @pytest.mark.parametrize("init_temperature", [0.1, 1.0, 2.0])
    @pytest.mark.parametrize("learnable_temperature", [True, False])
    @pytest.mark.parametrize("hidden_state_conditioning", [True, False])
    @pytest.mark.parametrize("with_ids", [True, False])
    def test_generate(
        self,
        test_model_and_tokenizer,
        device,
        use_bpttoken,
        stgs_hard,
        init_temperature,
        learnable_temperature,
        hidden_state_conditioning,
        with_ids,
    ):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": stgs_hard,
                "init_temperature": init_temperature,
                "learnable_temperature": learnable_temperature,
                "hidden_state_conditioning": hidden_state_conditioning,
            },
        )
        
        input_text = "This is a test"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_key = 'input_ids' if with_ids else 'input_one_hots'

        if not with_ids:
            inputs['input_one_hots'] = F.one_hot(inputs['input_ids'], num_classes=tokenizer.vocab_size).float()
            del inputs['input_ids']
        
        output_one_hots = diff_model.generate(
            **inputs,
            max_length=20,
            use_bpttoken=use_bpttoken
        )
        
        # Basic output validation
        assert isinstance(output_one_hots, torch.Tensor)
        assert output_one_hots.shape[0] == 1  # Batch size 1
        assert output_one_hots.shape[1] <= 20  # Requested max_length
        
    @pytest.mark.parametrize("stgs_hard", [True, False])
    @pytest.mark.parametrize("init_temperature", [0.1, 1.0, 2.0])
    @pytest.mark.parametrize("learnable_temperature", [True, False])
    @pytest.mark.parametrize("hidden_state_conditioning", [True, False])
    @pytest.mark.parametrize("with_ids", [True, False])
    def test_gradient_flow(
        self,
        test_model_and_tokenizer,
        device,
        stgs_hard,
        init_temperature,
        learnable_temperature,
        hidden_state_conditioning,
        with_ids,
    ):
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": stgs_hard,
                "init_temperature": init_temperature,
                "learnable_temperature": learnable_temperature,
                "hidden_state_conditioning": hidden_state_conditioning,
            },
        )
        
        # Create test input
        input_text = "This is a test"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        input_key = 'input_ids' if with_ids else 'input_one_hots'

        if not with_ids:
            inputs['input_one_hots'] = F.one_hot(inputs['input_ids'], num_classes=tokenizer.vocab_size).float()
            del inputs['input_ids']
        
        # Forward pass
        outputs = diff_model(**inputs)
        output_one_hots = outputs.sampled_diff_one_hot
        
        # Compute entropy loss and backpropagate
        entropy_loss = -torch.mean(output_one_hots*output_one_hots.log())
        entropy_loss.backward()
        
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
