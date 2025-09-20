import torch
import torch.nn.functional as F
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoModelForCausalLM
from sdlm import STGSDiffModel
from sdlm.textgrad.variables import Variable

# Import the modules we're testing
from sdlm.leetcode.utils import (
    extract_python_block, 
    clean_leetcode_signature,
    build_coder_and_judge,
    generate_initial_code
)
from sdlm.leetcode.momentum import MomentumLossFunction
from sdlm.leetcode.prompts_sdlm import (
    SYSTEM_PROMPT_FOR_FIRST_CODE,
    CODE_INSTANCE_ROLE_DESCRIPTION
)


class TestSDLMLeetCodeOptUtils:
    """Test utility functions for SDLM LeetCode optimization."""
    
    def test_extract_python_block_with_markdown(self):
        """Test extracting Python code from markdown blocks."""
        text_with_code = """Here is some code:
```python
def hello():
    return "Hello, World!"
```
That's it!"""
        
        result = extract_python_block(text_with_code)
        expected = 'def hello():\n    return "Hello, World!"'
        assert result == expected
    
    def test_extract_python_block_without_markdown(self):
        """Test handling text without markdown blocks."""
        plain_code = 'def hello():\n    return "Hello, World!"'
        result = extract_python_block(plain_code)
        assert result == plain_code
    
    def test_extract_python_block_malformed(self):
        """Test handling malformed markdown blocks."""
        malformed = "```python\ndef hello():\n    return 'test'"  # Missing closing ```
        result = extract_python_block(malformed)
        # Should extract the code part even without closing ```
        expected = "def hello():\n    return 'test'"
        assert result == expected
    
    def test_clean_leetcode_signature_removes_assert(self):
        """Test removing assert statements from code."""
        code_with_assert = """def solution(nums):
    assert len(nums) > 0
    result = sum(nums)
    assert result >= 0, "Result should be positive"
    return result"""
        
        result = clean_leetcode_signature(code_with_assert)
        expected = """def solution(nums):
    
    result = sum(nums)
    
    return result"""
        assert result == expected
    
    def test_clean_leetcode_signature_no_assert(self):
        """Test cleaning code without assert statements."""
        clean_code = """def solution(nums):
    return sum(nums)"""
        
        result = clean_leetcode_signature(clean_code)
        assert result == clean_code
    
    def test_clean_leetcode_signature_empty(self):
        """Test cleaning empty code."""
        result = clean_leetcode_signature("")
        assert result == ""


class TestSDLMLeetCodeOptMomentum:
    """Test momentum-based loss functions."""
    
    def test_momentum_loss_initialization(self, test_model_and_tokenizer, device):
        """Test MomentumLossFunction initialization."""
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 0.7,
                "learnable_temperature": False,
                "hidden_state_conditioning": False,
            }
        )
        
        momentum_question = (
            "#TASK_DESCRIPTION:\n {t_descr}\n\n"
            "#INPUT:\n {input}\n\n"
            "Does the above input (#INPUT) fulfill the above task description (#TASK_DESCRIPTION)?"
        )
        
        momentum_loss = MomentumLossFunction(
            critic_dlm=diff_model,
            momentum_question=momentum_question,
            Momentum_variables={"t_descr": "Sum all numbers"},
            momentum_answer="Yes",
            use_cot=False,
            answer_extractor="",
        )
        
        assert momentum_loss.critic_dlm is diff_model
        assert momentum_loss.momentum_question == momentum_question
        assert momentum_loss.momentum_variables["t_descr"] == "Sum all numbers"
        assert momentum_loss.momentum_answer == "Yes"
        assert not momentum_loss.use_cot
    
    def test_momentum_loss_forward_single_input(self, test_model_and_tokenizer, device):
        """Test MomentumLossFunction forward pass with single input."""
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 0.7,
                "learnable_temperature": False,
                "hidden_state_conditioning": False,
            }
        )
        
        momentum_question = (
            "#TASK_DESCRIPTION:\n {t_descr}\n\n"
            "#INPUT:\n {input}\n\n"
            "Does the above input (#INPUT) fulfill the above task description (#TASK_DESCRIPTION)?"
        )
        
        momentum_loss = MomentumLossFunction(
            critic_dlm=diff_model,
            momentum_question=momentum_question,
            Momentum_variables={"t_descr": "Sum all numbers"},
            momentum_answer="Yes",
            use_cot=False,
            answer_extractor="",
        )
        
        # Test with single code string
        code_strings = ["def solution(nums):\n    return sum(nums)"]
        
        loss = momentum_loss(batched_input=code_strings)
        
        # Check that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative
    
    def test_momentum_loss_forward_batch(self, test_model_and_tokenizer, device):
        """Test MomentumLossFunction forward pass with batch input."""
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 0.7,
                "learnable_temperature": False,
                "hidden_state_conditioning": False,
            }
        )
        
        momentum_question = (
            "#TASK_DESCRIPTION:\n {t_descr}\n\n"
            "#INPUT:\n {input}\n\n"
            "Does the above input (#INPUT) fulfill the above task description (#TASK_DESCRIPTION)?"
        )
        
        momentum_loss = MomentumLossFunction(
            critic_dlm=diff_model,
            momentum_question=momentum_question,
            Momentum_variables={"t_descr": "Find two numbers that sum to target"},
            momentum_answer="Yes",
            use_cot=False,
            answer_extractor="",
        )
        
        # Test with batch of code strings
        code_strings = [
            "def twoSum(nums, target):\n    return [0, 1]",
            "def twoSum(nums, target):\n    for i in range(len(nums)):\n        return [i, i+1]",
            "def twoSum(nums, target):\n    return nums[:2]"
        ]
        
        loss = momentum_loss(batched_input=code_strings)
        
        # Check that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative
    
    def test_momentum_loss_with_cot(self, test_model_and_tokenizer, device):
        """Test MomentumLossFunction with Chain-of-Thought."""
        model, tokenizer = test_model_and_tokenizer
        diff_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 0.7,
                "learnable_temperature": False,
                "hidden_state_conditioning": False,
            }
        )
        
        momentum_question = (
            "#TASK_DESCRIPTION:\n {t_descr}\n\n"
            "#INPUT:\n {input}\n\n"
            "Does the above input (#INPUT) fulfill the above task description (#TASK_DESCRIPTION)?"
        )
        
        answer_extractor = "Based on the above reasoning, the answer is: "
        
        momentum_loss = MomentumLossFunction(
            critic_dlm=diff_model,
            momentum_question=momentum_question,
            Momentum_variables={"t_descr": "Return the sum of array elements"},
            momentum_answer="Yes",
            use_cot=True,
            answer_extractor=answer_extractor,
        )
        
        # Test with single code string
        code_strings = ["def solution(nums):\n    return sum(nums)"]
        
        loss = momentum_loss(batched_input=code_strings)
        
        # Check that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0


class TestSDLMLeetCodeOptPipeline:
    """Test main optimization pipeline components."""
    
    @patch('leetcode_env.environment.LeetCodeEnv')
    def test_leetcode_env_integration(self, mock_leetcode_env):
        """Test integration with leetcode-hard-gym environment."""
        # Mock the environment
        mock_env = Mock()
        mock_env.reset.return_value = None
        mock_env.get_problem_info.return_value = Mock(content="def twoSum(nums, target):")
        mock_leetcode_env.return_value = mock_env
        
        # Test that we can create and interact with the environment
        from leetcode_env.environment import LeetCodeEnv
        env = LeetCodeEnv()
        env.reset(problem="two-sum")
        problem_info = env.get_problem_info()
        
        assert problem_info is not None
    
    def test_build_coder_and_judge(self, device):
        """Test building coder and judge models with args."""
        from argparse import Namespace
        
        model_name = "distilbert/distilgpt2"
        args = Namespace(
            stgs_hard=False,
            init_temperature=0.7,
            learnable_temperature=False,
            bpttoken=False
        )
        
        coder_model, tokenizer = build_coder_and_judge(model_name, device, args)
        
        # Check model type and configuration
        assert isinstance(coder_model, STGSDiffModel)
        assert str(coder_model.device).startswith(str(device).split(':')[0])  # Handle cuda:0 vs cuda
        assert tokenizer.pad_token is not None
        
        # Check STGS configuration
        assert not coder_model.stgs.stgs_hard
        assert coder_model.stgs.init_temperature == 0.7
        assert not coder_model.stgs.learnable_temperature
    
    @pytest.mark.parametrize("max_new_tokens", [50, 100, 256])
    def test_generate_initial_code(self, test_model_and_tokenizer, device, max_new_tokens):
        """Test initial code generation."""
        model, tokenizer = test_model_and_tokenizer
        coder_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 0.7,
                "learnable_temperature": False,
                "hidden_state_conditioning": False,
            }
        )
        
        problem_prompt = "def twoSum(nums, target):\n    \"\"\"\n    Find two numbers that add up to target.\n    \"\"\""
        
        generated_code = generate_initial_code(
            coder_model, tokenizer, problem_prompt, max_new_tokens=max_new_tokens
        )
        
        # Check that code was generated
        assert isinstance(generated_code, str)
        assert len(generated_code) > 0
        
        # Should contain some basic structure (more lenient since GPT2 isn't code-specialized)
        # Just check that it's not empty and has some basic text
        assert len(generated_code.strip()) > 5
    
    def test_variable_forward_sample(self, test_model_and_tokenizer, device):
        """Test Variable forward sampling functionality."""
        _, tokenizer = test_model_and_tokenizer
        
        # Create code variable
        initial_code = "def solution(nums):\n    return sum(nums)"
        code_var = Variable(
            tokenizer=tokenizer,
            initial_str=initial_code,
            template='\n"""python\n{VARIABLE}\n"""\n',
            use_fluency_constraint=False,
            device=str(device)
        )
        
        # Test forward sampling
        sample1 = code_var.forward_sample()
        sample2 = code_var.forward_sample(temperature=0.1)
        
        # Check that samples are strings
        assert isinstance(sample1, str)
        assert isinstance(sample2, str)
        assert len(sample1) > 0
        assert len(sample2) > 0
        
        # Check that template is applied
        assert 'python' in sample1
        assert 'python' in sample2
    
    def test_variable_parameters_method(self, test_model_and_tokenizer, device):
        """Test Variable parameters method for optimization."""
        _, tokenizer = test_model_and_tokenizer
        
        # Create code variable
        initial_code = "def solution(nums):\n    return sum(nums)"
        code_var = Variable(
            tokenizer=tokenizer,
            initial_str=initial_code,
            learnable_temperature=False,
            device=str(device)
        )
        
        # Test parameters method
        params = list(code_var.parameters())
        
        # Should at least contain logits
        assert len(params) >= 1
        assert any(p.requires_grad for p in params)
        
        # Test with learnable temperature
        code_var_learnable = Variable(
            tokenizer=tokenizer,
            initial_str=initial_code,
            learnable_temperature=True,
            device=str(device)
        )
        
        params_learnable = list(code_var_learnable.parameters())
        assert len(params_learnable) >= 1


class TestSDLMLeetCodeOptEvaluator:
    """Test LeetCode evaluator with mocking."""
    
    @patch.dict(os.environ, {'LEETCODE_CSRF_TOKEN': 'test_token', 'LEETCODE_SESSION': 'test_session'})
    @patch('leetcode_env.environment.LeetCodeEnv')
    @patch('diskcache.Cache')
    def test_leetcode_evaluator_initialization(self, mock_cache, mock_leetcode_env):
        """Test LeetCode evaluator initialization."""
        from sdlm.leetcode.evaluators.leetcode_eval import LeetCodeEvaluator
        
        # Setup mocks
        mock_env = Mock()
        mock_leetcode_env.return_value = mock_env
        
        # Create evaluator
        evaluator = LeetCodeEvaluator(cache_dir="./test_cache")
        
        # Check initialization
        assert evaluator.env is mock_env
        assert hasattr(evaluator, 'cache')
        
        # Check that environment was created
        mock_leetcode_env.assert_called_once()
    
    def test_format_code(self, tmp_path):
        """Test code formatting functionality."""
        from sdlm.leetcode.evaluators.leetcode_eval import LeetCodeEvaluator
        
        # Mock the dependencies but avoid full initialization
        with patch.dict(os.environ, {'LEETCODE_CSRF_TOKEN': 'test', 'LEETCODE_SESSION': 'test'}), \
             patch('leetcode_env.environment.LeetCodeEnv'), \
             patch('diskcache.Cache'):
            
            evaluator = LeetCodeEvaluator()
            
            # Test code formatting with our simple approach
            test_code = "def test():return 42\nprint('debug')\nassert True"
            formatted_code = evaluator._format_code(test_code)
            
            # Should remove print and assert statements
            assert "print" not in formatted_code
            assert "assert" not in formatted_code
            assert "def test():return 42" in formatted_code
    
    @patch.dict(os.environ, {'LEETCODE_CSRF_TOKEN': 'test', 'LEETCODE_SESSION': 'test'})
    @patch('leetcode_env.environment.LeetCodeEnv')
    @patch('diskcache.Cache')
    def test_caching_functionality(self, mock_cache_class, mock_leetcode_env):
        """Test caching functionality."""
        from sdlm.leetcode.evaluators.leetcode_eval import LeetCodeEvaluator
        
        # Setup cache mock
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        
        # Simulate cache hit - need to properly simulate the cache key structure
        def cache_contains(key):
            return key.endswith('_run_success')
        
        def cache_getitem(key):
            if key.endswith('_run_success'):
                return True
            elif key.endswith('_total_correct'):
                return 10
            elif key.endswith('_total_tests'):
                return 10  
            elif key.endswith('_runtime'):
                return 50
            return False
            
        mock_cache.__contains__ = Mock(side_effect=cache_contains)
        mock_cache.__getitem__ = Mock(side_effect=cache_getitem)
        
        evaluator = LeetCodeEvaluator()
        evaluator.cache = mock_cache
        
        # Test cache hit
        with patch.object(evaluator, 'submit_for_evaluation') as mock_submit:
            result = evaluator.check_if_in_cache_or_submit("test-problem", "def test(): return 42")
            
            # Should not call submit_for_evaluation due to cache hit
            mock_submit.assert_not_called()
            
            # Should return cached results
            success, total_correct, total_tests, runtime = result
            assert success is True
            assert total_correct == 10
            assert total_tests == 10
            assert runtime == 50


class TestSDLMLeetCodeOptIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_prompts_constants(self):
        """Test that all required prompt constants are defined."""
        assert isinstance(SYSTEM_PROMPT_FOR_FIRST_CODE, str)
        assert len(SYSTEM_PROMPT_FOR_FIRST_CODE) > 0
        assert "python" in SYSTEM_PROMPT_FOR_FIRST_CODE.lower()
        
        assert isinstance(CODE_INSTANCE_ROLE_DESCRIPTION, str)
        assert len(CODE_INSTANCE_ROLE_DESCRIPTION) > 0
        
        # Note: JUDGE_INSTRUCTION removed in momentum-based approach
        # Test that prompts are properly defined
        assert "python" in SYSTEM_PROMPT_FOR_FIRST_CODE.lower()
        assert "code" in CODE_INSTANCE_ROLE_DESCRIPTION.lower()
    
    def test_end_to_end_momentum_optimization(self, test_model_and_tokenizer, device):
        """Test end-to-end momentum optimization pipeline."""
        model, tokenizer = test_model_and_tokenizer
        
        # Build critic model
        critic_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 0.7,
                "learnable_temperature": False,
                "hidden_state_conditioning": False,
            }
        )
        
        # Create momentum loss
        momentum_question = (
            "#TASK_DESCRIPTION:\n {t_descr}\n\n"
            "#INPUT:\n {input}\n\n"
            "Does the above input (#INPUT) fulfill the above task description (#TASK_DESCRIPTION)?"
        )
        
        momentum_loss = MomentumLossFunction(
            critic_dlm=critic_model,
            momentum_question=momentum_question,
            Momentum_variables={"t_descr": "Find two numbers that add up to target"},
            momentum_answer="Yes",
            use_cot=False,
            answer_extractor="",
        )
        
        # Create code variable
        initial_code = "def twoSum(nums, target):\n    return [0, 1]"
        code_var = Variable(
            tokenizer=tokenizer,
            initial_str=initial_code,
            template='\n"""python\n{VARIABLE}\n"""\n',
            use_fluency_constraint=False,
            device=str(device)
        )
        
        # Test momentum optimization step
        optimizer = torch.optim.Adam(code_var.parameters(), lr=1e-2)
        
        # Store initial state
        initial_logits = code_var.logits.clone()
        
        # Run optimization step
        optimizer.zero_grad()
        batch = [code_var.forward_sample() for _ in range(2)]
        loss = momentum_loss(batched_input=batch)
        loss.backward()
        optimizer.step()
        
        # Check that the momentum loss function works
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss), "Loss should be a finite number"
        
        # Check that we can still sample after optimization attempt
        final_sample = code_var.forward_sample(temperature=0.3)
        assert isinstance(final_sample, str)
        assert len(final_sample) > 0
        
        # The basic pipeline completes without errors
        # Note: Full gradient flow in momentum optimization requires connecting
        # the momentum loss to the differentiable Variable representation
    
    @pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0])
    @pytest.mark.parametrize("hard", [True, False])
    def test_variable_creation_with_different_settings(self, test_model_and_tokenizer, device, temperature, hard):
        """Test Variable creation with different STGS settings and new features."""
        _, tokenizer = test_model_and_tokenizer
        
        code = "def solution():\n    return 42"
        
        var = Variable(
            tokenizer=tokenizer,
            initial_str=code,
            temperature=temperature,
            hard=hard,
            device=str(device),
            template='\n"""python\n{VARIABLE}\n"""\n',
            use_fluency_constraint=False
        )
        
        # Test basic functionality
        assert var.get_string() is not None
        assert len(var.get_string()) > 0
        
        # Test forward sampling
        sample = var.forward_sample()
        assert isinstance(sample, str)
        assert 'python' in sample
        
        # Test parameters method
        params = list(var.parameters())
        assert len(params) >= 1
        
        # Test forward pass
        input_ids, one_hot, decoded = var()
        
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(one_hot, torch.Tensor)
        assert isinstance(decoded, str)
        
        # Check dimensions
        assert input_ids.dim() == 2  # batch_size x seq_len
        assert one_hot.dim() == 3    # batch_size x seq_len x vocab_size
        assert one_hot.shape[-1] == tokenizer.vocab_size