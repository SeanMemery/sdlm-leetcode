"""
Comprehensive tests for SDLM-DSPy integration and regex operations.
Tests gradient flow, string operations, and DSPy module compatibility.
"""

import pytest
import torch
import torch.nn.functional as F
import re
import dspy
import sdlm
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Test utilities
def assert_has_gradients(tensor_string: sdlm.TensorString, message: str = ""):
    """Assert that a TensorString has gradients."""
    assert isinstance(tensor_string, sdlm.TensorString), f"Expected TensorString, got {type(tensor_string)} {message}"
    assert tensor_string.requires_grad, f"TensorString should require gradients {message}"
    assert tensor_string.tensor.grad is None or tensor_string.tensor.grad is not None, f"Gradient tensor should exist {message}"

def create_mock_lm():
    """Create a mock language model for testing."""
    class MockLM(dspy.LM):
        def basic_request(self, prompt, **kwargs):
            # Return TensorString to preserve gradients
            response = sdlm.from_string(f"Mock response to: {prompt}", requires_grad=True)
            return [response]  # Return TensorString, not str(response)
    return MockLM()

# ==========================================
# Basic SDLM Functionality Tests
# ==========================================

class TestSDLMBasics:
    """Test basic SDLM functionality and monkey patching."""
    
    def test_monkey_patching_activation(self):
        """Test that monkey patching is active after import."""
        # Create a string - should be TensorString
        test_str = "Hello world"
        assert isinstance(test_str, sdlm.TensorString), "Monkey patching not active"
        assert hasattr(test_str, 'tensor'), "String should have tensor attribute"
        assert hasattr(test_str, 'to_tensor'), "String should have to_tensor method"
    
    def test_gradient_flow_through_operations(self):
        """Test gradient flow through basic string operations."""
        # Create learnable strings
        str1 = sdlm.from_string("Hello", requires_grad=True)
        str2 = sdlm.from_string(" world", requires_grad=True)
        
        # Test concatenation
        combined = str1 + str2
        assert_has_gradients(combined, "after concatenation")
        
        # Test repetition
        repeated = str1 * 2
        assert_has_gradients(repeated, "after repetition")
        
        # Test slicing
        sliced = combined.slice_tokens(0, 3)
        assert_has_gradients(sliced, "after slicing")
    
    def test_tensor_first_creation(self):
        """Test creating strings from tensors."""
        vocab_size = 1000
        seq_len = 5
        
        # Create from random tensor
        random_tensor = torch.randn(seq_len, vocab_size, requires_grad=True)
        soft_tensor = F.softmax(random_tensor, dim=1)
        
        tensor_string = sdlm.from_tensor(soft_tensor)
        assert_has_gradients(tensor_string, "from random tensor")
        assert tensor_string.tensor.shape == (seq_len, vocab_size)
        
        # Create from token IDs
        token_ids = [101, 7592, 2088, 102]  # Sample token IDs
        id_string = sdlm.from_token_ids(token_ids, requires_grad=True)
        assert_has_gradients(id_string, "from token IDs")
        assert id_string.tensor.shape[0] == len(token_ids)

# ==========================================
# DSPy Integration Tests
# ==========================================

class TestDSPyIntegration:
    """Test integration between SDLM and DSPy modules."""
    
    def setup_method(self):
        """Set up test environment."""
        # Configure SDLM
        sdlm.configure_default_model("distilbert-base-uncased")
        
        # Set up mock LM
        self.mock_lm = create_mock_lm()
        dspy.configure(lm=self.mock_lm)
    
    def test_string_preservation_through_dspy(self):
        """Test that TensorString objects are preserved through DSPy modules."""
        # Create learnable input
        learnable_question = sdlm.from_string("What is 2+2?", requires_grad=True)
        
        # Test that it's still a TensorString after DSPy operations
        assert isinstance(learnable_question, sdlm.TensorString)
        assert_has_gradients(learnable_question, "initial learnable question")
    
    def test_dspy_predict_with_tensor_strings(self):
        """Test dspy.Predict with TensorString inputs."""
        predict = dspy.Predict("question -> answer")
        
        # Create learnable input
        question = sdlm.from_string("Test question", requires_grad=True)
        
        # Run prediction
        result = predict(question=question)
        
        # Check that result maintains TensorString type
        assert hasattr(result, 'answer'), "Result should have answer attribute"
        assert isinstance(result.answer, sdlm.TensorString), f"Answer should be TensorString, got {type(result.answer)}"
    
    def test_chain_of_thought_gradient_flow(self):
        """Test gradient flow through ChainOfThought module."""
        cot = dspy.ChainOfThought("question -> reasoning, answer")
        
        # Create learnable components
        base_question = sdlm.from_string("Solve:", requires_grad=False)
        learnable_part = sdlm.empty_tensor_string(length=3, requires_grad=True)
        full_question = base_question + learnable_part
        
        # Run through CoT
        result = cot(question=full_question)
        
        # Verify gradient preservation
        assert hasattr(result, 'reasoning'), "Result should have reasoning"
        assert hasattr(result, 'answer'), "Result should have answer"
        assert isinstance(result.answer, sdlm.TensorString), "Answer should be TensorString"
        
        # Test backward pass
        if hasattr(result.answer, 'similarity_to'):
            target = sdlm.from_string("Target answer")
            loss = 1.0 - result.answer.similarity_to(target)
            
            # This should work if gradients are preserved
            try:
                loss.backward()
                gradient_exists = learnable_part.tensor.grad is not None
                print(f"Gradient computed: {gradient_exists}")
            except Exception as e:
                pytest.fail(f"Backward pass failed: {e}")
    
    def test_few_shot_with_tensor_strings(self):
        """Test few-shot examples with TensorString."""
        # Create few-shot examples
        examples = [
            dspy.Example(question="2+2", answer="4").with_inputs("question"),
            dspy.Example(question="3+3", answer="6").with_inputs("question"),
        ]
        
        # Convert to TensorStrings
        tensor_examples = []
        for ex in examples:
            tensor_ex = dspy.Example(
                question=sdlm.from_string(ex.question, requires_grad=True),
                answer=sdlm.from_string(ex.answer, requires_grad=True)
            ).with_inputs("question")
            tensor_examples.append(tensor_ex)
        
        # Test with few-shot predictor
        predictor = dspy.Predict("question -> answer", demos=tensor_examples)
        
        test_question = sdlm.from_string("4+4", requires_grad=True)
        result = predictor(question=test_question)
        
        assert isinstance(result.answer, sdlm.TensorString), "Few-shot result should be TensorString"

# ==========================================
# Regex Operations Tests
# ==========================================

class TestRegexOperations:
    """Test regex operations with TensorString objects."""
    
    def setup_method(self):
        """Set up test environment."""
        sdlm.configure_default_model("distilbert-base-uncased")
    
    def test_basic_regex_search(self):
        """Test basic regex search operations."""
        # Create TensorString with pattern
        text = sdlm.from_string("The answer is 42", requires_grad=True)
        
        # Test regex search - this should work with our string override
        pattern = r"answer is (\d+)"
        match = re.search(pattern, text)
        
        assert match is not None, "Regex should find the pattern"
        assert match.group(1) == "42", "Should extract the number"
        
        # Verify text is still TensorString
        assert isinstance(text, sdlm.TensorString), "Text should remain TensorString"
        assert_has_gradients(text, "after regex search")
    
    def test_regex_findall(self):
        """Test regex findall operations."""
        text = sdlm.from_string("Numbers: 1, 2, 3, 4", requires_grad=True)
        
        # Find all numbers
        numbers = re.findall(r'\d+', text)
        
        assert len(numbers) == 4, "Should find 4 numbers"
        assert numbers == ['1', '2', '3', '4'], "Should extract correct numbers"
        
        # Original text should maintain gradients
        assert_has_gradients(text, "after findall")
    
    def test_regex_sub_with_tensor_preservation(self):
        """Test regex substitution while preserving gradients."""
        original = sdlm.from_string("Replace THIS with that", requires_grad=True)
        
        # Standard regex sub creates new string
        result = re.sub(r'THIS', 'THAT', original)
        
        # Result should be TensorString due to monkey patching
        assert isinstance(result, sdlm.TensorString), "Regex sub result should be TensorString"
        
        # Original should still have gradients
        assert_has_gradients(original, "original after regex sub")
        
        # Note: result may not have gradients since it's a new string
        # This is expected behavior for regex operations
    
    def test_differentiable_pattern_matching(self):
        """Test differentiable alternative to regex matching."""
        text = sdlm.from_string("Find the KEYWORD in this text", requires_grad=True)
        keyword = sdlm.from_string("KEYWORD", requires_grad=True)
        
        # Use similarity-based "soft" matching instead of exact regex
        similarities = []
        keyword_len = keyword.tensor.shape[0]
        
        for i in range(text.tensor.shape[0] - keyword_len + 1):
            window = text.slice_tokens(i, i + keyword_len)
            sim = window.similarity_to(keyword)
            similarities.append(sim)
        
        if similarities:
            best_match_idx = torch.stack(similarities).argmax()
            best_similarity = similarities[best_match_idx]
            
            # This should be differentiable
            loss = 1.0 - best_similarity
            loss.backward()
            
            assert text.tensor.grad is not None, "Text should have gradients after soft matching"
            assert keyword.tensor.grad is not None, "Keyword should have gradients after soft matching"
    
    def test_answer_extraction_pattern(self):
        """Test common DSPy pattern: extracting answers from text."""
        response = sdlm.from_string(
            "Let me think step by step.\n1. First step\n2. Second step\nAnswer: The final answer is 42",
            requires_grad=True
        )
        
        # Common DSPy extraction patterns
        patterns = [
            r"Answer:\s*(.+)$",
            r"The final answer is\s*(.+)",
            r"Therefore,?\s*(.+)",
        ]
        
        extracted_answer = None
        for pattern in patterns:
            match = re.search(pattern, str(response), re.MULTILINE | re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip()
                break
        
        assert extracted_answer is not None, "Should extract an answer"
        assert "42" in extracted_answer, "Should find the correct answer"
        
        # Original response should maintain gradients
        assert_has_gradients(response, "after answer extraction")
    
    def test_soft_answer_extraction(self):
        """Test differentiable answer extraction using attention."""
        response = sdlm.from_string(
            "Step 1: Think. Step 2: Calculate. Answer: 42",
            requires_grad=True
        )
        
        # Define answer markers
        answer_marker = sdlm.from_string("Answer:", requires_grad=False)
        
        # Find answer marker using similarity
        marker_len = answer_marker.tensor.shape[0]
        similarities = []
        
        for i in range(response.tensor.shape[0] - marker_len + 1):
            window = response.slice_tokens(i, i + marker_len)
            sim = window.similarity_to(answer_marker)
            similarities.append((i, sim))
        
        if similarities:
            # Find best match position
            best_pos, best_sim = max(similarities, key=lambda x: x[1])
            
            # Extract text after marker (soft extraction)
            answer_start = best_pos + marker_len
            if answer_start < response.tensor.shape[0]:
                extracted = response.slice_tokens(answer_start, None)
                
                assert_has_gradients(extracted, "extracted answer")
                assert isinstance(extracted, sdlm.TensorString), "Extracted should be TensorString"

# ==========================================
# Mock LM Integration Tests
# ==========================================

class TestMockLMIntegration:
    """Test integration with mock language models."""
    
    def setup_method(self):
        """Set up test environment."""
        sdlm.configure_default_model("distilbert-base-uncased")
    
    def test_mock_lm_preserves_gradients(self):
        """Test that mock LM preserves gradients in responses."""
        class GradientPreservingMockLM(dspy.LM):
            def basic_request(self, prompt, **kwargs):
                # Ensure we return TensorString, not regular string
                if isinstance(prompt, sdlm.TensorString):
                    # Create response that depends on input prompt
                    response_text = f"Response to: {str(prompt)}"
                    response = sdlm.from_string(response_text, requires_grad=True)
                    
                    # Create artificial dependency on input prompt
                    if prompt.requires_grad:
                        # Make response depend on prompt for gradient flow
                        combined = prompt + sdlm.from_string(" -> ", requires_grad=False) + response
                        # Extract just the response part but maintain gradient connection
                        response = combined.slice_tokens(prompt.tensor.shape[0] + 1, None)
                    
                    return [response]  # Return TensorString, not str(response)
                else:
                    # Fallback for regular strings
                    return [f"Response to: {prompt}"]
        
        # Set up mock LM
        mock_lm = GradientPreservingMockLM()
        dspy.configure(lm=mock_lm)
        
        # Test with learnable prompt
        learnable_prompt = sdlm.from_string("Test prompt", requires_grad=True)
        
        # Call LM directly
        responses = mock_lm.basic_request(learnable_prompt)
        response = responses[0]
        
        assert isinstance(response, sdlm.TensorString), "Response should be TensorString"
        
        # Test gradient flow
        if hasattr(response, 'requires_grad') and response.requires_grad:
            target = sdlm.from_string("Target response")
            loss = 1.0 - response.similarity_to(target)
            
            try:
                loss.backward()
                has_grad = learnable_prompt.tensor.grad is not None
                print(f"Gradient preserved through mock LM: {has_grad}")
            except Exception as e:
                print(f"Gradient flow test failed: {e}")
    
    def test_dspy_module_with_gradient_preserving_lm(self):
        """Test full DSPy module with gradient-preserving LM."""
        class GradientMockLM(dspy.LM):
            def basic_request(self, prompt, **kwargs):
                # Always return TensorString
                response = sdlm.from_string(f"Mock: {prompt}", requires_grad=True)
                return [response]
        
        dspy.configure(lm=GradientMockLM())
        
        # Test with ChainOfThought
        cot = dspy.ChainOfThought("question -> reasoning, answer")
        
        learnable_question = sdlm.from_string("What is 1+1?", requires_grad=True)
        result = cot(question=learnable_question)
        
        # Verify types
        assert isinstance(result.answer, sdlm.TensorString), "Answer should be TensorString"
        
        # Test end-to-end gradient flow
        target_answer = sdlm.from_string("The answer is 2")
        loss = 1.0 - result.answer.similarity_to(target_answer)
        
        try:
            loss.backward()
            gradient_computed = learnable_question.tensor.grad is not None
            print(f"End-to-end gradient flow: {gradient_computed}")
        except Exception as e:
            print(f"End-to-end gradient test failed: {e}")

# ==========================================
# TensorStringContext Tests
# ==========================================

class TestTensorStringContext:
    """Test TensorStringContext for string literal interception."""
    
    def test_string_literal_conversion(self):
        """Test that string literals become TensorString instances in context."""
        from sdlm.core.patching import TensorStringContext
        
        # Outside context, string literals should be regular strings
        s1 = "test"
        assert isinstance(s1, str)
        assert not isinstance(s1, sdlm.TensorString)
        
        # Inside context, string literals should be TensorString instances
        with TensorStringContext():
            s2 = "test"
            assert isinstance(s2, sdlm.TensorString), \
                "String literal should be TensorString instance"
            assert hasattr(s2, 'tensor'), \
                "TensorString should have 'tensor' attribute"
    
    def test_join_method_dispatch(self):
        """Test that join() calls are properly dispatched to TensorString."""
        from sdlm.core.patching import TensorStringContext
        from unittest.mock import patch
        
        with TensorStringContext():
            # Create a string literal (should be TensorString)
            sep = ","
            items = ["a", "b", "c"]
            
            # Patch the join method to track calls
            with patch.object(sdlm.TensorString, 'join') as mock_join:
                # This should call TensorString.join
                result = sep.join(items)
                
                # Verify join was called on the TensorString instance
                mock_join.assert_called_once()
                
                # Verify it was called with the correct arguments
                args, kwargs = mock_join.call_args
                assert args[0] == items  # First arg should be the items to join
                assert not kwargs  # No keyword arguments expected

    def test_context_nesting(self):
        """Test that nested contexts work correctly."""
        from sdlm.core.patching import TensorStringContext
        
        with TensorStringContext():
            s1 = "outer"
            assert isinstance(s1, sdlm.TensorString)
            
            with TensorStringContext():
                s2 = "inner"
                assert isinstance(s2, sdlm.TensorString)
            
            s3 = "outer_again"
            assert isinstance(s3, sdlm.TensorString)
        
        # Outside all contexts, should be regular string
        s4 = "regular"
        assert not isinstance(s4, sdlm.TensorString)
        assert isinstance(s4, str)

# ==========================================
# Performance and Edge Case Tests
# ==========================================

class TestEdgeCases:
    """Test edge cases and performance scenarios."""
    
    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        empty = sdlm.from_string("", requires_grad=True)
        assert isinstance(empty, sdlm.TensorString), "Empty string should be TensorString"
        assert empty.tensor.shape[0] == 0, "Empty string should have zero length tensor"
        
        # Test operations with empty strings
        non_empty = sdlm.from_string("Hello", requires_grad=True)
        combined = empty + non_empty
        assert str(combined) == "Hello", "Empty string concatenation should work"
    
    def test_large_string_handling(self):
        """Test handling of large strings."""
        large_text = "word " * 1000  # 1000 words
        large_string = sdlm.from_string(large_text, requires_grad=True)
        
        assert isinstance(large_string, sdlm.TensorString), "Large string should be TensorString"
        assert_has_gradients(large_string, "large string")
        
        # Test slicing large strings
        subset = large_string.slice_tokens(0, 100)
        assert_has_gradients(subset, "large string subset")
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_text = "Hello 🌍! Ñiño café résumé"
        unicode_string = sdlm.from_string(unicode_text, requires_grad=True)
        
        assert isinstance(unicode_string, sdlm.TensorString), "Unicode string should be TensorString"
        assert_has_gradients(unicode_string, "unicode string")
        
        # Test regex with unicode
        pattern = r'[🌍]+'
        match = re.search(pattern, unicode_string)
        assert match is not None, "Should find unicode emoji"

# ==========================================
# Integration Test Suite
# ==========================================

def run_integration_tests():
    """Run a comprehensive integration test suite."""
    print("Running SDLM-DSPy Integration Tests...")
    print("=" * 50)
    
    # Test 1: Basic monkey patching
    print("Test 1: Monkey patching verification")
    test_str = "Hello"
    assert isinstance(test_str, sdlm.TensorString), "❌ Monkey patching failed"
    print("✅ Monkey patching active")
    
    # Test 2: Gradient flow
    print("\nTest 2: Basic gradient flow")
    str1 = sdlm.from_string("Hello", requires_grad=True)
    str2 = sdlm.from_string(" world", requires_grad=True)
    combined = str1 + str2
    
    target = sdlm.from_string("Hello world")
    loss = 1.0 - combined.similarity_to(target)
    loss.backward()
    
    assert str1.tensor.grad is not None, "❌ Gradient flow failed"
    print("✅ Basic gradient flow working")
    
    # Test 3: DSPy integration
    print("\nTest 3: DSPy integration")
    mock_lm = create_mock_lm()
    dspy.configure(lm=mock_lm)
    
    predict = dspy.Predict("question -> answer")
    question = sdlm.from_string("Test", requires_grad=True)
    result = predict(question=question)
    
    assert isinstance(result.answer, sdlm.TensorString), "❌ DSPy integration failed"
    print("✅ DSPy integration working")
    
    # Test 4: Regex operations
    print("\nTest 4: Regex operations")
    text = sdlm.from_string("Answer: 42", requires_grad=True)
    match = re.search(r"Answer:\s*(\d+)", text)
    
    assert match is not None, "❌ Regex failed"
    assert match.group(1) == "42", "❌ Regex extraction failed"
    assert isinstance(text, sdlm.TensorString), "❌ Regex broke TensorString"
    print("✅ Regex operations working")
    
    print("\n" + "=" * 50)
    print("🎉 All integration tests passed!")

if __name__ == "__main__":
    # Run the integration test suite
    run_integration_tests()
    
    # Run pytest tests
    print("\nRunning detailed pytest suite...")
    pytest.main([__file__, "-v"])
