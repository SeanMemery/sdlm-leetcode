from sdlm.core.patching import DSPyPatcher, TensorStringContext, StringPatching
from sdlm.core.tensor_string import TensorString
import pytest

def test_selective_patching():
    """Test that selective patching doesn't break other packages."""
    print("Testing selective patching strategies...")
    
    # Test 1: DSPy-specific patching
    print("\n1. Testing DSPy-specific patching")
    dspy_patcher = DSPyPatcher()
    dspy_patcher.activate()
    
    # Test that transformers still works
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # This should use regular strings, not TensorString
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"✅ Transformers works: '{test_text}' -> {tokens} -> '{decoded}'")
        print(f"   Decoded type: {type(decoded)}")
        
    except Exception as e:
        print(f"❌ Transformers broken: {e}")
    
    # Test 2: Context manager approach
    print("\n2. Testing context manager approach")
    with TensorStringContext():
        # Inside context: str should be TensorString
        test_str = "This should be TensorString"
        print(f"Inside context: '{test_str}' (type: {type(test_str)})")
    
    # Outside context: str should be normal
    test_str_outside = "This should be normal str"
    print(f"Outside context: '{test_str_outside}' (type: {type(test_str_outside)})")
    
    # Test 3: Check that HuggingFace still works
    print("\n3. Final HuggingFace compatibility check")
    try:
        # This should not be affected by our patching
        normal_string = "Normal string operation"
        upper_string = normal_string.upper()
        
        print(f"✅ Normal string ops work: {type(normal_string)} -> {type(upper_string)}")
        
    except Exception as e:
        print(f"❌ Normal string ops broken: {e}")
    
    dspy_patcher.deactivate()
    print("\n🎉 Selective patching tests completed!")

def test_string_join_patching():
    """Test that string join patching works with TensorString."""
    print("\nTesting string join patching...")
    
    # Create test strings
    ts1 = TensorString("Hello")
    ts2 = TensorString("world")
    
    # Test 1: Basic join with TensorString elements
    try:
        result = ", ".join([ts1, ts2])
        print(f"✅ Basic join works: {result}")
        print(f"   Result type: {type(result)}")
        assert isinstance(result, TensorString)
        assert str(result) == "Hello, world"
    except Exception as e:
        print(f"❌ Basic join failed: {e}")
        raise
    
    # Test 2: Mixed string and TensorString elements
    try:
        result = " ".join([ts1, "beautiful", ts2])
        print(f"✅ Mixed types join works: {result}")
        print(f"   Result type: {type(result)}")
        assert isinstance(result, TensorString)
        assert str(result) == "Hello beautiful world"
    except Exception as e:
        print(f"❌ Mixed types join failed: {e}")
        raise
    
    # Test 3: Newline separator
    try:
        result = "\n".join([ts1, ts2])
        print(f"✅ Newline join works: {result!r}")
        assert str(result) == "Hello\nworld"
    except Exception as e:
        print(f"❌ Newline join failed: {e}")
        raise

def test_string_patching_unpatch():
    """Test that unpatching restores original string behavior."""
    print("\nTesting string unpatching...")
    
    # Get original join method
    original_join = str.join
    
    # Create patcher and patch
    patcher = StringPatching()
    patcher.patch_string_methods()
    
    # Verify patched
    assert str.join != original_join
    
    # Unpatch
    patcher.unpatch_string_methods()
    
    # Verify restored
    assert str.join == original_join
    print("✅ String unpatching works correctly")

if __name__ == "__main__":
    test_selective_patching()
    test_string_join_patching()
    test_string_patching_unpatch()
    print("\n✅ All patching tests passed!")
