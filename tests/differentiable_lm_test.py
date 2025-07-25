import sdlm
from sdlm.core.differentiable_lm import DifferentiableLM
from sdlm.core.tensor_string import TensorString

# ==========================================
# Testing and Validation
# ==========================================

def test_prompt_gradient_flow():
    """Test that gradients flow through the differentiable LM."""
    print("Testing gradient flow through DifferentiableLM...")
    
    # Create differentiable LM
    sdlm.configure_default_model("distilbert/distilgpt2")
    lm = DifferentiableLM("distilbert/distilgpt2")
    
    # Create learnable prompt
    prompt = TensorString("User: Say hello to the world.\nAssistant: ", model_name="distilbert/distilgpt2", device=lm.device)
    prompt.requires_grad_(True)
    
    print(f"Prompt: {prompt}")
    print(f"Prompt requires grad: {prompt.requires_grad}")
    
    # Test __call__ interface
    responses = lm(
        prompt=prompt,
        temperature=0.1,
        max_new_tokens=10,
        repetition_penalty=1.2,
    )
    response = responses[0]
    
    print(f"Response: {response}")
    print(f"Response type: {type(response)}")
    print(f"Response requires grad: {response.requires_grad}")
    
    # Compute dummy loss
    target = TensorString("Hello world", model_name="distilbert/distilgpt2", device=lm.device)
    loss = 1.0 - response.similarity_to(target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    try:
        loss.backward()
        grad_computed = prompt.tensor.grad is not None
        print(f"Gradients computed: { '✅' if grad_computed else '❌'}")
        
        if grad_computed:
            print(f"Gradient norm: {prompt.tensor.grad.norm().item():.6f}")
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")

def test_messages_gradient_flow():
    """Test that gradients flow through the differentiable LM using messages."""
    print("Testing gradient flow through DifferentiableLM using messages...")
    
    # Create differentiable LM
    sdlm.configure_default_model("distilbert/distilgpt2")
    lm = DifferentiableLM("distilbert/distilgpt2")
    
    # Create learnable prompt
    messages = [{"role": "user", "content": sdlm.from_string("Say hello to the world.")}]
    messages[0]["content"].requires_grad_(True)
    
    # Test __call__ interface
    responses = lm(
        messages=messages,
        temperature=0.1,
        max_new_tokens=10,
        repetition_penalty=1.2,
    )
    response = responses[0]
    
    print(f"Response: {response}")
    print(f"Response type: {type(response)}")
    print(f"Response requires grad: {response.requires_grad}")
    
    # Compute dummy loss
    target = TensorString("Hello world", model_name="distilbert/distilgpt2", device=lm.device)
    loss = 1.0 - response.similarity_to(target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    try:
        loss.backward()
        grad_computed = messages[0]["content"].tensor.grad is not None
        print(f"Gradients computed: { '✅' if grad_computed else '❌'}")
        
        if grad_computed:
            print(f"Gradient norm: {messages[0]['content'].tensor.grad.norm().item():.6f}")
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
    
def test_dspy_interface():
    """Test DSPy interface compatibility."""
    print("\nTesting DSPy interface compatibility...")
    
    # Test both interfaces
    sdlm.configure_default_model("distilbert/distilgpt2")
    lm = DifferentiableLM("distilbert/distilgpt2")
    
    # Test _messages_to_prompt
    messages = [{"role": "user", "content": sdlm.from_string("Say hello")}]
    prompt = lm._messages_to_prompt(messages)
    print(f"Prompt: {prompt}")
    # Gradient flow maintained?
    print(f"Prompt requires grad: {prompt.requires_grad}")

    # Test __call__ with prompt
    response1 = lm(
        prompt="Test prompt",
        temperature=0.1,
        max_new_tokens=5,
        repetition_penalty=1.2,
    )
    print(f"__call__ with prompt: {response1}")
    
    # Test __call__ with messages
    messages = [{"role": "user", "content": sdlm.from_string("Say hello")}]
    response2 = lm(
        messages=messages,
        temperature=0.1,
        max_new_tokens=5,
        repetition_penalty=1.2,
    )
    print(f"__call__ with messages: {response2}")
    
    print("✅ All interfaces working")

if __name__ == "__main__":
    #test_prompt_gradient_flow()
    test_messages_gradient_flow()
    #test_dspy_interface()

