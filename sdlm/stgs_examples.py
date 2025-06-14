"""
Examples demonstrating the STGSDiffModel wrapper for differentiable text generation.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from stgs_wrapper import STGSDiffModel

def example_basic_usage():
    """Basic example showing how to use STGSDiffModel for text generation."""
    print("=== Basic Usage Example ===")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load a pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Wrap the model with STGSDiffModel
    model = STGSDiffModel(
        model=base_model,
        tokenizer=tokenizer,
        temperature=0.7,
        hard=True,  # Use straight-through estimator
        learnable_temperature=False
    )
    
    # Prepare input
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Generate text using STGS sampling
    print("\nGenerating with STGS sampling:")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=30,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print("-" * 80)

def example_differentiable_training():
    """Example showing how to train the model with differentiable sampling."""
    print("\n=== Differentiable Training Example ===")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load a small model for this example
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Wrap the model with STGSDiffModel
    model = STGSDiffModel(
        model=base_model,
        tokenizer=tokenizer,
        temperature=1.0,
        hard=True,
        learnable_temperature=True  # Make temperature learnable
    )
    
    # Example training data
    target_text = "The quick brown fox jumps over the lazy dog."
    target_tokens = tokenizer(target_text, return_tensors="pt")["input_ids"]
    
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, text, num_samples=100):
            self.inputs = tokenizer([text] * num_samples, return_tensors="pt", padding=True, truncation=True, max_length=32)
            
        def __len__(self):
            return len(self.inputs["input_ids"])
            
        def __getitem__(self, idx):
            return {
                "input_ids": self.inputs["input_ids"][idx],
                "attention_mask": self.inputs["attention_mask"][idx],
                "labels": self.inputs["input_ids"][idx]
            }
    
    dataset = SimpleDataset(tokenizer, target_text, num_samples=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 3
    
    print("\nStarting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with STGS sampling
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            # Backward pass
            loss = outputs.loss
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Generate text after training
    print("\nGenerating after training:")
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device),
            max_length=20,
            temperature=0.7
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print("-" * 80)

def example_conditional_temperature():
    """Example showing conditional temperature based on hidden states."""
    print("\n=== Conditional Temperature Example ===")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load a small model for this example
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Wrap the model with STGSDiffModel and enable conditional temperature
    model = STGSDiffModel(
        model=base_model,
        tokenizer=tokenizer,
        temperature=1.0,
        hard=True,
        learnable_temperature=True,
        conditioning_dim=base_model.config.hidden_size  # Use hidden size for conditioning
    )
    
    # Prepare input
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text with conditional temperature
    print("\nGenerating with conditional temperature:")
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=30,
            temperature=0.7  # Initial temperature, will be adjusted by the model
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    # Show how temperature changes based on input
    print("\nTemperature adaptation:")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
    
    print(f"Input: {prompt}")
    print(f"Adaptive temperature: {outputs['temperature'].mean().item():.4f}")
    print("-" * 80)

def example_gradient_flow():
    """Example demonstrating gradient flow through the sampling process."""
    print("\n=== Gradient Flow Example ===")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load a small model for this example
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Wrap the model with STGSDiffModel
    model = STGSDiffModel(
        model=base_model,
        tokenizer=tokenizer,
        temperature=1.0,
        hard=False  # Use soft sampling to see gradients
    )
    
    # Prepare input
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Forward pass with gradient computation
    model.train()
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    
    # Get sampled tokens and their probabilities
    sampled_tokens = outputs['sampled_tokens']
    sampled_probs = outputs['sampled_probs']
    
    # Compute a dummy loss (e.g., maximize probability of specific tokens)
    target_tokens = tokenizer("jumps over", return_tensors="pt")["input_ids"][0, :2]
    target_probs = sampled_probs[0, :2, target_tokens].mean()
    
    # Backward pass
    target_probs.backward()
    
    # Check gradients
    print("\nGradient flow check:")
    print(f"Target tokens: {tokenizer.decode(target_tokens)}")
    print(f"Target probability: {target_probs.item():.4f}")
    print("Gradient for input embeddings:", model.get_input_embeddings().weight.grad.norm().item())
    print("-" * 80)

if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_differentiable_training()
    example_conditional_temperature()
    example_gradient_flow()
