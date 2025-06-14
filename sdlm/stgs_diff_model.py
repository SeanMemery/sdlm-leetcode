import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch import Tensor

from .stgs import STGS


class STGSDiffModel(PreTrainedModel):
    """
    A wrapper for HuggingFace models that makes token generation differentiable
    using the Straight-Through Gumbel-Softmax trick.
    
    Args:
        model: A HuggingFace PreTrainedModel
        tokenizer: The corresponding tokenizer
        temperature: Initial temperature for Gumbel-Softmax
        hard: If True, uses hard (discrete) samples in forward pass
        learnable_temperature: If True, makes temperature a learnable parameter
        conditioning_dim: Dimension of conditioning vector for adaptive temperature (0 for fixed temperature)
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        temperature: float = 1.0,
        hard: bool = False,
        learnable_temperature: bool = False,
        conditioning_dim: int = 0,
        device: str = None,
    ):
        # Initialize with the config from the base model
        super().__init__(model.config)
        
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.device = device or next(model.parameters()).device
        
        # Initialize STGS module
        self.stgs = STGS(
            vocab_size=self.vocab_size,
            stgs_hard=hard,
            init_temperature=temperature,
            learnable_temperature=learnable_temperature,
            conditioning_dim=conditioning_dim,
            device=self.device
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Forward pass with STGS sampling.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target token IDs for loss computation
            inputs_embeds: Optional precomputed embeddings
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments for the model
            
        Returns:
            Dict containing:
                - logits: Original model logits
                - stgs_logits: Logits after STGS sampling
                - sampled_tokens: Sampled token IDs (discrete and differentiable)
                - sampled_one_hot: Sampling one-hot (hard or soft, depending on STGS settings, and differentiable)
                - temperature: Sampling temperature
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - loss: Computed loss if labels are provided
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        
        # Apply STGS sampling
        diff_output_ids, diff_one_hot, temperature, stgs_logits = self.stgs(
            logits,
            hidden_states=hidden_states
        )
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Prepare output
        output_dict = {
            'logits': logits,
            'stgs_logits': stgs_logits,
            'sampled_diff_tokens': diff_output_ids,
            'sampled_diff_one_hot': diff_one_hot,
            'temperature': temperature,
            'loss': loss,
        }
        
        if output_hidden_states and hidden_states is not None:
            output_dict['hidden_states'] = hidden_states
            
        return output_dict
    
    def generate(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        max_length: int = 50,
        temperature: Optional[float] = None,
        num_beams: int = 1,
        use_bpttoken: bool = False,
        **kwargs
    ) -> Tensor:
        """
        Generate sequences using the wrapped model with STGS sampling.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (overrides the initialized value if provided)
            num_beams: Number of beams for beam search (1 for sampling)
            use_bpttoken: If True, uses Backpropagation Through Tokens with STGS sampling
            **kwargs: Additional arguments for generation
            
        Returns:
            Tensor of generated token IDs (batch_size, max_length)
        """
        if use_bpttoken and num_beams > 1:
            raise ValueError("BPTT is not compatible with beam search (num_beams > 1)")
            
        # Save original mode and set to eval if not using BPTT
        original_mode = self.training
        if not use_bpttoken:
            self.eval()
        
        # Override temperature if provided
        original_temp = self.stgs.init_temperature
        if temperature is not None:
            self.stgs.init_temperature = temperature
        
        try:
            # Prepare inputs
            if input_ids is None:
                input_ids = torch.tensor([[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]], 
                                       device=self.device)
            
            batch_size = input_ids.size(0)
            
            if attention_mask is None and input_ids is not None:
                attention_mask = (input_ids != self.pad_token_id).long()
            
            # Get embedding layer
            embedding_layer = self.model.get_input_embeddings()
            
            # If using BPTToken, we need to handle the generation differently
            if use_bpttoken:
                # Prepare for BPTToken generation
                past_key_values = None
                all_logits = []
                
                # Initial forward pass
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                    **kwargs
                )
                
                # Get logits and filter if needed
                logits = outputs.logits
                all_logits.append(logits[:, -1:, :])
                
                # Get next token using STGS
                next_token_diff, next_token_one_hot, _, _ = self.stgs(logits[:, -1:, :])
                
                # Get embeddings for the next token
                next_token_embedding = torch.matmul(next_token_one_hot, embedding_layer.weight.unsqueeze(0))
                
                # Update input sequence
                input_ids = torch.cat([input_ids, next_token_diff.long()], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                ], dim=-1)
                
                # Generation loop with BPTT
                for _ in range(1, max_length - input_ids.size(1) + 1):
                    # Forward pass with past key values for efficient generation
                    outputs = self(
                        inputs_embeds=next_token_embedding,
                        attention_mask=attention_mask,
                        past_key_values=outputs.past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    # Get logits
                    logits = outputs.logits
                    all_logits.append(logits)
                    
                    # Get next token using STGS
                    next_token_diff, next_token_one_hot, _, _ = self.stgs(logits)
                    
                    # Get embeddings for the next token
                    next_token_embedding = torch.matmul(next_token_one_hot, embedding_layer.weight.unsqueeze(0))
                    
                    # Update input sequence
                    input_ids = torch.cat([input_ids, next_token_diff.long()], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                    ], dim=-1)
                
                return input_ids
                
            else:
                # Standard generation without BPTT
                for _ in range(max_length - input_ids.size(1)):
                    # Get model outputs
                    outputs = self(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        **kwargs
                    )
                    
                    # Get next token (greedy sampling from STGS distribution)
                    next_token = outputs['sampled_diff_tokens'][:, -1:]
                    
                    # Update input_ids and attention_mask
                    input_ids = torch.cat([input_ids, next_token.long()], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                    ], dim=-1)
                
                return input_ids
                
        finally:
            # Restore original mode and temperature
            self.train(original_mode)
            self.stgs.init_temperature = original_temp
    
    def to(self, device=None, *args, **kwargs):
        """Moves the model to the specified device."""
        if device is not None:
            self.device = device
        self.model = self.model.to(device, *args, **kwargs)
        self.stgs = self.stgs.to(device, *args, **kwargs)
        return self
    
    def train(self, mode: bool = True):
        """Set the model in training mode."""
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set the model in evaluation mode."""
        return self.train(False)
    
    def parameters(self, recurse: bool = True):
        """Returns an iterator over model parameters."""
        yield from self.model.parameters(recurse=recurse)
        yield from self.stgs.parameters()
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Returns an iterator over model parameters with names."""
        for name, param in self.model.named_parameters(prefix=prefix, recurse=recurse):
            yield name, param
        for name, param in self.stgs.named_parameters(prefix=prefix + 'stgs.'):
            yield name, param
    
    def state_dict(self, *args, **kwargs):
        """Returns the state dict including both model and STGS parameters."""
        state_dict = {}
        state_dict.update(self.model.state_dict(*args, **kwargs))
        state_dict.update({f'stgs.{k}': v for k, v in self.stgs.state_dict(*args, **kwargs).items()})
        return state_dict
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Loads the state dict for both model and STGS parameters."""
        model_state = {k: v for k, v in state_dict.items() if not k.startswith('stgs.')}
        stgs_state = {k[5:]: v for k, v in state_dict.items() if k.startswith('stgs.')}
        
        self.model.load_state_dict(model_state, *args, **kwargs)
        self.stgs.load_state_dict(stgs_state, *args, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation."""
        return self.model.prepare_inputs_for_generation(input_ids, **kwargs)
    
    def resize_token_embeddings(self, *args, **kwargs):
        """Resize token embeddings and update vocab size in STGS."""
        model_embeds = self.model.resize_token_embeddings(*args, **kwargs)
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def get_input_embeddings(self):
        """Get the input embeddings."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set the input embeddings."""
        return self.model.set_input_embeddings(value)


# Example usage:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Wrap the model with STGS
stgs_model = STGSWrapper(
    model=model,
    tokenizer=tokenizer,
    temperature=1.0,
    hard=True,
    learnable_temperature=True
)

# Example forward pass
input_text = "The quick brown fox"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = stgs_model(**inputs)

# Example generation
generated = stgs_model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50
)
print(tokenizer.decode(generated[0]))
"""
