import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
from torch import Tensor

from .stgs import STGS

class STGSOutput(CausalLMOutputWithPast):
    def __init__(self, *args, **kwargs):
        # Pop our custom fields before calling parent's __init__
        self.stgs_logits = kwargs.pop('stgs_logits', None)
        self.sampled_diff_tokens = kwargs.pop('sampled_diff_tokens', None)
        self.sampled_diff_one_hot = kwargs.pop('sampled_diff_one_hot', None)
        self.temperature = kwargs.pop('temperature', None)
        super().__init__(*args, **kwargs)
    
    stgs_logits: Tensor
    sampled_diff_tokens: Tensor
    sampled_diff_one_hot: Tensor
    temperature: Tensor

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return f"STGSOutput({self.__dict__})"

class STGSDiffModel(PreTrainedModel):
    """
    A wrapper for HuggingFace models that makes token generation differentiable
    using the Straight-Through Gumbel-Softmax trick.
    
    Args:
        model: A HuggingFace PreTrainedModel
        tokenizer: The corresponding tokenizer
        stgs_kwargs: Dictionary of STGS parameters
        device: Device to run the model on
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        stgs_kwargs: Dict[str, bool] = {
            "stgs_hard": False,
            "init_temperature": 1.0,
            "learnable_temperature": False,
            "hidden_state_conditioning": False,
        },
        stgs_logits_generation: Optional[bool] = True,
        device: Optional[str] = None,
    ):
        # Initialize with the config from the base model
        super().__init__(model.config)
        
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        assert self.device == device or device is None 
        
        # Initialize STGS module
        self.stgs_kwargs = stgs_kwargs
        conditioning_dim = 0 if not self.stgs_kwargs["hidden_state_conditioning"] else self.model.config.hidden_size
        self.stgs_logits_generation = stgs_logits_generation
        self.stgs = STGS(
            vocab_size=self.vocab_size,
            stgs_hard=self.stgs_kwargs["hard"],
            init_temperature=self.stgs_kwargs["temperature"],
            learnable_temperature=self.stgs_kwargs["learnable_temperature"],
            conditioning_dim=conditioning_dim,
            device=self.device,
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        input_one_hots: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: bool = True,
        output_past_key_values: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Union[STGSOutput, Tensor]:
        """
        Forward pass with STGS sampling.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            input_one_hots: Optional precomputed one-hot representation
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target token IDs for loss computation
            inputs_embeds: Optional precomputed embeddings
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return an STGSOutput dictionary of outputs or a tensor of differentiable one-hot sampled tokens
            **kwargs: Additional arguments for the model
            
        Returns:
            Union[STGSOutput, Tensor]:
            if return_dict=True, STGSOutput containing:
                - logits: Original model logits
                - stgs_logits: Logits after STGS sampling
                - sampled_tokens: Sampled token IDs (discrete and differentiable)
                - sampled_one_hot: Sampling one-hot (hard or soft, depending on STGS settings, and differentiable)
                - temperature: Sampling temperature
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - loss: Computed loss if labels are provided
            or
            Tensor of differentiable one-hot sampled tokens if return_dict=False
        """

        if input_one_hots is not None:
            assert inputs_embeds is None and input_ids is None
            inputs_embeds = torch.matmul(input_one_hots, self.model.get_input_embeddings().weight)
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            use_cache=True, #output_past_key_values=output_past_key_values,
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
        output_dict: STGSOutput = STGSOutput(
            logits=logits,
            stgs_logits=stgs_logits,
            sampled_diff_tokens=diff_output_ids,
            sampled_diff_one_hot=diff_one_hot,
            temperature=temperature,
            loss=loss, 
        )
        
        if output_hidden_states and hidden_states is not None:
            output_dict.hidden_states = hidden_states
        if output_past_key_values and outputs.past_key_values is not None:
            output_dict.past_key_values = outputs.past_key_values    

        if return_dict:
            return output_dict
        else:
            return diff_one_hot
    
    def generate(
        self,
        input_ids: Optional[Tensor] = None,
        input_one_hots: Optional[Tensor] = None,
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
            input_one_hots: Input one-hot representation (batch_size, seq_len, vocab_size)
            attention_mask: Attention mask (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (overrides the initialized value if provided)
            num_beams: Number of beams for beam search (1 for sampling)
            use_bpttoken: If True, uses Backpropagation Through Tokens with STGS sampling
            **kwargs: Additional arguments for generation
            
        Returns:
            Tensor of differentiable one-hot sampled tokens (batch_size, max_length)
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
        
        # Prepare inputs
        if input_ids is None and input_one_hots is None:
            raise ValueError("At least one of input_ids or input_one_hots must be provided")
        
        if input_ids is not None:
            batch_size = input_ids.size(0)
            input_len = input_ids.size(1)
        elif input_one_hots is not None:
            batch_size = input_one_hots.size(0)
            input_len = input_one_hots.size(1)
        
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = (input_ids != self.pad_token_id).long()
            elif input_one_hots is not None:
                attention_mask = (input_one_hots.sum(-1) != 0).long()
        
        # Get embedding layer
        embedding_layer = self.model.get_input_embeddings()
        
        # Generate inputs_embeds:
        if input_ids is not None:
            inputs_embeds = embedding_layer(input_ids)
        elif input_one_hots is not None:
            inputs_embeds = torch.matmul(input_one_hots, embedding_layer.weight.unsqueeze(0))
        
        all_one_hot = []
        
        # Initial forward pass
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_past_key_values=True,
            use_cache=True,
            return_dict=True,
            **kwargs
        )
        
        next_token_logits = outputs.logits[:,-1:]
        next_token_stgs_logits = outputs.stgs_logits[:,-1:]
        next_token_id = next_token_logits.argmax(-1)
        next_token_stgs_id = outputs.sampled_diff_tokens[:,-1:]
        next_token_one_hot = F.one_hot(next_token_id, num_classes=self.model.config.vocab_size)
        next_token_stgs_one_hot = outputs.sampled_diff_one_hot[:,-1:]
        
        all_one_hot.append(next_token_stgs_one_hot)            

        # Get embeddings for the next token
        if self.stgs_logits_generation:
            next_token_embedding = torch.matmul(all_one_hot[-1], embedding_layer.weight.unsqueeze(0))
        else:
            next_token_logits_distr = torch.softmax(next_token_logits, dim=-1)
            next_token_embedding = torch.matmul(next_token_logits_distr, embedding_layer.weight.unsqueeze(0))
        # batch_size x 1 x embedding_dim

        if not use_bpttoken:
            next_token_embedding = next_token_embedding.detach()
        
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
        ], dim=-1)
        # batch_size x seq_len+1

        # Generation loop with BPTT
        for it in range(1, max_length - input_len + 1):
            # Forward pass with past key values for efficient generation
            outputs = self.forward(
                inputs_embeds=next_token_embedding,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
                **kwargs
            )
            
            # Get elements for ONLY THE NEXT TOKEN:
            next_token_logits = outputs.logits[:,-1:]
            next_token_stgs_logits = outputs.stgs_logits[:,-1:]
            next_token_id = next_token_logits.argmax(-1)
            next_token_stgs_id = outputs.sampled_diff_tokens[:,-1:]
            next_token_one_hot = F.one_hot(next_token_id, num_classes=self.model.config.vocab_size)
            next_token_stgs_one_hot = outputs.sampled_diff_one_hot[:,-1:]
            all_one_hot.append(next_token_stgs_one_hot)

            # Get embeddings for the next token
            #next_token_embedding = torch.matmul(all_one_hot[-1], embedding_layer.weight.unsqueeze(0))
            if self.stgs_logits_generation:
                next_token_embedding = torch.matmul(all_one_hot[-1], embedding_layer.weight.unsqueeze(0))
            else:
                next_token_logits_distr = torch.softmax(next_token_logits, dim=-1)
                next_token_embedding = torch.matmul(next_token_logits_distr, embedding_layer.weight.unsqueeze(0))
            # batch_size x 1 x embedding_dim
            
            if not use_bpttoken:
                next_token_embedding = next_token_embedding.detach()
            
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
            ], dim=-1)
            # batch_size x seq_len+it+1
        
        all_one_hot = torch.cat(all_one_hot, dim=1)
        # batch_size x max_length x vocab_size
        
        # Restore original mode and temperature
        self.train(original_mode)
        self.stgs.init_temperature = original_temp
        
        return all_one_hot
    
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
