"""
Text loss functions for gradient-based text optimization.
"""

import torch
import torch.nn.functional as F
from typing import Union, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer


class TextLoss:
    """
    Collection of text-based loss functions for gradient-based optimization.
    """
    
    @staticmethod
    def similarity_loss(
        pred: Union[str, torch.Tensor],
        target: Union[str, List[int], torch.Tensor],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        metric: str = 'cosine',
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute similarity-based loss between generated and target text.
        
        Args:
            pred: Predicted text (str) or diff_one_hot tensor (batch_size, seq_len, vocab_size)
            target: Target text (str) or token IDs (batch_size, seq_len)
            model: Model to use for embeddings
            tokenizer: Tokenizer for text processing
            metric: Similarity metric ('cosine' or 'mse')
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
            
        Returns:
            Loss tensor
        """
        # Get embeddings for both texts
        def get_embeddings(text, is_target=False):
            if isinstance(text, str):
                # For string inputs, use the tokenizer
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                # Use last hidden state and mean pool
                return outputs.hidden_states[-1].mean(dim=1)
            elif isinstance(text, torch.Tensor):
                # For target, assume it's one_hot (batch_size, seq_len, vocab_size)
                # Get embeddings for each token
                embeddings = model.model.get_input_embeddings().weight  # (vocab_size, hidden_size)
                # Compute weighted sum of embeddings using diff_one_hot
                input_embeds = torch.matmul(text, embeddings)  # (batch_size, seq_len, hidden_size)
                # Forward pass with input_embeds
                outputs = model(inputs_embeds=input_embeds, output_hidden_states=True)
                # Mean pool the hidden states
                return outputs.hidden_states[-1].mean(dim=1)
            else:
                raise ValueError(f"Unsupported input type: {type(text)}")
        
        pred_emb = get_embeddings(pred)
        target_emb = get_embeddings(target)
        
        # Compute similarity
        if metric == 'cosine':
            similarity = F.cosine_similarity(pred_emb, target_emb, dim=-1)
            loss = 1 - similarity
        elif metric == 'mse':
            loss = F.mse_loss(pred_emb, target_emb, reduction='none').mean(dim=-1)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    
    @staticmethod
    def fluency_loss(
        text: Union[str, torch.Tensor],
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute fluency loss using language model perplexity.
        
        Args:
            text: Input text (str) or diff_one_hot tensor (batch_size, seq_len, vocab_size)
            model: Language model for computing fluency
            tokenizer: Tokenizer (required if text is a string)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
            
        Returns:
            Perplexity loss tensor
        """
        if isinstance(text, str):
            if tokenizer is None:
                raise ValueError("tokenizer is required when text is a string")
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            return model(**inputs, labels=inputs["input_ids"].clone()).loss
        elif isinstance(text, torch.Tensor):
            # Assume text is diff_one_hot (batch_size, seq_len, vocab_size)
            # Get embeddings for each token
            embeddings = model.get_input_embeddings().weight  # (vocab_size, hidden_size)
            # Compute weighted sum of embeddings using diff_one_hot
            input_embeds = torch.matmul(text, embeddings)  # (batch_size, seq_len, hidden_size)
            
            # For language modeling, we need to shift the inputs and labels
            # Inputs:   [BOS] x1 x2 ... xn-1
            # Labels:   x1    x2 ... xn    [EOS]
            
            # Create labels (shifted input)
            # For now, we'll just use the most likely token as the target
            # This is a simplification and could be improved
            labels = text.argmax(dim=-1)  # (batch_size, seq_len)
            
            # Forward pass with input_embeds
            outputs = model(
                inputs_embeds=input_embeds,
                labels=labels,
                output_hidden_states=True
            )
            return outputs.loss
        else:
            raise ValueError(f"text must be string or tensor, got {type(text)}")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"].clone())
            
        # Cross-entropy loss is already averaged over sequence length
        # and batch size if reduction='mean' (default)
        if reduction == 'none':
            return outputs.loss.unsqueeze(0)  # Ensure at least 1D
        return outputs.loss
    
    @staticmethod
    def classifier_loss(
        text: Union[str, torch.Tensor],
        classifier: PreTrainedModel,
        target_class: Union[int, torch.Tensor],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute loss based on classifier output.
        
        Args:
            text: Input text (str) or diff_one_hot tensor (batch_size, seq_len, vocab_size)
            classifier: Text classifier model
            target_class: Target class index or indices
            tokenizer: Tokenizer (required if text is a string)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
            
        Returns:
            Classification loss tensor
        """
        if isinstance(text, str):
            if tokenizer is None:
                raise ValueError("tokenizer is required when text is a string")
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(classifier.device) for k, v in inputs.items()}
            logits = classifier(**inputs).logits
        elif isinstance(text, torch.Tensor):
            # Assume text is diff_one_hot (batch_size, seq_len, vocab_size)
            # Get embeddings for each token
            embeddings = classifier.get_input_embeddings().weight  # (vocab_size, hidden_size)
            # Compute weighted sum of embeddings using diff_one_hot
            input_embeds = torch.matmul(text, embeddings)  # (batch_size, seq_len, hidden_size)
            
            # For classification, we typically use the [CLS] token or mean pooling
            # Here we'll use mean pooling for simplicity
            attention_mask = torch.ones(text.size(0), text.size(1), device=text.device)
            
            # Forward pass with input_embeds
            outputs = classifier(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = outputs.logits
        else:
            raise ValueError(f"text must be string or tensor, got {type(text)}")
            
        if isinstance(target_class, int):
            target = torch.tensor([target_class], device=classifier.device)
        else:
            target = target_class.to(classifier.device)
            
        if logits.dim() > 2:  # Handle sequence classification
            logits = logits[:, 0, :]  # Use first token for now
            
        loss = F.cross_entropy(
            logits, 
            target,
            reduction=reduction
        )
        
        return loss
