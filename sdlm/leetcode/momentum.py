import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Union, Tuple

class MomentumLossFunction:
    """
    Differentiable momentum-style loss using a critic DLM.
    Supports two input modes:
      - Training: batched_one_hot (list of (1, Lc, V) relaxed one-hots from C_var)
      - Eval/logging: batched_input (list of code strings)
    Loss: cross-entropy on the next token being the first token of momentum_answer.
    """

    def __init__(
        self,
        critic_dlm,
        momentum_question: str,
        Momentum_variables: Dict[str, str],
        momentum_answer: str = "Yes",
        use_cot: bool = False,
        answer_extractor: str = "",
    ):
        self.critic_dlm = critic_dlm
        self.momentum_question = momentum_question
        self.momentum_variables = Momentum_variables or {}
        self.momentum_answer = momentum_answer
        self.use_cot = use_cot
        self.answer_extractor = answer_extractor

        self._pre_ids: Optional[torch.Tensor] = None
        self._post_ids: Optional[torch.Tensor] = None
        self._prepared_for_vocab: Optional[int] = None  # cache keyed by vocab size

    @property
    def device(self):
        return self.critic_dlm.device

    @property
    def tokenizer(self):
        return self.critic_dlm.tokenizer

    def _format_question_parts(self) -> tuple[str, str]:
        if "{input}" not in self.momentum_question:
            raise ValueError("momentum_question must contain the '{input}' placeholder exactly once.")
        safe_vars = {k: v for (k, v) in self.momentum_variables.items() if k != "input"}
        question_with_vars = self.momentum_question.format(**safe_vars, input="{input}")
        pre, post = question_with_vars.split("{input}", maxsplit=1)

        # Also ensure pre isn't accidentally empty after formatting
        if not pre.strip():
            pre = "Question:\n"

        return pre, post

    def _answer_first_token_id(self) -> int:
        tok = self.tokenizer
        # GPT-2/BPE tokenizers are space-sensitive; prepend a space for stable tokenization
        candidate_texts = [self.momentum_answer, " " + self.momentum_answer, self.momentum_answer.strip()]
        for txt in candidate_texts:
            ids = tok(txt, add_special_tokens=False).input_ids
            if ids:
                return ids[0]
        # Fallbacks
        for txt in ["Yes", " Yes", "YES"]:
            ids = tok(txt, add_special_tokens=False).input_ids
            if ids:
                return ids[0]
        raise ValueError("Cannot derive target token id for momentum_answer — tokenization returned empty.")

    def _one_hot_from_ids(self, ids: torch.Tensor, vocab_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # ids: (1, L) -> (1, L, V) with desired dtype/device
        return F.one_hot(ids, num_classes=vocab_size).to(device=device, dtype=dtype)

    def _oh_to_emb(self, one_hot: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # Cast one_hot to embedding dtype if needed, then matmul
        if one_hot.dtype != W.dtype:
            one_hot = one_hot.to(dtype=W.dtype)
        return torch.matmul(one_hot, W.unsqueeze(0))  # (1, L, V) x (V, D) -> (1, L, D)

    def _prepare_pre_post_ids_for_text(self, pre_text: str, post_text: str, vocab_size: int):
        """Prepare pre/post token IDs for given text."""
        tok, device = self.tokenizer, self.device
        pre_ids = tok(pre_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        post_ids = tok(post_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if pre_ids.size(1) == 0:
            pre_ids = tok("Q:", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if post_ids.size(1) == 0:
            post_ids = tok(" A", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        self._pre_ids, self._post_ids = pre_ids, post_ids

    def _forward_differentiable(self, batched_one_hot: Sequence[torch.Tensor]) -> torch.Tensor:
        device = self.device
        model = self.critic_dlm
        W = model.get_input_embeddings().weight  # (V, D)
        vocab_size = W.shape[0]

        target_id = self._answer_first_token_id()
        losses = []

        for code_oh in batched_one_hot:
            # Expect (1, Lc, V) with grad; cast to correct device/dtype
            if code_oh.dim() != 3:
                raise ValueError("Each code sample must be (1, Lc, V) one-hot tensor")
            code_oh = code_oh.to(device=device, dtype=W.dtype)

            # If code has zero tokens, pad with one EOS token (also in W.dtype)
            if code_oh.size(1) == 0:
                eos_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
                pad = torch.zeros((1, 1, vocab_size), device=device, dtype=W.dtype)
                pad[0, 0, eos_id] = 1
                code_oh = pad

            code_emb = self._oh_to_emb(code_oh, W)  # (1, Lc, D)
            
            if self.use_cot:
                loss = self._forward_cot_differentiable(code_emb, W, target_id)
            else:
                loss = self._forward_direct_differentiable(code_emb, W, target_id)
            
            losses.append(loss)

        return torch.stack(losses).mean()
    
    def _forward_direct_differentiable(self, code_emb: torch.Tensor, W: torch.Tensor, target_id: int) -> torch.Tensor:
        """Direct evaluation without chain of thought."""
        device = self.device
        model = self.critic_dlm
        vocab_size = W.shape[0]
        
        pre, post = self._format_question_parts()
        post = post + "\nAnswer:"
        if self.answer_extractor:
            post = post + " " + self.answer_extractor
            
        self._prepare_pre_post_ids_for_text(pre, post, vocab_size)
        
        pre_oh = self._one_hot_from_ids(self._pre_ids, vocab_size, dtype=W.dtype, device=device)
        post_oh = self._one_hot_from_ids(self._post_ids, vocab_size, dtype=W.dtype, device=device)
        pre_emb = self._oh_to_emb(pre_oh, W)
        post_emb = self._oh_to_emb(post_oh, W)
        
        inputs_embeds = torch.cat([pre_emb, code_emb, post_emb], dim=1)
        if inputs_embeds.size(1) == 0:
            return torch.zeros((), device=device, dtype=W.dtype).sum()

        attn = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=device)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn, return_dict=True)
        next_logits = outputs.logits[:, -1, :]  # (1, V)
        return self._loss_from_next_token_logits(next_logits, target_id)
    
    def _forward_cot_differentiable(self, code_emb: torch.Tensor, W: torch.Tensor, target_id: int) -> torch.Tensor:
        """Chain of thought evaluation: first generate reasoning, then get answer."""
        device = self.device
        model = self.critic_dlm
        vocab_size = W.shape[0]
        
        pre, post = self._format_question_parts()
        reasoning_prompt = post + "\n\nLet me think step by step.\n"
        
        # Step 1: Generate reasoning
        reasoning_pre_ids = self.tokenizer(pre, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        reasoning_post_ids = self.tokenizer(reasoning_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        reasoning_pre_oh = self._one_hot_from_ids(reasoning_pre_ids, vocab_size, dtype=W.dtype, device=device)
        reasoning_post_oh = self._one_hot_from_ids(reasoning_post_ids, vocab_size, dtype=W.dtype, device=device)
        reasoning_pre_emb = self._oh_to_emb(reasoning_pre_oh, W)
        reasoning_post_emb = self._oh_to_emb(reasoning_post_oh, W)
        
        reasoning_inputs = torch.cat([reasoning_pre_emb, code_emb, reasoning_post_emb], dim=1)
        reasoning_attn = torch.ones(reasoning_inputs.shape[:-1], dtype=torch.long, device=device)
        
        # Generate reasoning (limit to ~250 tokens to keep it manageable)
        with torch.no_grad():
            # The issue is in the STGS model's generate method, bypass it completely
            if hasattr(model, 'model'):
                # This is an STGSDiffModel wrapper, use the underlying transformers model
                base_model = model.model
            else:
                base_model = model
            
            reasoning_output = base_model.generate(
                inputs_embeds=reasoning_inputs,
                attention_mask=reasoning_attn,
                max_new_tokens=50,  # Reduce from 250 to prevent freezing
                max_length=reasoning_inputs.shape[1] + 50,  # Explicit max length
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,  # Prevent repetition loops
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,  # Stop at EOS token
            )
        
        # Step 2: Get final answer
        reasoning_text = self.tokenizer.decode(reasoning_output[0], skip_special_tokens=True)
        answer_prompt = reasoning_text + f"\n\nBased on the above reasoning, the answer is "
        
        answer_inputs = self.tokenizer(answer_prompt, return_tensors="pt").to(device)
        answer_outputs = model(
            input_ids=answer_inputs["input_ids"],
            attention_mask=answer_inputs.get("attention_mask"),
            return_dict=True
        )
        next_logits = answer_outputs.logits[:, -1, :]  # (1, V)
        return self._loss_from_next_token_logits(next_logits, target_id)
    
    def _loss_from_next_token_logits(self, next_logits: torch.Tensor, target_id: int) -> torch.Tensor:
        # next_logits: (1, V)
        yes_target = next_logits.new_tensor([target_id]).long()
        return F.cross_entropy(next_logits, yes_target)

    def _forward_strings(self, batched_input: Sequence[str]) -> torch.Tensor:
        """
        Non-differentiable convenience path for logging/eval (tokenizes strings).
        """
        tok = self.tokenizer
        device = self.device
        model = self.critic_dlm
        target_id = self._answer_first_token_id()

        losses = []
        for code_string in batched_input:
            if self.use_cot:
                loss = self._forward_cot_strings(code_string, target_id)
            else:
                loss = self._forward_direct_strings(code_string, target_id)
            losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    
    def _forward_direct_strings(self, code_string: str, target_id: int) -> torch.Tensor:
        """Direct evaluation for string input."""
        tok = self.tokenizer
        device = self.device
        model = self.critic_dlm
        
        pre, post = self._format_question_parts()
        post = post + "\nAnswer:"
        if self.answer_extractor:
            post = post + " " + self.answer_extractor
            
        full_prompt = f"{pre}{code_string}{post}"
        inputs = tok(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=2048
        ).to(device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True
        )
        next_logits = outputs.logits[:, -1, :]  # (1, V)
        return self._loss_from_next_token_logits(next_logits, target_id)
    
    def _forward_cot_strings(self, code_string: str, target_id: int) -> torch.Tensor:
        """Chain of thought evaluation for string input."""
        tok = self.tokenizer
        device = self.device
        model = self.critic_dlm
        
        pre, post = self._format_question_parts()
        reasoning_prompt = f"{pre}{code_string}{post}\n\nLet me think step by step.\n"
        
        # Step 1: Generate reasoning
        reasoning_inputs = tok(
            reasoning_prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=1024
        ).to(device)
        
        with torch.no_grad():
            # The issue is in the STGS model's generate method, not the input tensors
            # We need to bypass the STGS wrapper entirely and use the base transformers model
            if hasattr(model, 'model'):
                # This is an STGSDiffModel wrapper, use the underlying transformers model
                base_model = model.model
            else:
                base_model = model
            
            reasoning_output = base_model.generate(
                input_ids=reasoning_inputs["input_ids"],
                attention_mask=reasoning_inputs.get("attention_mask"),
                max_new_tokens=30,  # Reduce to prevent freezing
                max_length=reasoning_inputs["input_ids"].shape[1] + 30,  # Explicit max length
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,  # Prevent repetition loops
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                early_stopping=True,  # Stop at EOS token
            )
        
        # Step 2: Get final answer
        reasoning_text = tok.decode(reasoning_output[0], skip_special_tokens=True)
        answer_prompt = reasoning_text + f"\n\nBased on the above reasoning, the answer is "
        
        answer_inputs = tok(answer_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        answer_outputs = model(
            input_ids=answer_inputs["input_ids"],
            attention_mask=answer_inputs.get("attention_mask"),
            return_dict=True
        )
        next_logits = answer_outputs.logits[:, -1, :]  # (1, V)
        return self._loss_from_next_token_logits(next_logits, target_id)

    def __call__(
        self,
        batched_one_hot: Optional[Sequence[torch.Tensor]] = None,
        batched_input: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        """
        Prefer batched_one_hot for training. Use batched_input for eval/logging.
        """
        if batched_one_hot is not None:
            return self._forward_differentiable(batched_one_hot)
        if batched_input is not None:
            # Do not backpropagate through this path (typically called under torch.no_grad()).
            return self._forward_strings(batched_input)
        raise ValueError("Provide either batched_one_hot (preferred for training) or batched_input (for eval).")