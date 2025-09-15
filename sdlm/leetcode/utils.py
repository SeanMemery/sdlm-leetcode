# sdlm/leetcode/utils.py

import re
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..stgs_diff_model import STGSDiffModel
from .prompts_sdlm import SYSTEM_PROMPT_FOR_FIRST_CODE

def extract_python_block(text: str) -> str:
    """Extract Python code from generated text."""
    # First try to extract from markdown code blocks
    if "```python" in text and "```" in text.split("```python")[1]:
        return text.split("```python")[1].split("```")[0].strip()
    
    # If no markdown blocks, look for function definitions
    import re
    func_match = re.search(r'(def\s+\w+.*?)(?=\n\ndef|\n\nclass|\Z)', text, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()
    
    # If no function definition, try to clean up the text to extract code-like content
    lines = text.split('\n')
    code_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip obvious non-code lines and keep potential code
        if (stripped and 
            not stripped.startswith('(input)') and 
            not stripped.startswith('- Input') and
            not stripped.startswith('Output') and
            not '///' in stripped and
            not 'Assistant:' in stripped and
            not 'User:' in stripped):
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Last resort - return as is but cleaned
    return text.strip()

def clean_leetcode_signature(code: str) -> str:
    """Clean code for LeetCode submission by removing asserts and other unwanted elements."""
    # Keep it conservative; avoid adding asserts, prints, etc.
    code = re.sub(r"\bassert\b.*", "", code)
    code = re.sub(r"\bprint\b.*", "", code)
    return code.strip()




def build_coder_and_judge(model_name: str, device, args) -> Tuple[STGSDiffModel, AutoTokenizer]:
    """Build both coder and judge models."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)

    stgs_kwargs = {
        # Provide both key styles to survive the known kwargs quirk
        "hard": args.stgs_hard, "temperature": args.init_temperature,
        "stgs_hard": args.stgs_hard, "init_temperature": args.init_temperature,
        "learnable_temperature": args.learnable_temperature,
        "bpttoken": args.bpttoken, "use_bpttoken": args.bpttoken,
        "hidden_state_conditioning": False,
    }
    model = STGSDiffModel(model=base, tokenizer=tokenizer, stgs_kwargs=stgs_kwargs, device=device)
    return model, tokenizer


def generate_initial_code(coder_model: STGSDiffModel, tokenizer, problem_prompt: str, max_new_tokens=256) -> str:
    """Generate initial code solution using the model directly."""
    
    full_prompt = SYSTEM_PROMPT_FOR_FIRST_CODE + "\n\nUser:\n" + problem_prompt + "\n\nAssistant:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(coder_model.device)

    with torch.no_grad():
        # Use the underlying base model for standard generation
        base_model = coder_model.model
        outputs = base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=inputs["input_ids"].shape[1] + max_new_tokens,
            temperature=0.3,  # Lower temperature for more focused code generation
            do_sample=True,
            top_p=0.9,  # Nucleus sampling for better quality
            top_k=50,   # Top-k sampling
            repetition_penalty=1.1,  # Reduce repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the newly generated part (after the original prompt)
    prompt_length = len(full_prompt)
    if len(decoded) > prompt_length:
        generated_part = decoded[prompt_length:].strip()
    else:
        generated_part = decoded
    
    # Now extract Python code from the generated part
    generated_code = extract_python_block(generated_part)
    return generated_code

