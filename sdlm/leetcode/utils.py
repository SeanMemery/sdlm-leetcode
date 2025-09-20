import re
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ..stgs_diff_model import STGSDiffModel

def extract_python_block(text: str) -> str:
    if "```python" in text and "```" in text.split("```python")[1]:
        return text.split("```python")[1].split("```")[0].strip()
    m = re.search(r'(def\s+\w+.*?)(?=\n\ndef|\n\nclass|\Z)', text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def clean_for_submission(code: str) -> str:
    """Clean code for LeetCode submission by fixing common syntax issues."""
    # Remove assert statements and print calls
    code = re.sub(r"\bassert\b.*", "", code)
    code = re.sub(r"\bprint\s*\(.*?\)", "", code)
    
    # Fix unterminated triple quotes
    # Count triple quotes and add closing if needed
    triple_single_count = code.count("'''")
    triple_double_count = code.count('"""')
    
    if triple_single_count % 2 == 1:
        code += "\n'''"
    if triple_double_count % 2 == 1:
        code += '\n"""'
    
    # Fix unterminated regular quotes
    # Count unescaped quotes
    single_quote_count = 0
    double_quote_count = 0
    i = 0
    while i < len(code):
        if code[i] == "'" and (i == 0 or code[i-1] != "\\"):
            single_quote_count += 1
        elif code[i] == '"' and (i == 0 or code[i-1] != "\\"):
            double_quote_count += 1
        i += 1
    
    if single_quote_count % 2 == 1:
        code += "'"
    if double_quote_count % 2 == 1:
        code += '"'
    
    # Remove incomplete lines that might cause syntax errors
    lines = code.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.rstrip()
        # Skip lines that end with incomplete constructs
        if line.endswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'with ')):
            continue
        # Skip lines that are just opening brackets/parens without closing
        if line.strip() in ['(', '[', '{']:
            continue
        clean_lines.append(line)
    
    code = '\n'.join(clean_lines)
    
    # Basic syntax check and fallback
    try:
        import ast
        ast.parse(code)
    except SyntaxError:
        # If still has syntax errors, try to extract just function definitions
        func_match = re.search(r'(def\s+\w+.*?)(?=\n\ndef|\n\nclass|\Z)', code, re.DOTALL)
        if func_match:
            code = func_match.group(1).strip()
        else:
            # Last resort: return a simple stub
            code = "def solution():\n    pass"
    
    return code.strip()

def build_model(model_name: str, device: str, stgs_kwargs: dict) -> Tuple[STGSDiffModel, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.attn_implementation = "eager"
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        attn_implementation="eager",
        torch_dtype=(torch.bfloat16 if device != "cpu" else torch.float32),
    ).to(device)
    model = STGSDiffModel(model=base, tokenizer=tok, stgs_kwargs=stgs_kwargs, device=device)
    return model, tok

def generate_initial_code(coder_model: STGSDiffModel, tokenizer, problem_prompt: str, starter_code: str, max_new_tokens=256, temperature=0.4) -> str:
    """Generate code completion for a LeetCode problem."""
    full_prompt = (
        "Complete the following Python function to solve the problem:\n\n"
        f"Problem: {problem_prompt}\n\n"
        f"Starter code:\n{starter_code}\n\n"
        "Complete the function:"
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(coder_model.device)
    with torch.no_grad():
        base = coder_model.model
        
        # Handle temperature=0 as greedy decoding
        if temperature == 0.0:
            out = base.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=inputs["input_ids"].shape[1] + max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            out = base.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=inputs["input_ids"].shape[1] + max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    generated_part = text[len(full_prompt):].strip()
    
    # Combine starter code with generated completion
    if generated_part:
        # Try to extract function body if generation includes function definition
        code_block = extract_python_block(generated_part)
        if code_block.strip().startswith('def '):
            # If generated a complete function, use it
            return code_block
        else:
            # Otherwise, combine with starter code
            return starter_code + "\n" + code_block
    else:
        # If no generation, return starter code as-is
        return starter_code