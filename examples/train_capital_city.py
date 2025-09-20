import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from sdlm.leetcode.utils import build_model, clean_for_submission
from sdlm.leetcode.momentum import MomentumLossFunction
from sdlm.textgrad.variables import Variable


def evaluate_answer(optimized_text: str, target_city: str = "Dublin") -> tuple[bool, str]:
    """
    Simple evaluator that checks if the optimized text contains the target city.
    
    Args:
        optimized_text: The optimized prompt text
        target_city: The correct capital city
        
    Returns:
        (is_correct, status_message)
    """
    # Clean and normalize for comparison
    text_clean = optimized_text.strip().lower()
    target_clean = target_city.lower()
    
    if target_clean in text_clean:
        return True, f"Correct: Found '{target_city}' in optimized text"
    else:
        return False, f"Wrong: '{target_city}' not found in optimized text"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--n_epoch", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--t_test", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=100, help="Log detailed info every N epochs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # STGS knobs (kept minimal)
    stgs_kwargs = dict(stgs_hard=False, hard=False, init_temperature=0.7, temperature=0.7,
                       learnable_temperature=False, use_bpttoken=False, bpttoken=False,
                       hidden_state_conditioning=False)

    # Build coder (generation) and critic (judge). To save vRAM, reuse same model.
    coder_model, tokenizer = build_model(args.model, device, stgs_kwargs)
    critic_model, _ = coder_model, tokenizer  # reuse

    # The question we want to optimize
    question = "What is the capital city of Ireland?"
    target_answer = "Dublin"
    
    print(f"\n{'='*60}")
    print(f"Optimizing answer to: {question}")
    print(f"Target answer: {target_answer}")
    print(f"{'='*60}")
    
    # 1) Use initial guess as starting point
    init_text = "The answer is London"  # Start with a wrong answer
    print(f"Using initial text: '{init_text}' ({len(init_text)} chars)")

    # 2) Differentiable variable over the text
    C_var = Variable(
        tokenizer=tokenizer,
        initial_str=init_text,
        template="{VARIABLE}",
        use_fluency_constraint=True,
        temperature=0.7,
        hard=False,
        learnable_temperature=stgs_kwargs.get('learnable_temperature', False),
        device=device,
    )

    # 3) Loss - momentum question asking if the optimized text contains the right answer
    use_cot = False
    momentum_question = (
        "#QUESTION:\n {question}\n\n"
        "#PROPOSED_ANSWER:\n {input}\n\n"
        "Does the proposed answer correctly answer the question?"
    )
    
    loss_fn = MomentumLossFunction(
        critic_dlm=critic_model,
        momentum_question=momentum_question,
        Momentum_variables={"question": question},
        momentum_answer="Yes",
        use_cot=use_cot,
        answer_extractor=f"{'' if not use_cot else 'Based on the above reasoning, the answer is '}",
    )

    # 4) Optimizer
    parameters = [C_var.parameters()]
    if stgs_kwargs.get('learnable_temperature', False):
        parameters.append(critic_model.stgs_parameters())
    params = []
    for param_group in parameters:
        params.extend(param_group)
    assert sum(p.numel() for p in params) > 0, "No trainable parameters"
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Initialize variables
    C_opt = None
    best_test_loss = 1e6
    
    # Log initial state
    with torch.no_grad():
        initial_loss = loss_fn(batched_input=[init_text]).item()
    
    print(f"Epoch 0: Initial loss={initial_loss:.4f}, text='{init_text.strip()}'")

    # 5) Train - differentiable inner loop
    print(f"Starting optimization for {args.n_epoch} epochs...")
    for epoch in range(1, args.n_epoch + 1):
        optimizer.zero_grad()

        # Build a batch of differentiable code samples (relaxed one-hot)
        batch_oh = []
        tracked_oh = None
        for _ in range(args.batch_size):
            _, code_one_hot, _ = C_var()  # differentiable one-hot
            assert code_one_hot.requires_grad, "code_one_hot must be differentiable"
            if tracked_oh is None:
                tracked_oh = code_one_hot
                tracked_oh.retain_grad()
            batch_oh.append(code_one_hot)

        # Differentiable loss and backward
        train_loss = loss_fn(batched_one_hot=batch_oh)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(C_var.parameters(), 1.0)
        optimizer.step()

        # Debug: verify gradients reached the variable
        oh_grad_sum = float(tracked_oh.grad.abs().sum().item()) if tracked_oh.grad is not None else 0.0
        var_grad_sum = sum((p.grad.abs().sum().item() for p in C_var.parameters() if p.grad is not None), 0.0)
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[DEBUG] epoch={epoch} train_loss={train_loss.item():.4f} "
                  f"code_oh_grad_sum={oh_grad_sum:.3e} C_var_grad_sum={var_grad_sum:.3e}")

        # Decode for evaluation/logging at a moderate temp to reveal changes
        test_text = C_var.forward_sample(temperature=max(args.t_test, 0.1))
        with torch.no_grad():
            test_loss = loss_fn(batched_input=[test_text]).item()

        # Track best by eval loss
        if test_loss < best_test_loss:
            C_opt = test_text
            best_test_loss = test_loss

        # Periodic logging
        if epoch % args.log_every == 0 or epoch == args.n_epoch:
            print(f"Epoch {epoch}: train_loss={train_loss.item():.4f}, "
                  f"test_loss={test_loss:.4f}, best_loss={best_test_loss:.4f}")
            print(f"  Current text: '{test_text.strip()}'")
            print(f"  Best text:    '{(C_opt or '').strip()}'")

            # Optional: literal correctness check (not differentiable)
            is_correct, eval_status = evaluate_answer(C_opt or test_text, target_answer)
            print(f"  Evaluation: {eval_status}")
            if is_correct:
                print(f"✓ Correct answer found at epoch {epoch}!")
                break

    # Return C_opt (as specified in the algorithm)
    print(f"{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Target answer: {target_answer}")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {best_test_loss:.4f}")
    print(f"Final optimized text: '{C_opt.strip() if C_opt else 'None'}'")
    print(f"Epochs trained: {epoch}")
    
    return C_opt

if __name__ == "__main__":
    main()