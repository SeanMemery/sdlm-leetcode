#!/usr/bin/env python3
"""
SDLM LeetCode Optimization Example (Momentum-based, Differentiable)

This script demonstrates:
    1. Initial code generation from a coder model
    2. Differentiable inner-loop optimization using MomentumLossFunction against a critic DLM
    3. Periodic outer-loop evaluation via LeetCode (if available)
    4. Per-problem result logging to CSV and summary reporting

Requirements:
    - torch, transformers, pandas, tqdm
    - sdlm (with STGSDiffModel, Variable)
    - Optional: leetcode-hard-gym (leetcode_env) and environment variables:
        export LEETCODE_SESSION=...
        export LEETCODE_CSRF_TOKEN=...

Usage Example:
    python leetcode_example.py --model gpt2 --max_programs 3 --n_epoch 30 --batch_size 8 --lr 1e-2 --t_test 0.05 --success_threshold 0.1 --eval_every 5

You can also pass --dataset ./my_dataset.jsonl with lines:
    {"task_id":"two-sum","prompt":"<problem text and signature>","tests":""}
"""

import os
import json
import argparse
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, AutoConfig

# SDLM utils and momentum loss
from sdlm.leetcode.utils import (
    extract_python_block,
    clean_leetcode_signature,
    build_coder_and_judge,
    generate_initial_code,
)
from sdlm.textgrad.variables import Variable
from sdlm.stgs_diff_model import STGSDiffModel
from sdlm.leetcode.momentum import MomentumLossFunction


def load_dataset_or_defaults(max_programs: int, dataset_path: str = None):
    """
    Load dataset from JSONL if provided, otherwise return a small set of built-in LeetCode-like problems.
    Returns: list of tuples (slug, problem_prompt, tests_feedback)
    """
    if dataset_path and os.path.exists(dataset_path):
        items = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                items.append((obj["task_id"], obj["prompt"], obj.get("tests", "")))
                if len(items) >= max_programs:
                    break
        return items

    # Default problems (lightweight descriptions with signature stubs; no fenced code blocks)
    problem_descriptions = {
        "two-sum": (
            "Given an array of integers nums and an integer target, return indices of the two numbers such "
            "that they add up to target. Assume exactly one solution exists and you may not use the same "
            "element twice. Return the answer in any order.\n\n"
            "Example:\n"
            "Input: nums = [2,7,11,15], target = 9\n"
            "Output: [0,1]\n\n"
            "def twoSum(nums, target):"
        ),
        "add-two-numbers": (
            "You are given two non-empty linked lists representing two non-negative integers. "
            "The digits are stored in reverse order, and each node contains a single digit. "
            "Add the two numbers and return the sum as a linked list.\n\n"
            "def addTwoNumbers(l1, l2):"
        ),
        "longest-substring-without-repeating-characters": (
            "Given a string s, find the length of the longest substring without repeating characters.\n\n"
            "Example:\n"
            "Input: s = \"abcabcbb\"\n"
            "Output: 3\n\n"
            "def lengthOfLongestSubstring(s):"
        ),
        "median-of-two-sorted-arrays": (
            "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.\n\n"
            "def findMedianSortedArrays(nums1, nums2):"
        ),
        "longest-palindromic-substring": (
            "Given a string s, return the longest palindromic substring in s.\n\n"
            "Example:\n"
            "Input: s = \"babad\"\n"
            "Output: \"bab\" (or \"aba\")\n\n"
            "def longestPalindrome(s):"
        ),
    }
    slugs = list(problem_descriptions.keys())[:max_programs]
    return [(slug, problem_descriptions[slug], "") for slug in slugs]


def build_critic(model_name: str, device: str, stgs_kwargs: dict):
    """
    Build a critic model for differentiable evaluation.
    Returns: STGSDiffModel instance
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Force eager attention at both config and load-time
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.attn_implementation = "eager"

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        attn_implementation="eager",
        torch_dtype=(torch.bfloat16 if device != "cpu" else torch.float32),
    ).to(device)

    critic = STGSDiffModel(model=base, tokenizer=tok, stgs_kwargs=stgs_kwargs, device=device)
    return critic


def main():
    """
    Main entry point for SDLM LeetCode optimization.
    Handles argument parsing, model setup, optimization loop, and result logging.
    """
    # ----------------------
    # Argument Parsing
    # ----------------------
    parser = argparse.ArgumentParser(description="SDLM LeetCode Momentum Optimization (Differentiable)")
    # Data
    parser.add_argument("--dataset", type=str, default=None, help="Optional JSONL dataset. If not set, uses built-in problems.")
    parser.add_argument("--max_programs", type=int, default=5, help="Max number of problems to optimize")
    # Models
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name for coder and critic (use a code model for better results)")
    # Optimization
    parser.add_argument("--use_fluency_constraint", action="store_true", default=True, help="Use fluency constraint in Variable")
    parser.add_argument("--use_cot", action="store_true", default=True, help="Use Chain-of-Thought in momentum loss prompt")
    parser.add_argument("--n_epoch", type=int, default=30, help="Number of inner-loop epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of differentiable samples")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--t_test", type=float, default=0.05, help="Low temperature for decoding a test sample")
    parser.add_argument("--success_threshold", type=float, default=0.1, help="Early-stop threshold on critic loss")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate on LeetCode every K epochs")
    # STGS knobs
    parser.add_argument("--stgs_hard", action="store_true", default=False, help="Use hard STGS during training (recommend False)")
    parser.add_argument("--learnable_temperature", action="store_true", default=False, help="Learn critic STGS temperature")
    parser.add_argument("--init_temperature", type=float, default=0.7, help="Initial temperature for STGS")
    parser.add_argument("--bpttoken", action="store_true", default=False, help="Use BPT token (if supported)")
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory to store results")
    args = parser.parse_args()

    # ----------------------
    # Setup & Output Directory
    # ----------------------
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("SDLM LeetCode Optimization (Momentum, Differentiable)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Seed:   {args.seed}")
    print("-" * 80)
    print("Inner-loop settings:")
    print(f"  n_epoch={args.n_epoch}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  train init_temperature={args.init_temperature}, t_test={args.t_test}")
    print(f"  STGS: hard={args.stgs_hard}, learnable_temperature={args.learnable_temperature}, bpttoken={args.bpttoken}")
    print(f"  use_fluency_constraint={args.use_fluency_constraint}, use_cot={args.use_cot}")
    print("-" * 80)
    print(f"Results will be saved to: {results_dir}")
    print()

    # ----------------------
    # Data Loading
    # ----------------------
    dataset = load_dataset_or_defaults(args.max_programs, args.dataset)
    print(f"Loaded {len(dataset)} problems")

    # ----------------------
    # Model Setup
    # ----------------------
    coder_model, tokenizer = build_coder_and_judge(args.model, device, args)

    # Optional: LeetCode evaluator
    evaluator = None
    try:
        from sdlm.leetcode.evaluators.leetcode_eval import LeetCodeEvaluator  # lazy import
        evaluator = LeetCodeEvaluator()
        print("LeetCode evaluator initialized.")
    except Exception as e:
        print(f"LeetCode evaluator not available: {e}")
        print("Continuing without external submission; only critic loss will be logged.")

    # ----------------------
    # Optimization Loop
    # ----------------------
    momentum_question = (
        "#TASK_DESCRIPTION:\n {t_descr}\n\n"
        "#INPUT:\n {input}\n\n"
        "Does the above input (#INPUT) fulfill the above task description (#TASK_DESCRIPTION)?"
    )
    answer_extractor = ("" if not args.use_cot else "Based on the above reasoning, the answer ($answer) to the above questions is $")

    stgs_kwargs = {
        "stgs_hard": args.stgs_hard, "hard": args.stgs_hard,
        "learnable_temperature": args.learnable_temperature,
        "init_temperature": args.init_temperature, "temperature": args.init_temperature,
        "bpttoken": args.bpttoken, "use_bpttoken": args.bpttoken,
        "hidden_state_conditioning": False,
    }

    summary_results = []

    for task_id, problem_prompt, tests in tqdm(dataset, desc="Processing problems"):
        summary_result = run_task_optimization(
            task_id=task_id,
            problem_prompt=problem_prompt,
            tests=tests,
            coder_model=coder_model,
            tokenizer=tokenizer,
            args=args,
            device=device,
            stgs_kwargs=stgs_kwargs,
            momentum_question=momentum_question,
            answer_extractor=answer_extractor,
            results_dir=results_dir,
            evaluator=evaluator
        )
        summary_results.append(summary_result)
def run_task_optimization(task_id, problem_prompt, tests, coder_model, tokenizer, args, device, stgs_kwargs, momentum_question, answer_extractor, results_dir, evaluator):
    """
    Run optimization and evaluation for a single LeetCode task.
    Returns: summary_result dict for this task.
    """
    # Step 1: Generate initial code
    initial_code = generate_initial_code(coder_model, tokenizer, problem_prompt, max_new_tokens=512)
    initial_code_clean = clean_leetcode_signature(initial_code)

    # Step 2: Build optimizable Variable (differentiable code window)
    C_var = Variable(
        tokenizer=tokenizer,
        initial_str=initial_code,
        template='\n"""python\n{VARIABLE}\n"""\n',
        use_fluency_constraint=args.use_fluency_constraint,
        temperature=args.init_temperature,
        hard=args.stgs_hard,
        learnable_temperature=args.learnable_temperature,
        device=str(device),
    )

    # Step 3: Build critic DLM
    DLM_critic = build_critic(args.model, device, stgs_kwargs)

    # Step 4: Build differentiable momentum loss
    Task_loss = MomentumLossFunction(
        critic_dlm=DLM_critic,
        momentum_question=momentum_question,
        Momentum_variables={"t_descr": problem_prompt},
        momentum_answer="Yes",
        use_cot=args.use_cot,
        answer_extractor=answer_extractor,
    )

    # Debug: confirm critic sees the code tokens
    pre, post = Task_loss._format_question_parts()
    pre_len = len(Task_loss.tokenizer(pre).input_ids)
    post_len = len(Task_loss.tokenizer(post).input_ids)
    print(f"[DEBUG] pre_len={pre_len}, post_len={post_len}")

    # Step 5: Setup optimizer
    params = list(C_var.parameters())
    num_params = sum(p.numel() for p in C_var.parameters())
    print(f"[DEBUG] C_var trainable params: {num_params}")
    assert num_params > 0, "C_var.parameters() is empty — optimizer cannot update anything!"

    # Optionally include STGS internals
    if hasattr(C_var, "stgs_parameters"):
        stgs_params = list(C_var.stgs_parameters())
        stgs_num = sum(p.numel() for p in stgs_params)
        print(f"[DEBUG] C_var.stgs_parameters(): {stgs_num}")
        params += stgs_params

    print("[DEBUG] C_var params:")
    for n, p in C_var.named_parameters():
        print(f"  {n}: {p.shape}, {p.dtype}, requires_grad={p.requires_grad}")

    if hasattr(C_var, 'train'):
        C_var.train()

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Step 6: Evaluate zero-shot (optional external LeetCode)
    task_results = []
    if evaluator is not None:
        try:
            success, total_correct, total_tests, runtime = evaluator.check_if_in_cache_or_submit(task_id, initial_code_clean)
        except Exception as e:
            print(f"Initial submission error for {task_id}: {e}")
            success, total_correct, total_tests, runtime = False, 0, 0, -1
    else:
        success, total_correct, total_tests, runtime = False, 0, 0, -1

    task_results.append({
        "epoch": 0,
        "code": initial_code_clean,
        "success": success,
        "test_cases_passed": total_correct,
        "test_cases_total": total_tests,
        "runtime": runtime,
        "loss": "N/A",
        "is_initial": True,
    })
    print(f"[{task_id}] Initial: success={success}, {total_correct}/{total_tests}, runtime={runtime}")

    # Training loop
    best_test_loss = float("inf")
    best_code = initial_code_clean
    success_achieved = False
    prev_params = [p.detach().clone() for p in C_var.parameters()]

    for epoch in range(1, args.n_epoch + 1):
        optimizer.zero_grad()
        batched_oh = []
        tracked_oh = None
        for _ in range(args.batch_size):
            _, code_one_hot, _ = C_var()
            assert code_one_hot.requires_grad, "code_one_hot is not differentiable — no gradient path."
            if epoch == 1:
                print(f"[DEBUG] code_one_hot grad_fn: {code_one_hot.grad_fn}")
            if tracked_oh is None:
                tracked_oh = code_one_hot
                tracked_oh.retain_grad()
            batched_oh.append(code_one_hot)

        loss = Task_loss(batched_one_hot=batched_oh)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(C_var.parameters(), 1.0)
        oh_grad = float(tracked_oh.grad.abs().sum().item()) if tracked_oh.grad is not None else 0.0
        var_grad = 0.0
        for p in C_var.parameters():
            if p.grad is not None:
                var_grad += float(p.grad.abs().sum().item())
        print(f"[DEBUG] epoch={epoch} train_loss={float(loss.item()):.4f} code_oh_grad_sum={oh_grad:.3e} C_var_grad_sum={var_grad:.3e}")
        optimizer.step()
        if hasattr(C_var, 'post_step'):
            C_var.post_step()
        elif hasattr(C_var, 'sync_parameters'):
            C_var.sync_parameters()
        elif hasattr(C_var, 'normalize'):
            C_var.normalize()
        total_delta = 0.0
        new_params = []
        for p_old, p_new in zip(prev_params, C_var.parameters()):
            delta = (p_new.detach() - p_old).norm().item()
            total_delta += delta
            new_params.append(p_new.detach().clone())
        print(f"[DEBUG] param L2 Δ sum: {total_delta:.6e}")
        prev_params = new_params
        debug_t = max(args.t_test, 0.4)
        test_code = C_var.forward_sample(temperature=debug_t) if hasattr(C_var, "forward_sample") else C_var.get_string()
        if not test_code.strip():
            test_code = "pass"
        clean_code = clean_leetcode_signature(test_code)
        with torch.no_grad():
            test_loss = Task_loss(batched_input=[clean_code]).item()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_code = clean_code
        do_eval = (epoch % args.eval_every == 0) or (test_loss <= args.success_threshold)
        if evaluator is not None and do_eval:
            try:
                temp_success, temp_correct, temp_total, temp_runtime = evaluator.check_if_in_cache_or_submit(task_id, clean_code)
            except Exception as e:
                print(f"[{task_id}] Submission error at epoch {epoch}: {e}")
                temp_success, temp_correct, temp_total, temp_runtime = False, 0, 0, -1
            task_results.append({
                "epoch": epoch,
                "code": clean_code,
                "success": temp_success,
                "test_cases_passed": temp_correct,
                "test_cases_total": temp_total,
                "runtime": temp_runtime,
                "loss": float(test_loss),
                "is_initial": False,
            })
            print(f"[{task_id}] Epoch {epoch}: eval_loss={test_loss:.4f}, success={temp_success}, {temp_correct}/{temp_total}")
            if temp_success:
                success_achieved = True
                best_code = clean_code
                print(f"[{task_id}] Found working solution at epoch {epoch}!")
                break
        if test_loss <= args.success_threshold and evaluator is None:
            print(f"[{task_id}] Early stop by critic threshold at epoch {epoch} (loss={test_loss:.4f})")
            break

    # Save per-task CSV
    task_df = pd.DataFrame(task_results)
    per_task_csv = os.path.join(results_dir, f"{task_id}.csv")
    task_df.to_csv(per_task_csv, index=False)
    print(f"[{task_id}] Saved {len(task_results)} records to {per_task_csv}")

    # Final external eval if not done or not successful yet
    final_success = None
    final_correct = None
    final_total = None
    final_runtime = None
    if evaluator is not None and not success_achieved:
        try:
            final_success, final_correct, final_total, final_runtime = evaluator.check_if_in_cache_or_submit(task_id, best_code)
        except Exception as e:
            print(f"[{task_id}] Final submission error: {e}")
            final_success, final_correct, final_total, final_runtime = False, 0, 0, -1

    return {
        "task_id": task_id,
        "seed": args.seed,
        "total_epochs": epoch,
        "success_achieved": bool(success_achieved or (final_success is True)),
        "best_loss": float(best_test_loss),
        "final_code_length": len(best_code) if best_code else 0,
        "epochs_logged": len(task_results) - 1,  # excluding initial
        "final_success": final_success if final_success is not None else (success if success else False),
        "final_passed": final_correct if final_correct is not None else (total_correct if total_correct else 0),
        "final_total": final_total if final_total is not None else (total_tests if total_tests else 0),
        "final_runtime": final_runtime if final_runtime is not None else (runtime if runtime is not None else -1),
    }

    # Save summary and settings
    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(results_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    settings = {
        "timestamp": timestamp,
        "model": args.model,
        "max_programs": args.max_programs,
        "n_epoch": args.n_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "success_threshold": args.success_threshold,
        "t_test": args.t_test,
        "use_fluency_constraint": args.use_fluency_constraint,
        "use_cot": args.use_cot,
        "stgs_hard": args.stgs_hard,
        "learnable_temperature": args.learnable_temperature,
        "init_temperature": args.init_temperature,
        "bpttoken": args.bpttoken,
        "seed": args.seed,
        "dataset": args.dataset,
    }
    pd.DataFrame([settings]).to_csv(os.path.join(results_dir, "settings.csv"), index=False)

    # Print final summary
    solved = sum(1 for r in summary_results if r["success_achieved"])
    total_attempted = len(summary_results)
    epochs_logged_total = sum(r["epochs_logged"] for r in summary_results)
    avg_epochs_logged = (epochs_logged_total / total_attempted) if total_attempted > 0 else 0.0

    files_list = ", ".join([f"{r['task_id']}.csv" for r in summary_results])
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Problems attempted: {total_attempted}")
    if total_attempted:
        print(f"Problems solved:    {solved} ({(solved/total_attempted*100.0):.1f}%)")
    else:
        print("Problems solved: N/A")
    print(f"Total epochs logged: {epochs_logged_total}")
    print(f"Average epochs per problem: {avg_epochs_logged:.1f}")
    print(f"Per-task CSVs: {files_list}")
    print(f"Summary: {summary_csv}")
    print(f"Settings: {os.path.join(results_dir, 'settings.csv')}")


if __name__ == "__main__":
    main()
