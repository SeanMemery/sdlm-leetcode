#!/usr/bin/env python3
"""
LeetCode optimization example with a differentiable inner loop.

Pipeline:
  - Load LeetCodeDataset problems (Hugging Face)
  - Initialize code from starter_code
  - Inner-loop: differentiate through a critic (MomentumLossFunction) using relaxed one-hot samples from Variable
  - Periodically evaluate on local tests (from the dataset)
  - Report overall accuracy and save per-task logs and a summary

Requirements:
  - torch, transformers, datasets, pandas, tqdm
  - sdlm (STGSDiffModel, Variable)

Run:
  python examples/leetcode_example.py --model gpt2 --max_programs 5 --n_epoch 300 --batch_size 4 --lr 0.05
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from sdlm.leetcode.dataset import load_leetcode_dataset, build_evaluator
from sdlm.leetcode.utils import build_model, clean_for_submission
from sdlm.leetcode.momentum import MomentumLossFunction
from sdlm.textgrad.variables import Variable


def create_results_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"results/{ts}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_task_log(results_dir: Path, slug: str, task_logs):
    if not task_logs:
        return
    pd.DataFrame(task_logs).to_csv(results_dir / f"{slug}.csv", index=False)


def save_summary(results_dir: Path, summary_rows, args):
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
        print(f"Saved summary: {results_dir / 'summary.csv'}")
    settings = pd.DataFrame({
        "parameter": list(vars(args).keys()),
        "value": list(vars(args).values())
    })
    settings.to_csv(results_dir / "settings.csv", index=False)
    print(f"Saved settings: {results_dir / 'settings.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2") # meta-llama/Llama-3.2-1B-Instruct
    ap.add_argument("--max_programs", type=int, default=10)
    ap.add_argument("--n_epoch", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--t_test", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--difficulty", type=str, default="Easy", choices=["Easy", "Medium", "Hard"])
    ap.add_argument("--max_new_tokens", type=int, default=512, help="If set, pad/truncate code to this length")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = create_results_dir()
    print(f"Results will be saved to: {results_dir}")

    # STGS settings (kept minimal and stable)
    stgs_kwargs = dict(
        stgs_hard=False, hard=False,
        init_temperature=0.7, temperature=0.7,
        learnable_temperature=False,
        use_bpttoken=False, bpttoken=False,
        hidden_state_conditioning=False
    )

    # Build coder and critic (reuse to save VRAM)
    coder_model, tokenizer = build_model(args.model, device, stgs_kwargs)
    critic_model, _ = coder_model, tokenizer

    # Dataset and evaluator (local execution of provided testcases)
    dataset = load_leetcode_dataset(max_items=args.max_programs, split=args.split, difficulty=args.difficulty)
    evaluator = build_evaluator()

    # Short judge prompt
    momentum_question = (
        "#TASK_DESCRIPTION:\n {t_descr}\n\n"
        "#INPUT:\n {input}\n\n"
        "Does the above input satisfy the task description?"
    )

    summary_rows = []
    solved = 0

    for idx, problem in enumerate(tqdm(dataset, desc="Problems")):
        slug = problem["task_id"]
        prompt = problem["problem_description"]
        starter_code = problem["starter_code"] + " " * (args.max_new_tokens - len(problem["starter_code"]))

        print(f"\n{'='*60}")
        print(f"Task {idx+1}/{len(dataset)}: {slug}")
        print(f"{'='*60}")

        # 1) Initial code from starter code, clean and (optionally) pad to fixed window
        init_code = clean_for_submission(starter_code)
        print(f"Using starter code ({len(init_code)} chars)")

        # 2) Differentiable code variable
        C_var = Variable(
            tokenizer=tokenizer,
            initial_str=init_code,
            template="{VARIABLE}",
            use_fluency_constraint=False,
            temperature=0.7,
            hard=False,
            learnable_temperature=False,
            device=device,
        )

        # 3) Differentiable judge loss (training path uses one-hot; eval path uses strings)
        loss_fn = MomentumLossFunction(
            critic_dlm=critic_model,
            momentum_question=momentum_question,
            Momentum_variables={"t_descr": prompt},
            momentum_answer="Yes",
            use_cot=False,  # keep off for speed/stability
            answer_extractor="",
        )

        # 4) Optimizer
        params = list(C_var.parameters())
        assert sum(p.numel() for p in params) > 0, "C_var has no trainable params"
        optim = torch.optim.Adam(params, lr=args.lr)

        # Logs and best tracking
        task_logs = []
        best_code = init_code
        with torch.no_grad():
            best_eval_loss = loss_fn(batched_input=[init_code]).item()
        success = False

        task_logs.append({
            "epoch": 0,
            "train_loss": float("nan"),
            "eval_loss": best_eval_loss,
            "best_eval_loss": best_eval_loss,
            "code_length": len(init_code),
            "sampled_code": init_code,
            "is_best": True,
            "test_submitted": False,
            "test_accepted": False,
            "test_status": "Not submitted"
        })

        print(f"Starting training for {args.n_epoch} epochs...")
        for epoch in range(1, args.n_epoch + 1):
            optim.zero_grad()

            # Differentiable batch of relaxed one-hots
            batch_oh = []
            for _ in range(args.batch_size):
                _, code_one_hot, _ = C_var()
                batch_oh.append(code_one_hot)

            train_loss = loss_fn(batched_one_hot=batch_oh)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(C_var.parameters(), 1.0)
            optim.step()

            # Decode for evaluation/logging
            code_candidate = C_var.forward_sample(temperature=max(args.t_test, 0.3))
            code_candidate = clean_for_submission(code_candidate)
            with torch.no_grad():
                eval_loss = loss_fn(batched_input=[code_candidate]).item()

            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
                best_code = code_candidate

            # Log and test locally every log_every
            if epoch % args.log_every == 0 or epoch == args.n_epoch:
                print(f"Epoch {epoch}/{args.n_epoch}: train_loss={train_loss.item():.4f}, eval_loss={eval_loss:.4f}, best={best_eval_loss:.4f}")
                print(f"Sampled code ({len(code_candidate)} chars):\n{code_candidate}\n")
                test_accepted, test_status = evaluator.evaluate(best_code.strip(), problem)
                print(f"Local test: {test_status}")

                task_logs.append({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "eval_loss": eval_loss,
                    "best_eval_loss": best_eval_loss,
                    "code_length": len(code_candidate),
                    "sampled_code": code_candidate,
                    "is_best": is_best,
                    "test_submitted": True,
                    "test_accepted": test_accepted,
                    "test_status": test_status
                })

                save_task_log(results_dir, slug, task_logs)

                if test_accepted:
                    success = True
                    print(f"✓ Solved at epoch {epoch}")
                    break

        # Summary row
        summary_rows.append({
            "idx": idx + 1,
            "slug": slug,
            "final_best_loss": best_eval_loss,
            "epochs_trained": epoch,
            "solved": success,
            "best_code_length": len(best_code),
            "initial_code_length": len(init_code),
        })
        solved += int(success)

    # Save summary and accuracy
    save_summary(results_dir, summary_rows, args)
    total = len(dataset)
    acc = solved / total if total else 0.0
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {solved}/{total} = {acc*100:.1f}%")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
