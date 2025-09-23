#!/usr/bin/env python3
"""
Full pipeline experiment: Optimization + Multi-Sample Testing + Evaluation.

This script implements the complete pipeline:
1. Load N random Easy LeetCode questions
2. Optimize code for each using SDLM framework
3. Sample K code snippets from optimized variable (temperature=0.3)
4. Test each sample on LeetCode problems
5. Save success rates and metrics to wandb

Pipeline:
  Raw code -> SDLM Optimization -> Sample K codes -> Test each -> Best result
"""

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import wandb
import pandas as pd
from tqdm import tqdm

from sdlm.leetcode.dataset import load_leetcode_dataset, build_evaluator
from sdlm.leetcode.utils import build_model, clean_for_submission, rewrite_to_valid_python_cot
from sdlm.leetcode.momentum import LeetCodeMomentumLoss, PythonSyntaxMomentumLoss
from sdlm.textgrad.variables import Variable


def create_results_dir(base_name: str = "full_pipeline") -> Path:
    """Create timestamped results directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/{base_name}_{ts}")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


class FullPipelineExperiment:
    """Full pipeline experiment runner."""
    
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create artifacts directory structure
        self.results_dir = Path(config.results_dir)
        self.artifacts_dir = self.results_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if not config.disable_wandb:
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                tags=["full_pipeline", "optimization", "reinterpretation", "evaluation"],
                notes="Complete pipeline: optimization -> LM reinterpretation -> evaluation"
            )
        
        # Load model and tokenizer (reuse for all phases)
        print(f"Loading model: {config.model}")
        self.model, self.tokenizer = build_model(config.model, self.device, {
            "stgs_hard": config.hard_mode,
            "hard": config.hard_mode,
            "init_temperature": config.temperature,
            "temperature": config.temperature,
            "learnable_temperature": False,
            "use_bpttoken": False,
            "bpttoken": False,
            "hidden_state_conditioning": False
        })
        
        # Load dataset and evaluator
        self.dataset = load_leetcode_dataset(
            max_items=config.n_problems, 
            split=config.split, 
            difficulty=config.difficulty
        )
        self.evaluator = build_evaluator()
        
        # Create loss functions
        self.leetcode_loss = LeetCodeMomentumLoss(
            critic_dlm=self.model,
            use_cot=config.use_cot
        )
        self.syntax_loss = PythonSyntaxMomentumLoss(
            critic_dlm=self.model,
            use_cot=config.use_cot
        )
        
        print(f"Initialized pipeline for {len(self.dataset)} problems")
    
    def optimize_code(self, problem: Dict, problem_idx: int) -> Tuple[Variable, Dict]:
        """
        Phase 1: Optimize code using SDLM framework.
        
        Returns:
            Tuple of (optimized_variable, optimization_metrics)
        """
        slug = problem["task_id"]
        prompt = problem["problem_description"]
        starter_code = problem["starter_code"]
        
        print(f"\n🔧 Phase 1: Optimizing code for {slug}")
        
        # Clean and prepare initial code
        init_code = clean_for_submission(starter_code)
        if len(init_code.strip()) == 0:
            init_code = f"# TODO: Implement solution for {slug}\npass"
        
        # Create differentiable variable
        code_var = Variable(
            tokenizer=self.tokenizer,
            initial_str=init_code,
            template="{VARIABLE}",
            use_fluency_constraint=False,
            temperature=self.config.temperature,
            hard=self.config.hard_mode,
            learnable_temperature=False,
            device=self.device,
        )
        
        # Configure PoE if enabled
        if self.config.poe_gamma > 0:
            code_var.configure_poe(
                lm=self.model,
                tokenizer=self.tokenizer,
                gamma=self.config.poe_gamma,
                prefix=prompt
            )
        
        # Setup optimizer
        params = list(code_var.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.lr)
        
        # Track optimization metrics
        optimization_metrics = {
            "initial_code": init_code,
            "optimization_steps": [],
            "best_code": init_code,
            "best_total_loss": float('inf'),
            "final_leetcode_loss": float('nan'),
            "final_syntax_loss": float('nan'),
        }
        
        print(f"Starting optimization for {self.config.n_steps} steps...")
        
        for step in range(1, self.config.n_steps + 1):
            optimizer.zero_grad()
            
            # Generate batch of relaxed samples for training
            batch_oh = []
            for _ in range(self.config.batch_size):
                _, code_one_hot, _ = code_var()
                batch_oh.append(code_one_hot)
            
            # Compute losses on one-hot samples
            leetcode_loss = self.leetcode_loss(
                batched_one_hot=batch_oh,
                Momentum_variables={"t_descr": prompt}
            )
            syntax_loss = self.syntax_loss(batched_one_hot=batch_oh)
            
            # Combine losses
            total_loss = (self.config.leetcode_weight * leetcode_loss + 
                         self.config.syntax_weight * syntax_loss)
            
            # Backprop and step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(code_var.parameters(), 1.0)
            optimizer.step()
            
            # Evaluation: sample code and compute eval losses
            sampled_code = code_var.forward_sample(temperature=0.3)
            sampled_code = clean_for_submission(sampled_code)
            
            with torch.no_grad():
                eval_leetcode_loss = self.leetcode_loss(
                    batched_input=[sampled_code],
                    Momentum_variables={"t_descr": prompt}
                ).item()
                eval_syntax_loss = self.syntax_loss(batched_input=[sampled_code]).item()
                eval_total_loss = (self.config.leetcode_weight * eval_leetcode_loss + 
                                 self.config.syntax_weight * eval_syntax_loss)
            
            # Track best code
            if eval_total_loss < optimization_metrics["best_total_loss"]:
                optimization_metrics["best_total_loss"] = eval_total_loss
                optimization_metrics["best_code"] = sampled_code
                optimization_metrics["final_leetcode_loss"] = eval_leetcode_loss
                optimization_metrics["final_syntax_loss"] = eval_syntax_loss
            
            # Log step metrics
            step_metrics = {
                "step": step,
                "train_total_loss": total_loss.item(),
                "train_leetcode_loss": leetcode_loss.item(),
                "train_syntax_loss": syntax_loss.item(),
                "eval_total_loss": eval_total_loss,
                "eval_leetcode_loss": eval_leetcode_loss,
                "eval_syntax_loss": eval_syntax_loss,
                "code_length": len(sampled_code),
                "sampled_code": sampled_code
            }
            optimization_metrics["optimization_steps"].append(step_metrics)
            
            # Print progress
            if step % self.config.log_every == 0 or step == self.config.n_steps:
                print(f"  Step {step}/{self.config.n_steps}: "
                      f"total_loss={eval_total_loss:.4f} "
                      f"(leetcode={eval_leetcode_loss:.4f}, syntax={eval_syntax_loss:.4f})")
        
        # Save optimization artifacts (will be saved later in sampling phase)
        optimization_metrics["optimization_completed"] = True
        
        print(f"✅ Optimization complete. Best total loss: {optimization_metrics['best_total_loss']:.4f}")
        return code_var, optimization_metrics
    
    def sample_and_test_codes(self, code_var: Variable, problem: Dict, problem_idx: int) -> Tuple[str, Dict]:
        """
        Phase 2: Sample K code snippets from optimized variable and test each.
        
        Returns:
            Tuple of (best_code, sampling_metrics)
        """
        slug = problem["task_id"]
        
        print(f"🎯 Phase 2: Sampling {self.config.k_samples} code snippets for {slug}")
        
        # Create problem-specific artifacts directory
        problem_artifacts_dir = self.artifacts_dir / f"problem_{problem_idx:03d}_{slug}"
        problem_artifacts_dir.mkdir(exist_ok=True)
        
        sampling_metrics = {
            "k_samples": self.config.k_samples,
            "sampled_codes": [],
            "test_results": [],
            "best_code": "",
            "best_code_idx": -1,
            "any_success": False,
            "success_count": 0,
            "success_rate": 0.0,
            "artifacts_dir": str(problem_artifacts_dir)
        }
        
        # Sample K code snippets using temperature 0.3
        for i in range(self.config.k_samples):
            try:
                # Sample from the optimized variable
                sampled_code = code_var.forward_sample(temperature=0.3)
                sampled_code = clean_for_submission(sampled_code)
                
                # Save sample artifact
                sample_file = problem_artifacts_dir / f"sample_{i:03d}.py"
                with open(sample_file, 'w') as f:
                    f.write(f"# Sample {i} for problem {slug}\n")
                    f.write(f"# Generated from optimized variable with temperature=0.3\n")
                    f.write(f"# Problem: {problem['problem_description'][:100]}...\n\n")
                    f.write(sampled_code)
                
                # Test this sample
                try:
                    test_success, test_status = self.evaluator.evaluate(sampled_code.strip(), problem)
                    
                    sample_result = {
                        "sample_idx": i,
                        "code": sampled_code,
                        "code_length": len(sampled_code),
                        "test_success": test_success,
                        "test_status": test_status,
                        "test_error": None,
                        "sample_file": str(sample_file)
                    }
                    
                    if test_success:
                        sampling_metrics["success_count"] += 1
                        sampling_metrics["any_success"] = True
                        if sampling_metrics["best_code_idx"] == -1:  # First success
                            sampling_metrics["best_code"] = sampled_code
                            sampling_metrics["best_code_idx"] = i
                        print(f"  ✅ Sample {i+1}/{self.config.k_samples}: SUCCESS")
                        
                        # Mark successful sample with special suffix
                        success_file = problem_artifacts_dir / f"sample_{i:03d}_SUCCESS.py"
                        sample_file.rename(success_file)
                        sample_result["sample_file"] = str(success_file)
                    else:
                        print(f"  ❌ Sample {i+1}/{self.config.k_samples}: {test_status}")
                        
                except Exception as e:
                    sample_result = {
                        "sample_idx": i,
                        "code": sampled_code,
                        "code_length": len(sampled_code),
                        "test_success": False,
                        "test_status": f"Evaluation error: {e}",
                        "test_error": str(e),
                        "sample_file": str(sample_file)
                    }
                    print(f"  💥 Sample {i+1}/{self.config.k_samples}: Evaluation error: {e}")
                
                # Save detailed result metadata
                result_file = problem_artifacts_dir / f"sample_{i:03d}_result.json"
                with open(result_file, 'w') as f:
                    import json
                    json.dump(sample_result, f, indent=2)
                
                sampling_metrics["sampled_codes"].append(sampled_code)
                sampling_metrics["test_results"].append(sample_result)
                
            except Exception as e:
                print(f"  💥 Sample {i+1}/{self.config.k_samples}: Sampling error: {e}")
                
                # Save error artifact
                error_file = problem_artifacts_dir / f"sample_{i:03d}_ERROR.txt"
                with open(error_file, 'w') as f:
                    f.write(f"# Sample {i} for problem {slug}\n")
                    f.write(f"# Sampling failed with error\n\n")
                    f.write(f"Error: {str(e)}\n")
                
                sample_result = {
                    "sample_idx": i,
                    "code": "",
                    "code_length": 0,
                    "test_success": False,
                    "test_status": f"Sampling error: {e}",
                    "test_error": str(e),
                    "sample_file": str(error_file)
                }
                
                # Save error result metadata
                result_file = problem_artifacts_dir / f"sample_{i:03d}_result.json"
                with open(result_file, 'w') as f:
                    import json
                    json.dump(sample_result, f, indent=2)
                
                sampling_metrics["sampled_codes"].append("")
                sampling_metrics["test_results"].append(sample_result)
        
        # Calculate success rate
        sampling_metrics["success_rate"] = (sampling_metrics["success_count"] / 
                                          self.config.k_samples if self.config.k_samples > 0 else 0.0)
        
        # If no success, use the first valid sample as best
        if not sampling_metrics["any_success"] and sampling_metrics["sampled_codes"]:
            valid_samples = [code for code in sampling_metrics["sampled_codes"] if code.strip()]
            if valid_samples:
                sampling_metrics["best_code"] = valid_samples[0]
                sampling_metrics["best_code_idx"] = 0
        
        # Save comprehensive problem summary
        problem_summary = {
            "problem_idx": problem_idx,
            "task_id": slug,
            "problem_description": problem["problem_description"],
            "starter_code": problem["starter_code"],
            "k_samples": self.config.k_samples,
            "success_count": sampling_metrics["success_count"],
            "success_rate": sampling_metrics["success_rate"],
            "any_success": sampling_metrics["any_success"],
            "best_code_idx": sampling_metrics["best_code_idx"],
            "best_code": sampling_metrics["best_code"],
            "all_sample_results": sampling_metrics["test_results"]
        }
        
        summary_file = problem_artifacts_dir / "problem_summary.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(problem_summary, f, indent=2)
        
        # Save the best/successful code prominently if any succeeded
        if sampling_metrics["any_success"]:
            best_code_file = problem_artifacts_dir / "BEST_SOLUTION.py"
            with open(best_code_file, 'w') as f:
                f.write(f"# SUCCESSFUL SOLUTION for problem {slug}\n")
                f.write(f"# Sample {sampling_metrics['best_code_idx']} passed all tests\n")
                f.write(f"# Problem: {problem['problem_description'][:100]}...\n\n")
                f.write(sampling_metrics["best_code"])
            print(f"  💾 Best solution saved to: {best_code_file}")
        
        print(f"📊 Sampling results: {sampling_metrics['success_count']}/{self.config.k_samples} "
              f"= {sampling_metrics['success_rate']*100:.1f}% success rate")
        print(f"📁 All artifacts saved to: {problem_artifacts_dir}")
        
        return sampling_metrics["best_code"], sampling_metrics
    
    
    def run_single_problem(self, problem: Dict, problem_idx: int) -> Dict:
        """Run the full pipeline on a single problem."""
        slug = problem["task_id"]
        
        print(f"\n{'='*80}")
        print(f"Problem {problem_idx + 1}/{len(self.dataset)}: {slug}")
        print(f"{'='*80}")
        
        # Phase 1: Optimization
        optimized_var, opt_metrics = self.optimize_code(problem, problem_idx)
        
        # Phase 2: Sample K codes and test each
        final_code, sampling_metrics = self.sample_and_test_codes(optimized_var, problem, problem_idx)
        
        # Save optimization artifacts to the same directory
        problem_artifacts_dir = Path(sampling_metrics["artifacts_dir"])
        
        # Save optimization history
        opt_history_file = problem_artifacts_dir / "optimization_history.json"
        with open(opt_history_file, 'w') as f:
            import json
            json.dump(opt_metrics, f, indent=2)
        
        # Save the best optimized code from training
        opt_code_file = problem_artifacts_dir / "optimized_code.py"
        with open(opt_code_file, 'w') as f:
            f.write(f"# Best optimized code for problem {slug}\n")
            f.write(f"# Achieved loss: {opt_metrics['best_total_loss']:.4f}\n")
            f.write(f"# Problem: {problem['problem_description'][:100]}...\n\n")
            f.write(opt_metrics["best_code"])
        
        # Combine all metrics
        combined_metrics = {
            "problem_idx": problem_idx,
            "task_id": slug,
            "problem_description": problem["problem_description"],
            "starter_code": problem["starter_code"],
            
            # Phase 1: Optimization
            "optimization_steps": len(opt_metrics["optimization_steps"]),
            "best_optimization_loss": opt_metrics["best_total_loss"],
            "final_leetcode_loss": opt_metrics["final_leetcode_loss"],
            "final_syntax_loss": opt_metrics["final_syntax_loss"],
            "optimized_code": opt_metrics["best_code"],
            "optimized_code_length": len(opt_metrics["best_code"]),
            
            # Phase 2: Sampling and Testing
            "k_samples": sampling_metrics["k_samples"],
            "success_count": sampling_metrics["success_count"],
            "sampling_success_rate": sampling_metrics["success_rate"],
            "any_sample_success": sampling_metrics["any_success"],
            "best_sample_idx": sampling_metrics["best_code_idx"],
            "final_code": final_code,
            "final_code_length": len(final_code) if final_code else 0,
            
            # Overall pipeline success
            "pipeline_success": sampling_metrics["any_success"],  # True if any sample passed
        }
        
        # Log to wandb
        if not self.config.disable_wandb:
            wandb.log({
                f"problem_{problem_idx}/optimization_loss": opt_metrics["best_total_loss"],
                f"problem_{problem_idx}/leetcode_loss": opt_metrics["final_leetcode_loss"],
                f"problem_{problem_idx}/syntax_loss": opt_metrics["final_syntax_loss"],
                f"problem_{problem_idx}/k_samples": sampling_metrics["k_samples"],
                f"problem_{problem_idx}/success_count": sampling_metrics["success_count"],
                f"problem_{problem_idx}/sampling_success_rate": sampling_metrics["success_rate"],
                f"problem_{problem_idx}/any_sample_success": sampling_metrics["any_success"],
                f"problem_{problem_idx}/pipeline_success": sampling_metrics["any_success"],
                f"problem_{problem_idx}/optimized_code_length": len(opt_metrics["best_code"]),
                f"problem_{problem_idx}/final_code_length": len(final_code) if final_code else 0,
            })
        
        return combined_metrics
    
    def run_all_problems(self) -> pd.DataFrame:
        """Run the full pipeline on all problems."""
        print(f"\n🚀 Starting full pipeline experiment on {len(self.dataset)} problems")
        
        all_results = []
        total_success = 0
        
        for problem_idx, problem in enumerate(tqdm(self.dataset, desc="Processing problems")):
            try:
                result = self.run_single_problem(problem, problem_idx)
                all_results.append(result)
                
                if result["pipeline_success"]:
                    total_success += 1
                    
            except Exception as e:
                print(f"💥 Pipeline failed for problem {problem_idx}: {e}")
                # Add failed result
                all_results.append({
                    "problem_idx": problem_idx,
                    "task_id": problem.get("task_id", f"unknown_{problem_idx}"),
                    "pipeline_success": False,
                    "pipeline_error": str(e)
                })
        
        # Calculate and log final metrics
        results_df = pd.DataFrame(all_results)
        
        total_problems = len(all_results)
        success_rate = total_success / total_problems if total_problems > 0 else 0.0
        
        # Aggregate metrics
        agg_metrics = {
            "total_problems": total_problems,
            "total_success": total_success,
            "success_rate": success_rate,
            "optimization_losses": results_df["best_optimization_loss"].dropna().tolist(),
            "avg_sampling_success_rate": results_df["sampling_success_rate"].mean() if "sampling_success_rate" in results_df else 0.0,
            "total_samples_tested": results_df["k_samples"].sum() if "k_samples" in results_df else 0,
            "total_sample_successes": results_df["success_count"].sum() if "success_count" in results_df else 0,
        }
        
        # Log final summary to wandb
        if not self.config.disable_wandb:
            wandb.log({
                "summary/total_problems": total_problems,
                "summary/total_success": total_success,
                "summary/success_rate": success_rate,
                "summary/avg_sampling_success_rate": agg_metrics["avg_sampling_success_rate"],
                "summary/total_samples_tested": agg_metrics["total_samples_tested"],
                "summary/total_sample_successes": agg_metrics["total_sample_successes"],
                "summary/avg_optimization_loss": results_df["best_optimization_loss"].mean() if "best_optimization_loss" in results_df else float("nan"),
            })
        
        print(f"\n📊 Final Results:")
        print(f"  Pipeline success rate: {total_success}/{total_problems} = {success_rate*100:.1f}%")
        print(f"  Average sampling success rate: {agg_metrics['avg_sampling_success_rate']*100:.1f}%")
        print(f"  Total samples tested: {agg_metrics['total_samples_tested']}")
        print(f"  Total sample successes: {agg_metrics['total_sample_successes']}")
        
        return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Full Pipeline Experiment")
    
    # Problem selection
    parser.add_argument("--n_problems", type=int, default=5,
                       help="Number of random Easy problems to process")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--difficulty", type=str, default="Easy", choices=["Easy", "Medium", "Hard"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model and optimization
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard_mode", action="store_true", default=True)
    parser.add_argument("--use_cot", action="store_true", default=True)
    
    # Training parameters
    parser.add_argument("--n_steps", type=int, default=500,
                       help="Optimization steps per problem")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--leetcode_weight", type=float, default=0.7)
    parser.add_argument("--syntax_weight", type=float, default=0.3)
    parser.add_argument("--poe_gamma", type=float, default=0.5,
                       help="Product of Experts gamma (0.0 = disabled)")
    
    # Sampling
    parser.add_argument("--k_samples", type=int, default=10,
                       help="Number of code samples to generate and test per problem")
    
    # Logging and output
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--results_dir", type=str, default="./results/full_pipeline")
    parser.add_argument("--wandb_project", type=str, default="sdlm-full-pipeline")
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    print("🚀 Full Pipeline Experiment")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Problems: {args.n_problems} {args.difficulty} {args.split}")
    print(f"Optimization: {args.n_steps} steps, lr={args.lr}")
    print(f"Loss weights: LeetCode={args.leetcode_weight}, Syntax={args.syntax_weight}")
    print(f"PoE gamma: {args.poe_gamma}")
    print(f"Use CoT: {args.use_cot}")
    print(f"Sampling: {args.k_samples} samples per problem (temp=0.3)")
    print("=" * 50)
    
    # Create results directory
    results_dir = create_results_dir()
    args.results_dir = str(results_dir)
    
    # Run experiment
    experiment = FullPipelineExperiment(args)
    results_df = experiment.run_all_problems()
    
    # Save results
    results_file = results_dir / "full_pipeline_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Save config
    config_file = results_dir / "config.json"
    import json
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"📝 Config saved to: {config_file}")
    
    # Create artifacts README
    artifacts_readme = experiment.artifacts_dir / "README.md"
    with open(artifacts_readme, 'w') as f:
        f.write("# Full Pipeline Experiment Artifacts\n\n")
        f.write("This directory contains all artifacts from the full pipeline experiment.\n\n")
        f.write("## Directory Structure\n\n")
        f.write("Each problem has its own subdirectory: `problem_XXX_<slug>/`\n\n")
        f.write("### Per-Problem Artifacts:\n\n")
        f.write("- `problem_summary.json` - Complete problem summary with all metrics\n")
        f.write("- `optimization_history.json` - Detailed optimization training logs\n")
        f.write("- `optimized_code.py` - Best code from optimization phase\n")
        f.write("- `sample_XXX.py` - Individual sampled code snippets\n")
        f.write("- `sample_XXX_SUCCESS.py` - Samples that passed LeetCode tests\n")
        f.write("- `sample_XXX_ERROR.txt` - Samples that failed to generate\n")
        f.write("- `sample_XXX_result.json` - Detailed results for each sample\n")
        f.write("- `BEST_SOLUTION.py` - The first successful solution (if any)\n\n")
        f.write("### Key Files to Check:\n\n")
        f.write("1. Look for `BEST_SOLUTION.py` files - these contain working solutions\n")
        f.write("2. Check `*_SUCCESS.py` files for all working samples\n")
        f.write("3. Review `problem_summary.json` for success rates per problem\n")
        f.write("4. Examine `optimization_history.json` for training dynamics\n\n")
        f.write(f"## Experiment Configuration\n\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Problems tested: {args.n_problems}\n")
        f.write(f"- Samples per problem: {args.k_samples}\n")
        f.write(f"- Optimization steps: {args.n_steps}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Use CoT: {args.use_cot}\n")
        f.write(f"- PoE gamma: {args.poe_gamma}\n\n")
    print(f"📖 Artifacts README saved to: {artifacts_readme}")
    
    if not args.disable_wandb:
        wandb.finish()
    
    print("\n✅ Full pipeline experiment completed!")


if __name__ == "__main__":
    main()