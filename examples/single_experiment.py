#!/usr/bin/env python3
"""
Single Llama 1B experiment runner for manual parameter testing.

This script allows you to run a single experiment with specific hyperparameters
to test what works before running a full sweep.
"""

import sys
from pathlib import Path

# Import the experiment runner
sys.path.append(str(Path(__file__).parent))
from kldiv_exp_full import ExperimentConfig, ExperimentRunner

# ================================ CONFIGURATION ================================ #

CONFIG = {
    # Core experiment parameters
    "init_strategy": "random",  # "fluency" or "random"
    "temperature": 3.0,
    "hard_mode": True,
    "poe_gamma": 0.0,  # 0.0 = disabled
    "leetcode_to_syntax": 0.75,  # 0.0=all syntax, 1.0=all leetcode
    
    # Training parameters
    "n_steps": 5000,
    "lr": 1e-3,
    "seed": 42,
    
    # Output settings
    "results_dir": "./results/llama1b_manual",
    "wandb_project": "sdlm-llama1b-manual",
    "disable_wandb": False,
    
    # Callback settings
    "print_every": 10,  # Print test string every N steps
}

def create_print_callback(print_every=10):
    """Create a callback function that prints test strings every N steps."""
    def callback(step, variables, metrics):
        if step % print_every == 0:
            print(f"\n--- Step {step} ---")
            for i, var in enumerate(variables):
                current_text = var.decode()
                print(f"Variable {i}: {current_text}")
            
            # Print relevant metrics
            leetcode_loss = metrics.get("loss/leetcode_momentum", "N/A")
            syntax_loss = metrics.get("loss/python_syntax", "N/A")
            temperature = metrics.get("optimization/temperature", "N/A")
            print(f"LeetCode loss: {leetcode_loss:.4f}, Syntax loss: {syntax_loss:.4f}, Temp: {temperature:.3f}")
            print("-" * 50)
    
    return callback

def main():
    """Main entry point."""
    # Use config values (command line args can still override)
    import argparse
    
    parser = argparse.ArgumentParser(description="Single Llama 1B Experiment")
    
    # Core experiment parameters (defaults from CONFIG)
    parser.add_argument("--init_strategy", choices=["fluency", "random"], 
                       default=CONFIG["init_strategy"], help="Initialization strategy")
    parser.add_argument("--temperature", type=float, default=CONFIG["temperature"],
                       help="Temperature for sampling")
    parser.add_argument("--hard_mode", action="store_true", default=CONFIG["hard_mode"],
                       help="Enable hard mode for Variable initialization")
    parser.add_argument("--poe_gamma", type=float, default=CONFIG["poe_gamma"],
                       help="Product-of-Experts gamma value (0.0 = disabled)")
    parser.add_argument("--leetcode_to_syntax", type=float, default=CONFIG["leetcode_to_syntax"],
                       help="Loss weight ratio: 0.0=all syntax, 1.0=all leetcode")
    
    # Training parameters (defaults from CONFIG)
    parser.add_argument("--n_steps", type=int, default=CONFIG["n_steps"],
                       help="Number of training steps")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"],
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"],
                       help="Random seed")
    
    # Output settings (defaults from CONFIG)
    parser.add_argument("--results_dir", default=CONFIG["results_dir"],
                       help="Results directory")
    parser.add_argument("--wandb_project", default=CONFIG["wandb_project"],
                       help="Wandb project name")
    parser.add_argument("--disable_wandb", action="store_true", default=CONFIG["disable_wandb"],
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Compute individual loss weights from ratio
    leetcode_weight = args.leetcode_to_syntax
    syntax_weight = 1.0 - leetcode_weight
    
    print("🚀 Single Llama 1B Experiment")
    print("=" * 50)
    print(f"Init strategy: {args.init_strategy}")
    print(f"Temperature: {args.temperature}")
    print(f"Hard mode: {args.hard_mode}")
    print(f"PoE gamma: {args.poe_gamma}")
    print(f"LeetCode weight: {leetcode_weight:.2f}")
    print(f"Syntax weight: {syntax_weight:.2f}")
    print(f"Steps: {args.n_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 50)
    
    # Create experiment configuration
    exp_config = ExperimentConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_items=1,
        split="train",
        difficulty="Easy",
        max_len_tokens=1024,
        init_strategies=[args.init_strategy],
        temperatures=[args.temperature],
        schedules=["cosine"],
        t_final=0.1,
        n_steps=args.n_steps,
        lr=args.lr,
        hard_mode=args.hard_mode,
        leetcode_loss_weight=leetcode_weight,
        syntax_loss_weight=syntax_weight,
        seeds=[args.seed],
        monitor_temperature=0.1,
        poe_gamma=args.poe_gamma,
        poe_every=1,
        results_dir=args.results_dir,
        wandb_project=args.wandb_project,
        disable_wandb=args.disable_wandb
    )
    
    # Create callback for printing progress
    callback = create_print_callback(print_every=CONFIG["print_every"])
    
    # Run the experiment
    runner = ExperimentRunner(exp_config, callback=callback)
    results_df = runner.run_all_experiments()
    
    # Extract final metrics
    if len(results_df) > 0:
        result = results_df.iloc[0]
        print("\n📊 Final Results:")
        print(f"  LeetCode momentum loss: {result.get('momentum_loss_mean', 'N/A'):.4f}")
        print(f"  Python syntax loss: {result.get('syntax_loss_mean', 'N/A'):.4f}")
        print(f"  KL C0->Ct: {result.get('final_kl_C0_Ct_mean', 'N/A'):.4f}")
        print(f"  KL Ct-1->Ct: {result.get('final_kl_Ctm1_Ct_mean', 'N/A'):.4f}")
        print(f"  Training time: {result.get('total_duration_min', 'N/A'):.1f} minutes")
        print(f"  Results saved to: {result.get('run_dir', 'N/A')}")
    else:
        print("❌ No results generated")
    
    print("\n✅ Experiment completed!")

if __name__ == "__main__":
    main()