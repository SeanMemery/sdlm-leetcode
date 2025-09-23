#!/usr/bin/env python3
"""
Wandb sweep experiment for Llama-3.2-1B hyperparameter optimization.

This script creates and runs a wandb sweep to find optimal hyperparameters across:
- Temperature values for fluency and random initialization
- Hard mode (enabled/disabled)
- PoE gamma values
- Loss weight ratio for LeetCode vs Python syntax losses

The objective is to minimize the LeetCode momentum loss.
"""

import sys
import wandb
from pathlib import Path

# Import the experiment runner
sys.path.append(str(Path(__file__).parent))
from kldiv_exp_full import ExperimentConfig, ExperimentRunner

# Sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "loss/leetcode_momentum",
        "goal": "minimize"
    },
    "parameters": {
        # Temperature exploration
        "fluency_temperature": {
            "distribution": "log_uniform_values",
            "min": 0.2,
            "max": 1.0
        },
        "random_temperature": {
            "distribution": "log_uniform_values", 
            "min": 0.7,
            "max": 10.0
        },
        
        # Initialization strategy
        "init_strategy": {
            "values": ["fluency", "random"]
        },
        
        # Hard mode exploration
        "hard_mode": {
            "values": [True, False]
        },
        
        # PoE gamma exploration
        "poe_gamma": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        
        # Loss weight ratio: leetcode_to_syntax
        # 0.0 = all syntax (leetcode=0.0, syntax=1.0)
        # 0.5 = equal weight (leetcode=0.5, syntax=0.5) 
        # 1.0 = all leetcode (leetcode=1.0, syntax=0.0)
        "leetcode_to_syntax": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        
        # Chain-of-thought mode
        "use_cot": {
            "values": [False, True]
        }
    }
}

def train():
    """Training function called by wandb.agent for each sweep run."""
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config
    
    print(f"Running: init={config.init_strategy}, hard={config.hard_mode}, temp={config.fluency_temperature if config.init_strategy == 'fluency' else config.random_temperature:.3f}")
    
    try:
        # Determine temperature based on init strategy
        if config.init_strategy == "fluency":
            temperature = config.fluency_temperature
        else:
            temperature = config.random_temperature
        
        # Compute individual loss weights from ratio
        leetcode_weight = config.leetcode_to_syntax
        syntax_weight = 1.0 - leetcode_weight
        
        # Create experiment configuration
        exp_config = ExperimentConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            max_items=1,
            split="train",
            difficulty="Easy",
            max_len_tokens=1024,
            init_strategies=[config.init_strategy],
            temperatures=[temperature],
            schedules=["cosine"],
            t_final=0.3,
            n_steps=5000,
            lr=5e-3,
            lr_scheduler="cosine",
            lr_scheduler_kwargs={},
            hard_mode=config.hard_mode,
            leetcode_loss_weight=leetcode_weight,
            syntax_loss_weight=syntax_weight,
            seeds=[325],
            monitor_temperature=0.3,
            poe_gamma=config.poe_gamma,
            use_cot=config.use_cot,
            results_dir=f"./results/llama1b_sweep/run_{run.id}",
            disable_wandb=True  # Disable wandb in the experiment
        )
        
        # Run the experiment directly
        runner = ExperimentRunner(exp_config)
        results_df = runner.run_all_experiments()
        
        # Extract the best LeetCode momentum loss from results DataFrame
        best_loss = float('inf')
        if 'momentum_loss_mean' in results_df.columns:
            momentum_losses = results_df['momentum_loss_mean'].dropna()
            if len(momentum_losses) > 0:
                valid_losses = momentum_losses[momentum_losses != float('inf')]
                if len(valid_losses) > 0:
                    best_loss = valid_losses.min()
        
        if best_loss != float('inf'):
            wandb.log({"loss/leetcode_momentum": best_loss})
            print(f"📊 Logged LeetCode loss: {best_loss:.4f}")
        else:
            print("⚠️ Could not find LeetCode loss metric")
            wandb.log({"loss/leetcode_momentum": float('nan')})
        
    except Exception as e:
        print(f"❌ Experiment error: {e}")
        wandb.log({"run_failed": True, "error": str(e), "loss/leetcode_momentum": float('nan')})

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama 1B Wandb Sweep Experiment")
    parser.add_argument("--project", default="sdlm-llama1b-hyperopt", 
                       help="Wandb project name")
    parser.add_argument("--entity", default=None,
                       help="Wandb entity (optional)")
    parser.add_argument("--count", type=int, default=999999,
                       help="Number of sweep runs")
    
    args = parser.parse_args()
    
    print("🚀 Llama 1B Hyperparameter Optimization Sweep")
    print("Optimizing: temperature, init strategy, hard mode, PoE gamma, loss ratio")
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project,
        entity=args.entity
    )
    
    print(f"Created sweep: {sweep_id}")
    print(f"Starting {args.count} sweep runs...")
    
    # Run sweep agent
    wandb.agent(
        sweep_id=sweep_id,
        function=train,
        count=args.count
    )
    
    print("✅ Sweep completed!")

if __name__ == "__main__":
    main()