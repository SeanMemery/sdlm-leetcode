#!/usr/bin/env python3
"""
Run KL divergence experiment with SmolLM-135M model.

This script configures and runs the full KL divergence experiment using the 
HuggingFaceTB/SmolLM-135M model with appropriate settings for the 135M parameter scale.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Script location
    script_dir = Path(__file__).parent
    kldiv_script = script_dir / "kldiv_exp_full.py"
    
    if not kldiv_script.exists():
        print(f"Error: {kldiv_script} not found!")
        sys.exit(1)
    
    # Experiment configuration for SmolLM-135M
    cmd = [
        sys.executable, str(kldiv_script),
        
        # Model configuration
        "--model", "HuggingFaceTB/SmolLM-135M",
        
        # Dataset settings - can handle more items with smaller model
        "--max_items", "25",
        "--split", "train", 
        "--difficulty", "Easy",
        "--max_len_tokens", "384",  # Moderate length for 135M model
        
        # Training settings optimized for 135M model
        "--inits", "fluency", "random",
        "--temperatures", "0.5", "1.0", "2.0", "5.0", "10.0",  # Wider range for exploration
        "--schedules", "constant", "linear", "cosine", "exp_decay",  # All schedules
        "--t_final", "0.5",
        "--n_steps", "3000",  # More steps since model is smaller/faster
        "--lr", "2e-2",  # Higher learning rate for smaller model
        "--clip_norm", "1.5",
        "--seeds", "42", "123", "456",  # More seeds for statistical power
        
        # Monitoring settings
        "--monitor_temperature", "1.0",
        
        # Product-of-Experts settings
        "--poe_gamma", "0.0",  # Disabled initially
        "--poe_every", "5",  # More frequent updates possible with smaller model
        
        # PPO-like clipping
        "--kl_clip", "5.0",  # More aggressive clipping for smaller model
        
        # Output settings
        "--results_dir", "./results/smolLM_135M_kldiv",
        
        # Wandb experiment tracking
        "--wandb_project", "sdlm-kldiv-smolLM-135M",
        "--wandb_tags", "smolLM", "135M", "kldiv", "momentum", "easy",
        "--wandb_notes", "KL divergence experiment with SmolLM-135M on Easy LeetCode problems",
    ]
    
    print("=" * 80)
    print("Running KL Divergence Experiment with SmolLM-135M")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("Results saved to: ./results/smolLM_135M_kldiv")
        print("=" * 80)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError: Experiment failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)