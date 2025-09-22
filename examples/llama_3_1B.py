#!/usr/bin/env python3
"""
Run KL divergence experiment with Llama-3.2-1B model.

This script configures and runs the full KL divergence experiment using the 
meta-llama/Llama-3.2-1B-Instruct model with appropriate settings for the 1B parameter scale.
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
    
    # Experiment configuration for Llama-3.2-1B
    cmd = [
        sys.executable, str(kldiv_script),
        
        # Model configuration
        "--model", "meta-llama/Llama-3.2-1B-Instruct",
        
        # Dataset settings - smaller for 1B model
        "--max_items", "15",
        "--split", "train",
        "--difficulty", "Easy",
        "--max_len_tokens", "1024",  # Reduced for 1B model
        
        # Training settings optimized for 1B model
        "--inits", "fluency", "random",
        "--temperatures", "0.3", "0.7", "1.0", "3.0",  # More conservative temperatures
        "--schedules", "constant", "cosine", "exp_decay",  # Skip linear for time
        "--t_final", "0.3",
        "--n_steps", "2000",  # Reduced steps for faster iteration
        "--lr", "1e-2",  # Higher learning rate for faster convergence
        "--clip_norm", "1.0",
        "--seeds", "42", "123",  # Multiple seeds for robustness
        
        # Monitoring settings
        "--monitor_temperature", "0.7",
        
        # Product-of-Experts settings
        "--poe_gamma", "0.0",  # Disabled initially
        "--poe_every", "10",
        
        # PPO-like clipping
        "--kl_clip", "10.0",  # Conservative clipping for 1B model
        
        # Output settings
        "--results_dir", "./results/llama_3_2_1B_kldiv",
        
        # Wandb experiment tracking
        "--wandb_project", "sdlm-kldiv-llama3.2-1B",
        "--wandb_tags", "llama", "1B", "kldiv", "momentum", "easy",
        "--wandb_notes", "KL divergence experiment with Llama-3.2-1B-Instruct on Easy LeetCode problems",
    ]
    
    print("=" * 80)
    print("Running KL Divergence Experiment with Llama-3.2-1B-Instruct")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("Results saved to: ./results/llama_3_2_1B_kldiv")
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