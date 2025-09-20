#!/usr/bin/env python3
"""
KL Divergence Experiment for Code Optimization

This experiment tracks KL divergence metrics during code optimization to understand:
- H1: KLDiv(C_{t-1} | C_t) should be flat with spikes when learning happens
- H2: KL spikes might need clipping for stability (PPO-like)  
- H3: Temperature effects should show in KLDiv(C_0 | C_t)

The experiment compares:
- Fluency vs Random initialization
- Different temperature settings
- KL divergence tracking over time

Visualizes:
- KLDiv(C_0 | C_t): divergence from initial distribution
- KLDiv(C_{t-1} | C_t): step-to-step divergence
- Loss curves and temperature effects
"""

import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

from sdlm.leetcode.dataset import load_leetcode_dataset, build_evaluator
from sdlm.leetcode.utils import build_model, clean_for_submission
from sdlm.leetcode.momentum import MomentumLossFunction
from sdlm.textgrad.variables import Variable
from scipy.ndimage import uniform_filter1d


def compute_kl_divergence(dist1, dist2):
    """
    Compute KL divergence between two distributions using torch.nn.functional.kl_div.
    
    Args:
        dist1, dist2: (batch_size, seq_len, vocab_size) probability distributions
        
    Returns:
        KL divergence averaged over batch and sequence dimensions
    """
    # torch.nn.functional.kl_div expects log probabilities as input and target probabilities
    # KL(dist1 || dist2) where dist1 is target, dist2 is input (log probs)
    
    # Ensure they are proper probability distributions
    dist1 = F.softmax(dist1, dim=-1)  # target (probabilities)
    log_dist2 = F.log_softmax(dist2, dim=-1)  # input (log probabilities)
    
    # Compute KL divergence
    kl = F.kl_div(log_dist2, dist1, reduction='batchmean')
    
    return kl.item()


def run_optimization_experiment(args, init_type="fluency", temperature=0.7):
    """Run a single optimization experiment with KL tracking."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # STGS settings
    stgs_kwargs = dict(
        stgs_hard=False, hard=False,
        init_temperature=temperature, temperature=temperature,
        learnable_temperature=False,
        use_bpttoken=False, bpttoken=False,
        hidden_state_conditioning=False
    )
    
    # Build model
    coder_model, tokenizer = build_model(args.model, device, stgs_kwargs)
    critic_model, _ = coder_model, tokenizer
    
    # Load one problem
    dataset = load_leetcode_dataset(max_items=1, split="test", difficulty="Easy")
    problem = dataset[0]
    evaluator = build_evaluator()
    
    slug = problem["task_id"]
    prompt = problem["problem_description"]
    starter_code = problem["starter_code"]
    init_code = clean_for_submission(starter_code)

    print(f"Problem: {slug}")
    print(f"Starter code: {starter_code}")
    
    # Initialize Variable with different strategies
    if init_type == "fluency":
        use_fluency = True
    else:  # random
        use_fluency = False
    
    # Pad to consistent length
    if len(init_code) < args.max_new_tokens:
        init_code = init_code + " " * (args.max_new_tokens - len(init_code))
    
    print(f"Using {init_type} initialization: '{init_code[:50]}...'")
    
    # Create Variable
    C_var = Variable(
        tokenizer=tokenizer,
        initial_str=init_code,
        template="{VARIABLE}",
        use_fluency_constraint=use_fluency,
        temperature=temperature,
        hard=False,
        learnable_temperature=False,
        device=device,
    )
    
    # Loss function
    momentum_question = (
        "#TASK_DESCRIPTION:\n {t_descr}\n\n"
        "#INPUT:\n {input}\n\n"
        "Does the above input satisfy the task description?"
    )
    
    loss_fn = MomentumLossFunction(
        critic_dlm=critic_model,
        momentum_question=momentum_question,
        Momentum_variables={"t_descr": prompt},
        momentum_answer="Yes",
        use_cot=False,
        answer_extractor="",
    )
    
    # Optimizer
    params = list(C_var.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # Get initial distribution C_0
    with torch.no_grad():
        _, C_0, _ = C_var()  # Initial distribution
        C_0 = F.softmax(C_0.squeeze(0), dim=-1)  # Convert to probabilities (seq_len, vocab_size)
    
    # Tracking arrays
    epochs = []
    train_losses = []
    test_losses = []
    kl_from_init = []  # KLDiv(C_0 | C_t)
    kl_step_to_step = []  # KLDiv(C_{t-1} | C_t)
    temperatures_tracked = []
    
    C_prev = C_0.clone()
    
    print(f"Starting optimization for {args.n_epoch} epochs...")
    
    for epoch in range(1, args.n_epoch + 1):
        optimizer.zero_grad()
        
        # Sample batch for training
        batch_oh = []
        for _ in range(args.batch_size):
            _, code_one_hot, _ = C_var()
            batch_oh.append(code_one_hot)
        
        # Compute loss and backpropagate
        train_loss = loss_fn(batched_one_hot=batch_oh)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(C_var.parameters(), 1.0)
        optimizer.step()
        
        # Get current distribution C_t for KL tracking
        with torch.no_grad():
            _, C_t_logits, _ = C_var()
            C_t = F.softmax(C_t_logits.squeeze(0), dim=-1)  # (seq_len, vocab_size)
            
            # Compute KL divergences
            kl_0_t = compute_kl_divergence(C_0.unsqueeze(0), C_t.unsqueeze(0))
            kl_prev_t = compute_kl_divergence(C_prev.unsqueeze(0), C_t.unsqueeze(0))
            
            # Test sampling
            test_text = C_var.forward_sample(temperature=args.t_test)
            test_loss = loss_fn(batched_input=[test_text]).item()
            
            # Update tracking
            epochs.append(epoch)
            train_losses.append(train_loss.item())
            test_losses.append(test_loss)
            kl_from_init.append(kl_0_t)
            kl_step_to_step.append(kl_prev_t)
            temperatures_tracked.append(temperature)
            
            # Update previous distribution
            C_prev = C_t.clone()
        
        # Periodic logging
        if epoch % args.log_every == 0 or epoch == args.n_epoch:
            print(f"Epoch {epoch}: train_loss={train_loss.item():.4f}, test_loss={test_loss:.4f}")
            print(f"  KL(C_0|C_t)={kl_0_t:.8f}, KL(C_{{t-1}}|C_t)={kl_prev_t:.8f}")
            print(f"  Sampled: '{test_text}...'")
            
            # Test on actual problem
            accepted, status = evaluator.evaluate(test_text.strip(), problem)
            if accepted:
                print(f"✓ Problem solved at epoch {epoch}!")
                break
    
    return {
        'init_type': init_type,
        'temperature': temperature,
        'epochs': epochs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'kl_from_init': kl_from_init,
        'kl_step_to_step': kl_step_to_step,
        'problem': slug
    }


def smooth_with_spikes(data, window_size=50, spike_threshold=2.0):
    """
    Smooth data while preserving spikes.
    
    Args:
        data: 1D array to smooth
        window_size: Size of smoothing window
        spike_threshold: Threshold for detecting spikes (in standard deviations)
    
    Returns:
        smoothed_data, spike_indices
    """
    data = np.array(data)
    
    # Compute smoothed version
    if len(data) > window_size:
        smoothed = uniform_filter1d(data.astype(float), size=window_size, mode='nearest')
    else:
        smoothed = data
    
    # Detect spikes: points that deviate significantly from smoothed version
    residuals = np.abs(data - smoothed)
    threshold = np.mean(residuals) + spike_threshold * np.std(residuals)
    spike_indices = np.where(residuals > threshold)[0]
    
    return smoothed, spike_indices


def plot_results(results_list, save_dir):
    """Create comprehensive plots of the KL divergence experiment."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('KL Divergence Experiment Results', fontsize=16)
    
    # Plot 1: Training Loss curves only (no test scores)
    ax = axes[0, 0]
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        ax.plot(result['epochs'], result['train_losses'], label=label, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: KL(C_0 | C_t) - Divergence from initial (smoothed)
    ax = axes[0, 1]
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        smoothed_kl, spike_indices = smooth_with_spikes(result['kl_from_init'])
        ax.plot(result['epochs'], smoothed_kl, label=label, linewidth=2)
        # Mark spikes with red dots
        if len(spike_indices) > 0:
            spike_epochs = [result['epochs'][i] for i in spike_indices if i < len(result['epochs'])]
            spike_values = [result['kl_from_init'][i] for i in spike_indices if i < len(result['kl_from_init'])]
            ax.scatter(spike_epochs, spike_values, color='red', s=20, alpha=0.7, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL(C_0 | C_t): Divergence from Initial (Smoothed, Spikes in Red)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: KL(C_{t-1} | C_t) - Step-to-step divergence (smoothed)
    ax = axes[0, 2]
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        smoothed_kl, spike_indices = smooth_with_spikes(result['kl_step_to_step'])
        ax.plot(result['epochs'], smoothed_kl, label=label, linewidth=2)
        # Mark spikes with red dots
        if len(spike_indices) > 0:
            spike_epochs = [result['epochs'][i] for i in spike_indices if i < len(result['epochs'])]
            spike_values = [result['kl_step_to_step'][i] for i in spike_indices if i < len(result['kl_step_to_step'])]
            ax.scatter(spike_epochs, spike_values, color='red', s=20, alpha=0.7, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL(C_{t-1} | C_t): Step-to-Step (Smoothed, Spikes in Red)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: KL step-to-step with log scale (raw data to see all spikes)
    ax = axes[1, 0]
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        # Use raw data for log scale to preserve all spikes
        kl_data = np.array(result['kl_step_to_step']) + 1e-8
        ax.semilogy(result['epochs'], kl_data, label=label, alpha=0.6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log KL Divergence')
    ax.set_title('KL(C_{t-1} | C_t): Log Scale (Raw Data, All Spikes Visible)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Temperature effect on final KL divergence
    ax = axes[1, 1]
    init_types = list(set(r['init_type'] for r in results_list))
    for init_type in init_types:
        temps = [r['temperature'] for r in results_list if r['init_type'] == init_type]
        final_kls = [r['kl_from_init'][-1] for r in results_list if r['init_type'] == init_type]
        ax.scatter(temps, final_kls, label=f"{init_type}_init", s=100, alpha=0.7)
        ax.plot(temps, final_kls, alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Final KL(C_0 | C_t)')
    ax.set_title('Temperature Effect on Final Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative KL divergence
    ax = axes[1, 2]
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        cumulative_kl = np.cumsum(result['kl_step_to_step'])
        ax.plot(result['epochs'], cumulative_kl, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative KL Divergence')
    ax.set_title('Cumulative Step-to-Step Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save main combined plot
    plot_path = save_dir / "kl_divergence_experiment_overview.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot: {plot_path}")
    plt.close()
    
    # Create two separate PNG files for KL curves: smoothed and raw
    
    # 1. Smoothed KL curves with spike detection
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Smoothed KL(C_0 | C_t) with spikes
    plt.subplot(2, 2, 1)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        smoothed_data, spike_indices = smooth_with_spikes(result['kl_from_init'], window_size=50, spike_threshold=2.0)
        plt.plot(result['epochs'], smoothed_data, label=label, linewidth=2)
        if len(spike_indices) > 0:
            spike_epochs = [result['epochs'][i] for i in spike_indices if i < len(result['epochs'])]
            spike_values = [result['kl_from_init'][i] for i in spike_indices if i < len(result['kl_from_init'])]
            plt.scatter(spike_epochs, spike_values, color='red', s=20, alpha=0.7, zorder=5)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL(C_0 | C_t): Smoothed with Spike Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed KL(C_{t-1} | C_t) with spikes
    plt.subplot(2, 2, 2)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        smoothed_data, spike_indices = smooth_with_spikes(result['kl_step_to_step'], window_size=50, spike_threshold=2.0)
        plt.plot(result['epochs'], smoothed_data, label=label, linewidth=2)
        if len(spike_indices) > 0:
            spike_epochs = [result['epochs'][i] for i in spike_indices if i < len(result['epochs'])]
            spike_values = [result['kl_step_to_step'][i] for i in spike_indices if i < len(result['kl_step_to_step'])]
            plt.scatter(spike_epochs, spike_values, color='red', s=20, alpha=0.7, zorder=5)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL(C_{t-1} | C_t): Smoothed with Spike Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Training loss (smoothed)
    plt.subplot(2, 2, 3)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        smoothed_loss, _ = smooth_with_spikes(result['train_losses'], window_size=20, spike_threshold=2.0)
        plt.plot(result['epochs'], smoothed_loss, label=label, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Temperature effect on final KL divergence
    plt.subplot(2, 2, 4)
    init_types = list(set(r['init_type'] for r in results_list))
    for init_type in init_types:
        temps = [r['temperature'] for r in results_list if r['init_type'] == init_type]
        final_kls = [r['kl_from_init'][-1] for r in results_list if r['init_type'] == init_type]
        plt.scatter(temps, final_kls, label=f"{init_type}_init", s=100, alpha=0.7)
        plt.plot(temps, final_kls, alpha=0.5)
    plt.xlabel('Temperature')
    plt.ylabel('Final KL(C_0 | C_t)')
    plt.title('Temperature Effect on Final Divergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    smoothed_path = save_dir / "kl_curves_smoothed.png"
    plt.savefig(smoothed_path, dpi=300, bbox_inches='tight')
    print(f"Saved smoothed plot: {smoothed_path}")
    plt.close()
    
    # 2. Raw unsmoothed KL curves
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw KL(C_0 | C_t)
    plt.subplot(2, 2, 1)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        plt.plot(result['epochs'], result['kl_from_init'], label=label, linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL(C_0 | C_t): Raw Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Raw KL(C_{t-1} | C_t)
    plt.subplot(2, 2, 2)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        plt.plot(result['epochs'], result['kl_step_to_step'], label=label, linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL(C_{t-1} | C_t): Raw Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Raw KL(C_{t-1} | C_t) in log scale
    plt.subplot(2, 2, 3)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        # Filter out non-positive values for log scale
        valid_indices = [i for i, val in enumerate(result['kl_step_to_step']) if val > 0]
        if valid_indices:
            epochs_filtered = [result['epochs'][i] for i in valid_indices]
            kl_filtered = [result['kl_step_to_step'][i] for i in valid_indices]
            plt.semilogy(epochs_filtered, kl_filtered, label=label, linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Log KL Divergence')
    plt.title('KL(C_{t-1} | C_t): Log Scale (Raw Data, All Spikes Visible)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative KL divergence
    plt.subplot(2, 2, 4)
    for result in results_list:
        label = f"{result['init_type']}_T{result['temperature']}"
        cumulative_kl = np.cumsum(result['kl_step_to_step'])
        plt.plot(result['epochs'], cumulative_kl, label=label, linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative KL Divergence')
    plt.title('Cumulative Step-to-Step Divergence (Raw)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    raw_path = save_dir / "kl_curves_raw.png"
    plt.savefig(raw_path, dpi=300, bbox_inches='tight')
    print(f"Saved raw plot: {raw_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='KL Divergence Experiment for Code Optimization')
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--n_epoch", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--t_test", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/kl_experiment_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Experimental conditions
    conditions = [
        ("fluency", 0.3),
        ("fluency", 0.6),
        ("fluency", 0.8),
        ("fluency", 1.0),
        ("random", 0.3),
        ("random", 0.6),
        ("random", 0.8),
        ("random", 1.0),
    ]
    
    results = []
    
    print("Starting KL Divergence Experiment")
    print("=" * 60)
    
    for init_type, temperature in conditions:
        print(f"\nRunning: {init_type} initialization, temperature={temperature}")
        result = run_optimization_experiment(args, init_type, temperature)
        results.append(result)
        
        # Save individual result
        torch.save(result, results_dir / f"result_{init_type}_T{temperature}.pt")
    
    # Create visualizations
    plot_results(results, results_dir)
    
    # Save all results
    torch.save(results, results_dir / "all_results.pt")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print("Hypothesis Testing:")
    print("H1: Check KL(C_{t-1}|C_t) plots for flat regions with spikes")
    print("H2: Look for instability spikes that might benefit from clipping")
    print("H3: Examine temperature effects on KL(C_0|C_t) progression")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()