#!/usr/bin/env python3
"""
Multi-model KL divergence experiment runner.

This script runs the KL divergence experiment across multiple language models:
- meta-llama/Llama-3.2-1B-Instruct
- HuggingFaceTB/SmolLM-135M-Instruct  
- distilbert/distilgpt2

Each model gets appropriate hyperparameter settings based on its size and capabilities.
"""

import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_id: str
    max_items: int
    max_len_tokens: int
    fluency_temperatures: List[float]
    random_temperatures: List[float]
    n_steps: int
    lr: float
    results_suffix: str
    wandb_tags: List[str]
    notes: str

@dataclass 
class ExperimentVariant:
    """A specific experiment variant (model + PoE setting)."""
    model_config: ModelConfig
    poe_enabled: bool
    poe_gamma: float
    variant_name: str
    results_suffix: str
    wandb_tags: List[str]

def get_experiment_variants() -> List[ExperimentVariant]:
    """Get all experiment variants (model + PoE combinations) to run."""
    model_configs = get_model_configs()
    variants = []
    
    for model_config in model_configs:
        # Without PoE
        variants.append(ExperimentVariant(
            model_config=model_config,
            poe_enabled=False,
            poe_gamma=0.0,
            variant_name=f"{model_config.name}-NoPoE",
            results_suffix=f"{model_config.results_suffix}_no_poe",
            wandb_tags=model_config.wandb_tags + ["no_poe"]
        ))
        
        # With PoE
        variants.append(ExperimentVariant(
            model_config=model_config,
            poe_enabled=True,
            poe_gamma=0.2,  # Reasonable PoE gamma value
            variant_name=f"{model_config.name}-PoE",
            results_suffix=f"{model_config.results_suffix}_with_poe",
            wandb_tags=model_config.wandb_tags + ["poe", "gamma_0.2"]
        ))
    
    return variants

def get_model_configs() -> List[ModelConfig]:
    """Get configurations for all models to test."""
    
    STEPS=2500
    LR=3e-3
    fluency_temperatures = [0.1, 0.5, 0.9]
    random_temperatures = [0.5, 3.0, 10.0]

    return [
        # Llama 3.2 1B - Medium model, balanced settings
        ModelConfig(
            name="Llama-3.2-1B",
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            max_items=1,
            max_len_tokens=1024,
            fluency_temperatures=fluency_temperatures,
            random_temperatures=random_temperatures,
            n_steps=STEPS,
            lr=LR,
            results_suffix="llama_3_2_1B_kldiv",
            wandb_tags=["llama", "1B", "kldiv", "momentum", "syntax"],
            notes="KL divergence experiment with Llama-3.2-1B-Instruct on Easy LeetCode problems with syntax validation"
        ),
        
        # SmolLM 135M - Smaller model, more conservative settings
        ModelConfig(
            name="SmolLM-135M", 
            model_id="HuggingFaceTB/SmolLM-135M-Instruct",
            max_items=1,
            max_len_tokens=1024,
            fluency_temperatures=fluency_temperatures,
            random_temperatures=random_temperatures,
            n_steps=STEPS,  
            lr=LR,     
            results_suffix="smollm_135M_kldiv",
            wandb_tags=["smollm", "135M", "kldiv", "momentum", "syntax"],
            notes="KL divergence experiment with SmolLM-135M-Instruct on Easy LeetCode problems with syntax validation"
        ),
        
        # DistilGPT2 - Legacy/baseline model, very conservative
        ModelConfig(
            name="DistilGPT2",
            model_id="distilbert/distilgpt2", 
            max_items=1,
            max_len_tokens=1024,
            fluency_temperatures=fluency_temperatures,
            random_temperatures=random_temperatures,
            n_steps=STEPS,  
            lr=LR,       
            results_suffix="distilgpt2_kldiv",
            wandb_tags=["distilgpt2", "baseline", "kldiv", "momentum", "syntax"],
            notes="KL divergence experiment with DistilGPT2 baseline on Easy LeetCode problems with syntax validation"
        ),
    ]

def build_command(variant: ExperimentVariant, base_config: Dict[str, Any]) -> List[str]:
    """Build the command line for running the experiment."""
    script_dir = Path(__file__).parent
    kldiv_script = script_dir / "kldiv_exp_full.py"
    
    model_config = variant.model_config
    
    cmd = [
        sys.executable, str(kldiv_script),
        
        # Model configuration
        "--model", model_config.model_id,
        
        # Dataset settings
        "--max_items", str(model_config.max_items),
        "--split", base_config["split"],
        "--difficulty", base_config["difficulty"],
        "--max_len_tokens", str(model_config.max_len_tokens),
        
        # Training settings
        "--inits"] + base_config["inits"] + [
        "--fluency_temperatures"] + [str(t) for t in model_config.fluency_temperatures] + [
        "--random_temperatures"] + [str(t) for t in model_config.random_temperatures] + [
        "--schedules"] + base_config["schedules"] + [
        "--t_final", str(base_config["t_final"]),
        "--n_steps", str(model_config.n_steps),
        "--lr", str(model_config.lr),
        "--seeds"] + [str(s) for s in base_config["seeds"]] + [
        
        # Monitoring settings
        "--monitor_temperature", str(base_config["monitor_temperature"]),
        
        # Product-of-Experts settings
        "--poe_gamma", str(variant.poe_gamma),
        "--poe_every", str(base_config["poe_every"]),

        # Output settings
        "--results_dir", f"./results/{variant.results_suffix}",
        
        # Wandb experiment tracking
        "--wandb_project", "sdlm-multi-model-kldiv",
        "--wandb_tags"] + variant.wandb_tags + [
        "--wandb_notes", f"{model_config.notes} (PoE: {'enabled' if variant.poe_enabled else 'disabled'})",
    ]
    
    return cmd

def run_experiment_variant(variant: ExperimentVariant, base_config: Dict[str, Any]) -> bool:
    """Run experiment for a single variant (model + PoE setting)."""
    print("=" * 80)
    print(f"Running KL Divergence Experiment: {variant.variant_name}")
    print("=" * 80)
    print(f"Model ID: {variant.model_config.model_id}")
    print(f"Max tokens: {variant.model_config.max_len_tokens}")
    print(f"Fluency temperatures: {variant.model_config.fluency_temperatures}")
    print(f"Random temperatures: {variant.model_config.random_temperatures}")
    print(f"Steps: {variant.model_config.n_steps}, LR: {variant.model_config.lr}")
    print(f"PoE enabled: {variant.poe_enabled}, gamma: {variant.poe_gamma}")
    print("=" * 80)
    
    cmd = build_command(variant, base_config)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ {variant.variant_name} experiment completed successfully!")
        print(f"Results saved to: ./results/{variant.results_suffix}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {variant.variant_name} experiment failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n💥 {variant.variant_name} experiment failed with unexpected error: {e}")
        return False

def save_experiment_summary(variants: List[ExperimentVariant], results: List[bool]):
    """Save a summary of all experiments."""
    summary = {
        "total_variants": len(variants),
        "successful": sum(results),
        "failed": len(results) - sum(results),
        "variants": []
    }
    
    for variant, success in zip(variants, results):
        summary["variants"].append({
            "variant_name": variant.variant_name,
            "model_name": variant.model_config.name,
            "model_id": variant.model_config.model_id,
            "poe_enabled": variant.poe_enabled,
            "poe_gamma": variant.poe_gamma,
            "success": success,
            "results_dir": f"./results/{variant.results_suffix}",
            "config": {
                "max_len_tokens": variant.model_config.max_len_tokens,
                "n_steps": variant.model_config.n_steps,
                "lr": variant.model_config.lr,
                "fluency_temperatures": variant.model_config.fluency_temperatures,
                "random_temperatures": variant.model_config.random_temperatures,
            }
        })
    
    summary_file = Path("./results/multi_model_experiment_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Experiment summary saved to: {summary_file}")
    return summary

def main():
    """Main entry point."""
    print("🚀 Starting Multi-Model KL Divergence Experiments (with and without PoE)")
    print("=" * 80)
    
    # Get configurations
    variants = get_experiment_variants()
    base_config = {
        "schedules": ["cosine"],
        "t_final": 0.1,
        "seeds": [42],
        "monitor_temperature": 0.1,
        "poe_every": 25,
        "split": "train",
        "difficulty": "Easy",
        "inits": ["fluency", "random"],
    }
    
    # Count unique models and variants
    unique_models = len(get_model_configs())
    total_variants = len(variants)
    
    # Calculate total training runs across all variants
    total_training_runs = 0
    for variant in variants:
        model_config = variant.model_config
        # Each variant runs: seeds × init_strategies × schedules × temperatures_for_init
        fluency_temps = len(model_config.fluency_temperatures)
        random_temps = len(model_config.random_temperatures)
        seeds = len(base_config["seeds"])
        schedules = len(base_config["schedules"])
        
        # fluency init runs with fluency temps, random init runs with random temps
        runs_per_variant = seeds * schedules * (fluency_temps + random_temps)
        total_training_runs += runs_per_variant
    
    print(f"Will run {total_variants} experiment variants:")
    print(f"  📊 {unique_models} models × 2 PoE settings (with/without) = {total_variants} total experiments")
    print(f"  🏃 Each experiment contains multiple training runs:")
    print(f"      Seeds: {len(base_config['seeds'])}, Schedules: {len(base_config['schedules'])}")
    print(f"      Init strategies: fluency + random (with different temperature sets)")
    print(f"  🎯 Total individual training runs across all experiments: {total_training_runs}")
    print()
    for i, variant in enumerate(variants, 1):
        model_config = variant.model_config
        fluency_temps = len(model_config.fluency_temperatures)
        random_temps = len(model_config.random_temperatures)
        seeds = len(base_config["seeds"])
        schedules = len(base_config["schedules"])
        runs_per_variant = seeds * schedules * (fluency_temps + random_temps)
        
        poe_status = "with PoE" if variant.poe_enabled else "without PoE"
        print(f"  {i}. {variant.model_config.name} {poe_status} ({runs_per_variant} training runs)")
    print()
    
    # Run experiments
    results = []
    for i, variant in enumerate(variants, 1):
        print(f"\n🔄 Starting experiment {i}/{len(variants)}")
        success = run_experiment_variant(variant, base_config)
        results.append(success)
        
        if not success:
            print(f"⚠️  Continuing with remaining variants despite {variant.variant_name} failure...")
    
    # Summary
    summary = save_experiment_summary(variants, results)
    
    print("\n" + "=" * 80)
    print("🏁 All Multi-Model Experiments Complete!")
    print("=" * 80)
    print(f"✅ Successful: {summary['successful']}/{summary['total_variants']}")
    print(f"❌ Failed: {summary['failed']}/{summary['total_variants']}")
    print("\nResults directories:")
    for variant in variants:
        poe_status = "PoE" if variant.poe_enabled else "NoPoE"
        print(f"  - {variant.model_config.name}-{poe_status}: ./results/{variant.results_suffix}")
    print(f"\nSummary: ./results/multi_model_experiment_summary.json")
    print("\n📊 All experiments use the same wandb project: 'sdlm-multi-model-kldiv'")
    print("=" * 80)
    
    # Return appropriate exit code
    if summary['failed'] > 0:
        print("⚠️  Some experiments failed. Check logs above for details.")
        return 1
    else:
        print("🎉 All experiments completed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)