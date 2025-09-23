#!/usr/bin/env python3
"""
Momentum optimization over a LeetCode dataset with KL monitoring and optional Product-of-Experts (PoE).

This is a modular, clean implementation of the KL divergence experiment framework.
"""

import os
import json
import math
import time
import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Run 'pip install wandb' for experiment tracking.")

from sdlm.leetcode.dataset import load_leetcode_dataset
from sdlm.leetcode.utils import build_model, clean_for_submission
from sdlm.leetcode.momentum import PythonSyntaxMomentumLoss, LeetCodeMomentumLoss
from sdlm.textgrad.variables import Variable

# Import SDLM to configure defaults
import sdlm


# ================================ Configuration ================================ #

@dataclass
class ExperimentConfig:
    """Complete configuration for a KL divergence experiment."""
    
    # Dataset configuration
    model: str = "gpt2"
    max_items: int = 10
    split: str = "train"
    difficulty: str = "Easy"
    max_len_tokens: int = 512
    
    # Training configuration
    init_strategies: List[str] = field(default_factory=lambda: ["fluency", "random"])
    temperatures: List[float] = field(default_factory=lambda: [0.7, 10.0, 100.0, 1000.0])
    schedules: List[str] = field(default_factory=lambda: ["constant", "linear", "cosine", "exp_decay"])
    t_final: float = 0.7
    n_steps: int = 10000
    lr: float = 5e-2
    seeds: List[int] = field(default_factory=lambda: [42])
    hard_mode: bool = False
    
    # Loss weights
    leetcode_loss_weight: float = 1.0
    syntax_loss_weight: float = 1.0
    
    # Monitoring configuration
    monitor_temperature: float = 0.7
    
    # Product-of-Experts configuration
    poe_gamma: float = 0.0
    poe_every: int = 1
    
    # PPO-like clipping
    kl_clip: Optional[float] = None
    
    # Output configuration
    results_dir: str = "./results/momentum_kldiv_ds"
    
    # Wandb configuration
    wandb_project: str = "sdlm-kldiv-experiments"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    disable_wandb: bool = False
    
    # Temperature settings by initialization strategy
    fluency_temperatures: Optional[List[float]] = None  # If None, uses default temperatures
    random_temperatures: Optional[List[float]] = None   # If None, uses default temperatures


@dataclass
class RunConfig:
    """Configuration for a single experimental run."""
    
    # Core identifiers
    model: str
    init_strategy: str
    temperature: float
    schedule: str
    seed: int
    
    # Derived from ExperimentConfig
    max_items: int
    split: str
    difficulty: str
    max_len_tokens: int
    t_final: float
    n_steps: int
    lr: float
    hard_mode: bool
    leetcode_loss_weight: float
    syntax_loss_weight: float
    monitor_temperature: float
    poe_gamma: float
    poe_every: int
    kl_clip: Optional[float]
    device: str
    
    # Wandb settings
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None


# ================================ Utilities ================================ #

class Utils:
    """Utility functions for the experiment."""
    
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass
    
    @staticmethod
    def schedule_temperature(step: int, total_steps: int, t0: float, tf: float, mode: str) -> float:
        """Compute temperature according to schedule."""
        if mode == "constant":
            return t0
        if mode == "linear":
            return t0 + (tf - t0) * (step / max(1, total_steps - 1))
        if mode == "cosine":
            cos_val = 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps - 1)))
            return tf + (t0 - tf) * cos_val
        if mode == "exp_decay":
            if t0 <= 0 or tf <= 0:
                return t0
            gamma = (tf / t0) ** (1.0 / max(1, total_steps - 1))
            return t0 * (gamma ** step)
        return t0
    
    @staticmethod
    def kl_divergence(q_from: torch.Tensor, q_to: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute KL(q_from || q_to) for monitoring."""
        if q_from.size(1) != q_to.size(1):
            L = min(q_from.size(1), q_to.size(1))
            q_from = q_from[:, :L, :]
            q_to = q_to[:, :L, :]
        q_from = q_from.clamp_min(eps)
        q_to = q_to.clamp_min(eps)
        return (q_from * (q_from.log() - q_to.log())).sum(dim=-1).sum()

# ================================ Wandb Integration ================================ #

class WandbLogger:
    """Handles all wandb logging functionality."""
    
    def __init__(self, config: RunConfig, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb and HAS_WANDB
        self.run_name = f"{config.init_strategy}_T{config.temperature:g}_{config.schedule}_seed{config.seed}"
        
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize wandb run."""
        try:
            wandb_tags = (self.config.wandb_tags or []) + [
                f"model:{self.config.model.replace('/', '_')}",
                f"init:{self.config.init_strategy}",
                f"schedule:{self.config.schedule}",
                f"difficulty:{self.config.difficulty}",
            ]
            
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.run_name,
                tags=wandb_tags,
                notes=self.config.wandb_notes,
                config=asdict(self.config),
                settings=wandb.Settings(start_method="fork"),
                reinit=True,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def log_metadata(self, model, dataset_size: int, run_dir: Path) -> None:
        """Log initial run metadata."""
        if not self.use_wandb:
            return
        
        wandb.log({
            "run_dir": str(run_dir),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "dataset_size": dataset_size,
        }, step=0)
    
    def log_step_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log per-step training metrics."""
        if not self.use_wandb:
            return
        
        wandb_metrics = {
            "train/leetcode_momentum": metrics["loss/leetcode_momentum"],
            "train/python_syntax": metrics["loss/python_syntax"],
            "train/total_combined": metrics["loss/total_combined"],
            "train/kl_C0_to_Ct": metrics["kl/C0_to_Ct"],
            "train/kl_Ct_minus_1_to_Ct": metrics["kl/Ct_minus_1_to_Ct"],
            "train/temperature": metrics["optimization/temperature"],
            "performance/step_duration_sec": metrics["timing/step_duration_sec"],
            "performance/elapsed_time_min": metrics["timing/elapsed_time_sec"] / 60,
        }
        
        if not math.isnan(metrics.get("poe/lm_entropy", float("nan"))):
            wandb_metrics["train/poe_lm_entropy"] = metrics["poe/lm_entropy"]
        
        # Log text sample occasionally
        if step % 100 == 0 and metrics.get("samples/generated_code", "").strip():
            wandb_metrics["samples/decoded_text"] = wandb.Html(
                f"<pre>{metrics['samples/generated_code'][:500]}</pre>"
            )
        
        wandb.log(wandb_metrics, step=step)
    
    def log_final_results(self, final_metrics: Dict[str, Any], training_stats: Dict[str, Any],
                         final_samples: List[str], trace_csv_path: Path) -> None:
        """Log final results and artifacts."""
        if not self.use_wandb:
            return
        
        # Final metrics
        wandb.log({
            "final/momentum_loss": final_metrics.get("momentum_loss_mean", float("nan")),
            "final/kl_C0_Ct": final_metrics.get("kl_C0_Ct_mean", float("nan")),
            "final/kl_Ctm1_Ct": final_metrics.get("kl_Ctm1_Ct_mean", float("nan")),
            "final/temperature": final_metrics.get("temperature", float("nan")),
            "performance/total_duration_min": training_stats["total_duration_min"],
            "performance/avg_step_duration_sec": training_stats["avg_step_duration_sec"],
            "performance/steps_completed": training_stats["steps_completed"],
        })
        
        # Final samples table
        if final_samples:
            sample_table = wandb.Table(columns=["item_id", "final_sample"])
            for i, sample in enumerate(final_samples):
                sample_table.add_data(i, sample[:1000])
            wandb.log({"final_samples": sample_table})
        
        # Upload trace as artifact
        if trace_csv_path.exists():
            trace_artifact = wandb.Artifact(f"trace_{self.run_name}", type="dataset")
            trace_artifact.add_file(str(trace_csv_path))
            wandb.log_artifact(trace_artifact)
    
    def finish(self) -> None:
        """Finish wandb run."""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")


# ================================ Product-of-Experts ================================ #

class ProductOfExperts:
    """Handles Product-of-Experts computations."""
    
    @staticmethod
    def compute_lm_distributions(model, tokenizer, prefix: str, code_text: str, device: str) -> torch.Tensor:
        """Compute per-position next-token distributions matching autoregressive generation."""
        with torch.no_grad():
            # Tokenize components
            code_ids = tokenizer(code_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
            
            # Concatenate prefix + code for full context
            full_context = torch.cat([prefix_ids, code_ids], dim=0).unsqueeze(0)  # (1, seq_len)
            attn_mask = torch.ones_like(full_context)
            
            # Get logits for entire sequence in one forward pass
            out = model(input_ids=full_context, attention_mask=attn_mask, return_dict=True, output_normal_logits=True)
            
            # Check if outputs has logits attribute
            if out is None:
                raise ValueError("Model returned None output")
            
            if not hasattr(out, 'logits') or out.logits is None:
                raise ValueError(f"Model output missing logits. Output type: {type(out)}")
            
            # Extract autoregressive logits for the code portion
            # out.logits shape: (1, seq_len, vocab_size)
            # Position i in logits predicts token i+1 in the sequence
            # So logits[prefix_len-1:prefix_len+code_len-1] predict the code tokens
            prefix_len = prefix_ids.size(0)
            code_len = code_ids.size(0)
            
            # Get logits that predict each code token autoregressively
            code_logits = out.logits[0, prefix_len-1:prefix_len + code_len - 1, :]  # (code_len, vocab_size)
            
            # Convert to probabilities
            p_lm = code_logits.softmax(dim=-1)  # (code_len, vocab_size)
            
            # Ensure correct device and dtype
            W = model.get_input_embeddings().weight
            return p_lm.to(device=W.device, dtype=W.dtype)
    
    @staticmethod
    def apply_poe(q: torch.Tensor, p_lm: torch.Tensor, gamma: float, eps: float = 1e-12) -> torch.Tensor:
        """Apply Product-of-Experts: q_poe ∝ q * (p_lm^gamma)."""
        if gamma <= 0:
            return q
        
        # Ensure q and p_lm have compatible sequence lengths
        q_len = q.size(1)  # (1, L_q, V)
        p_len = p_lm.size(0)  # (L_p, V)
        
        if q_len != p_len:
            # Truncate to the shorter length to avoid dimension mismatch
            min_len = min(q_len, p_len)
            q = q[:, :min_len, :]  # (1, min_len, V)
            p_lm = p_lm[:min_len, :]  # (min_len, V)
        
        p = p_lm.clamp_min(eps).unsqueeze(0)  # (1, L, V)
        mix = q * (p ** gamma)
        mix = mix.clamp_min(eps)
        return mix / mix.sum(dim=-1, keepdim=True)


# ================================ Single Run Executor ================================ #

class SingleRunExecutor:
    """Executes a single experimental run."""
    
    def __init__(self, config: RunConfig, items: List[Dict], model, tokenizer, 
                 momentum_question_template: str):
        self.config = config
        self.items = items
        self.model = model
        self.tokenizer = tokenizer
        self.momentum_template = momentum_question_template
        self.utils = Utils()
        self.poe = ProductOfExperts()
        
    def setup_run(self, run_dir: Path) -> Tuple[List[Variable], List[LeetCodeMomentumLoss], List[PythonSyntaxMomentumLoss], WandbLogger]:
        """Set up variables, losses, and logging for the run."""
        # Initialize wandb logger
        logger = WandbLogger(self.config, use_wandb=not self.config.wandb_project is None)
        logger.log_metadata(self.model, len(self.items), run_dir)
        
        # Create variables and momentum losses
        variables = []
        momentum_losses = []
        syntax_losses = []
        
        for item in self.items:
            # Initialize code
            init_text = item["starter_text"] + " " * (self.config.max_len_tokens - len(item["starter_text"]))
            
            # Create variable - ensure tokenizer vocab size matches
            print(f"Creating Variable with tokenizer vocab_size: {self.tokenizer.vocab_size}")
            var = Variable(
                tokenizer=self.tokenizer,
                initial_str=init_text,
                template="{VARIABLE}",
                init_strategy=self.config.init_strategy,
                temperature=self.config.temperature,
                hard=self.config.hard_mode,
                learnable_temperature=False,
                device=self.config.device,
            )
            if hasattr(var, "train"):
                var.train()
            variables.append(var)
            
            # Create LeetCode momentum loss
            loss_fn = LeetCodeMomentumLoss(
                critic_dlm=self.model,
                problem_description=item["desc"]
            )
            momentum_losses.append(loss_fn)
            
            # Create syntax validation loss
            syntax_loss_fn = PythonSyntaxMomentumLoss(
                critic_dlm=self.model
            )
            syntax_losses.append(syntax_loss_fn)
        
        return variables, momentum_losses, syntax_losses, logger
    
    def initialize_kl_tracking(self, variables: List[Variable]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Initialize KL tracking tensors."""
        q_init_list = []
        q_prev_list = []
        
        with torch.no_grad():
            for var in variables:
                # Set monitoring temperature temporarily
                restored = self._set_monitoring_temp(var)
                _, q0, _ = var()
                q0 = q0.detach().clone()
                q_init_list.append(q0)
                q_prev_list.append(q0.clone())
                # Restore original temperature
                self._restore_temp(var, restored)
        
        return q_init_list, q_prev_list
    
    def training_step(self, step: int, variables: List[Variable], momentum_losses: List[LeetCodeMomentumLoss],
                     syntax_losses: List[PythonSyntaxMomentumLoss], q_init_list: List[torch.Tensor], 
                     q_prev_list: List[torch.Tensor], optimizer: torch.optim.Optimizer, start_time: float) -> Dict[str, Any]:
        """Execute a single training step."""
        step_start_time = time.time()
        optimizer.zero_grad()
        
        # Update temperature schedule
        T_now = self.utils.schedule_temperature(
            step, self.config.n_steps, t0=self.config.temperature, 
            tf=self.config.t_final, mode=self.config.schedule
        )
        self._update_variable_temperatures(variables, T_now)
        
        # Compute losses and KL divergences
        momentum_sum = 0.0
        syntax_sum = 0.0
        kl_c0_ct_sum = 0.0
        kl_ctm1_ct_sum = 0.0
        ent_lm_sum = 0.0
        n_items = len(variables)
        
        for idx, (var, item, loss_fn, syntax_fn) in enumerate(zip(variables, self.items, momentum_losses, syntax_losses)):
            # Get training distribution
            _, q_train, _ = var()
            
            # Apply PoE if enabled
            q_eff = self._apply_poe_if_enabled(var, item, q_train, step)
            
            # Compute momentum loss
            m_loss = loss_fn(batched_one_hot=[q_eff])
            
            # Compute syntax loss
            s_loss = syntax_fn(batched_one_hot=[q_eff])
            
            # Monitor KL divergences
            kl_c0_ct, kl_ctm1_ct = self._monitor_kl_divergences(var, q_init_list[idx], q_prev_list, idx)
            
            momentum_sum += m_loss
            syntax_sum += s_loss
            kl_c0_ct_sum += kl_c0_ct
            kl_ctm1_ct_sum += kl_ctm1_ct
        
        # Backward pass with combined loss
        momentum_mean = momentum_sum / max(1, n_items)
        syntax_mean = syntax_sum / max(1, n_items)
        weighted_momentum = self.config.leetcode_loss_weight * momentum_mean
        weighted_syntax = self.config.syntax_loss_weight * syntax_mean
        total_loss = weighted_momentum + weighted_syntax
        total_loss.backward()
        
        optimizer.step()
        
        # Sample for logging
        _, _, text_sample = variables[0].forward(temperature=max(0.1, min(1.0, T_now)))
        
        step_end_time = time.time()
        
        return {
            "step": step,
            "loss/leetcode_momentum": float(momentum_mean.item()),
            "loss/python_syntax": float(syntax_mean.item()),
            "loss/leetcode_momentum_weighted": float(weighted_momentum.item()),
            "loss/python_syntax_weighted": float(weighted_syntax.item()),
            "loss/total_combined": float(total_loss.item()),
            "kl/C0_to_Ct": float(kl_c0_ct_sum.item() / max(1, n_items)),
            "kl/Ct_minus_1_to_Ct": float(kl_ctm1_ct_sum.item() / max(1, n_items)),
            "poe/lm_entropy": ent_lm_sum / max(1, n_items) if self.config.poe_gamma > 0 else float("nan"),
            "optimization/temperature": float(T_now),
            "samples/generated_code": text_sample,
            "timing/step_duration_sec": step_end_time - step_start_time,
            "timing/elapsed_time_sec": step_end_time - start_time,
        }
    
    def generate_final_samples(self, variables: List[Variable], samples_dir: Path, temperature: float) -> List[str]:
        """Generate and save final samples."""
        final_samples = []
        for i, var in enumerate(variables):
            try:
                _, _, txt = var.forward(temperature=temperature)
                final_samples.append(txt)
            except Exception:
                txt = ""
                final_samples.append(txt)
            
            with open(samples_dir / f"final_item{i}.txt", "w") as f:
                f.write(txt)
        
        return final_samples
    
    def _set_monitoring_temp(self, var: Variable) -> Any:
        """Set monitoring temperature and return original value."""
        if hasattr(var, "stgs"):
            try:
                restored = getattr(var.stgs, "temperature", None)
                setattr(var.stgs, "temperature", self.config.monitor_temperature)
                return restored
            except Exception:
                pass
        return None
    
    def _restore_temp(self, var: Variable, restored_temp: Any) -> None:
        """Restore original temperature."""
        if restored_temp is not None and hasattr(var, "stgs"):
            try:
                setattr(var.stgs, "temperature", restored_temp)
            except Exception:
                pass
    
    def _update_variable_temperatures(self, variables: List[Variable], temperature: float) -> None:
        """Update all variable temperatures."""
        for var in variables:
            if hasattr(var, "set_temperature"):
                try:
                    var.set_temperature(temperature)
                except Exception:
                    pass
            elif hasattr(var, "stgs"):
                try:
                    setattr(var.stgs, "temperature", temperature)
                except Exception:
                    pass
    
    def _apply_poe_if_enabled(self, var: Variable, item: Dict, q_train: torch.Tensor, step: int) -> torch.Tensor:
        """Apply Product-of-Experts if enabled."""
        if self.config.poe_gamma <= 0 or step % max(1, self.config.poe_every) != 0:
            return q_train
        
        try:
            code_argmax = var.forward_sample(temperature=0.0)
        except Exception:
            code_argmax = ""
        
        p_lm = self.poe.compute_lm_distributions(
            self.model, self.tokenizer, item["poe_prefix"], code_argmax, self.config.device
        )
        
        return self.poe.apply_poe(q_train, p_lm, self.config.poe_gamma)
    
    def _monitor_kl_divergences(self, var: Variable, q_init: torch.Tensor, 
                               q_prev_list: List[torch.Tensor], idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monitor KL divergences and update previous state."""
        with torch.no_grad():
            restored = self._set_monitoring_temp(var)
            _, q_mon, _ = var()
            q_mon = q_mon.detach()
            self._restore_temp(var, restored)
            
            kl_c0_ct = self.utils.kl_divergence(q_init, q_mon)
            kl_ctm1_ct = self.utils.kl_divergence(q_prev_list[idx], q_mon)
            
            # Update previous state
            q_prev_list[idx] = q_mon.clone()
            
            return kl_c0_ct, kl_ctm1_ct


# ================================ Main Experiment Runner ================================ #

class ExperimentRunner:
    """Main experiment runner that coordinates everything."""
    
    def __init__(self, config: ExperimentConfig, callback=None):
        self.config = config
        self.utils = Utils()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.callback = callback
    
    def get_temperatures_for_init(self, init_strategy: str) -> List[float]:
        """Get the appropriate temperatures for the given initialization strategy."""
        if init_strategy == "fluency" and self.config.fluency_temperatures is not None:
            return self.config.fluency_temperatures
        elif init_strategy == "random" and self.config.random_temperatures is not None:
            return self.config.random_temperatures
        else:
            # Fall back to default temperatures
            return self.config.temperatures
        
    def setup_experiment(self) -> Tuple[Path, Any, Any, List[Dict], str]:
        """Set up the experiment environment."""
        # Configure SDLM with the correct model FIRST
        print(f"Configuring SDLM with model: {self.config.model}")
        try:
            sdlm.configure_default_model(self.config.model)
            print(f"SDLM configured with model: {self.config.model}")
        except Exception as e:
            print(f"Warning: Could not configure SDLM: {e}")
        
        # Create results directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = self.utils.ensure_dir(Path(self.config.results_dir) / ts)
        
        # Build model
        stgs_kwargs = dict(
            stgs_hard=False, hard=False, init_temperature=0.7, temperature=0.7,
            learnable_temperature=False, use_bpttoken=False, bpttoken=False,
            hidden_state_conditioning=False
        )
        print(f"Building model: {self.config.model}")
        model, tokenizer = build_model(self.config.model, self.device, stgs_kwargs)
        print(f"Model loaded successfully: {self.config.model}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        
        # Load and prepare dataset
        dataset = load_leetcode_dataset(
            max_items=self.config.max_items, 
            split=self.config.split, 
            difficulty=self.config.difficulty
        )
        items = self._prepare_dataset_items(dataset, tokenizer)
        
        # Momentum question template
        momentum_template = (
            "#TASK_DESCRIPTION:\n {t_descr}\n\n"
            "#INPUT:\n {input}\n\n"
            "Does the above input satisfy the task description?"
        )
        
        print(f"Results directory: {root_dir}")
        print(f"Loaded {len(dataset)} items (split={self.config.split}, difficulty={self.config.difficulty})")
        print(f"Device: {self.device}")
        
        return root_dir, model, tokenizer, items, momentum_template
    
    def run_all_experiments(self) -> None:
        """Run all experimental configurations."""
        root_dir, model, tokenizer, items, momentum_template = self.setup_experiment()
        summary_rows = []
        
        # Calculate total runs considering different temperatures for different init strategies
        total_runs = 0
        for init_strategy in self.config.init_strategies:
            temperatures = self.get_temperatures_for_init(init_strategy)
            total_runs += len(self.config.seeds) * len(self.config.schedules) * len(temperatures)
        print(f"Running {total_runs} total experiments...")
        
        for seed in self.config.seeds:
            self.utils.set_seed(seed)
            for init_strategy in self.config.init_strategies:
                # Get appropriate temperatures for this initialization strategy
                temperatures = self.get_temperatures_for_init(init_strategy)
                for schedule in self.config.schedules:
                    for temperature in temperatures:
                        run_config = self._create_run_config(init_strategy, temperature, schedule, seed)
                        summary_row = self._execute_single_run(
                            run_config, root_dir, model, tokenizer, items, momentum_template
                        )
                        summary_rows.append(summary_row)
        
        # Save overall summary
        self._save_experiment_summary(root_dir, summary_rows)
        print(f"\nAll experiments completed. Results saved to: {root_dir}")
        print("Check wandb for visualizations and analysis.")

        return pd.DataFrame(summary_rows)
    
    def _prepare_dataset_items(self, dataset: List[Dict], tokenizer) -> List[Dict]:
        """Prepare dataset items for the experiment."""
        items = []
        for i, prob in enumerate(dataset):
            slug = prob.get("task_id", f"item{i}")
            desc = prob.get("problem_description", "")
            starter = clean_for_submission(prob.get("starter_code", "") or "def solution():\n    pass")
            
            poe_prefix = f"Problem:\n{desc}\n\nCode:```python\n"
            
            starter_ids = tokenizer(starter, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0]
            if starter_ids.size(0) > self.config.max_len_tokens:
                starter_ids = starter_ids[:self.config.max_len_tokens]
            starter_text = tokenizer.decode(starter_ids, skip_special_tokens=True)
            
            items.append({
                "slug": slug,
                "desc": desc,
                "starter_text": starter_text,
                "poe_prefix": poe_prefix,
                "length": starter_ids.size(0),
            })
        
        return items
    
    def _create_run_config(self, init_strategy: str, temperature: float, schedule: str, seed: int) -> RunConfig:
        """Create configuration for a single run."""
        return RunConfig(
            model=self.config.model,
            init_strategy=init_strategy,
            temperature=temperature,
            schedule=schedule,
            seed=seed,
            max_items=self.config.max_items,
            split=self.config.split,
            difficulty=self.config.difficulty,
            max_len_tokens=self.config.max_len_tokens,
            t_final=self.config.t_final,
            n_steps=self.config.n_steps,
            lr=self.config.lr,
            hard_mode=self.config.hard_mode,
            leetcode_loss_weight=self.config.leetcode_loss_weight,
            syntax_loss_weight=self.config.syntax_loss_weight,
            monitor_temperature=self.config.monitor_temperature,
            poe_gamma=self.config.poe_gamma,
            poe_every=self.config.poe_every,
            kl_clip=self.config.kl_clip,
            device=self.device,
            wandb_project=self.config.wandb_project if not self.config.disable_wandb else None,
            wandb_entity=self.config.wandb_entity,
            wandb_tags=self.config.wandb_tags,
            wandb_notes=self.config.wandb_notes,
        )
    
    def _execute_single_run(self, config: RunConfig, root_dir: Path, model, tokenizer, 
                           items: List[Dict], momentum_template: str) -> Dict:
        """Execute a single experimental run."""
        # Create run directory
        run_dir = self.utils.ensure_dir(
            root_dir / f"init={config.init_strategy}__temp={config.temperature:g}__sched={config.schedule}__seed={config.seed}"
        )
        start_time = time.time()
        
        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Set up run
        executor = SingleRunExecutor(config, items, model, tokenizer, momentum_template)
        variables, momentum_losses, syntax_losses, logger = executor.setup_run(run_dir)
        q_init_list, q_prev_list = executor.initialize_kl_tracking(variables)
        
        # Set up optimizer
        all_params = []
        for var in variables:
            all_params.extend(list(var.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=config.lr)
        
        # Training loop
        rows = []
        desc = f"run[{config.init_strategy}|T={config.temperature}|{config.schedule}] seed={config.seed}"
        print(f"Starting training loop with {config.n_steps} steps...")
        for step in tqdm(range(config.n_steps), desc=desc):
            metrics = executor.training_step(
                step, variables, momentum_losses, syntax_losses, q_init_list, q_prev_list, optimizer, start_time
            )
            
            # Log to wandb
            logger.log_step_metrics(metrics, step)
            
            # Add to local rows
            rows.append(metrics)
            
            # Call callback if provided
            if self.callback:
                self.callback(step, variables, metrics)
            
            # Save periodically
            if (step + 1) % 50 == 0 or step == config.n_steps - 1:
                trace_df = pd.DataFrame(rows)
                trace_df.to_csv(run_dir / "trace.csv", index=False)
                if (step + 1) % 500 == 0:
                    trace_df.to_csv(run_dir / f"trace_step_{step+1}.csv", index=False)
        
        # Final processing
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate final samples
        samples_dir = self.utils.ensure_dir(run_dir / "samples")
        final_samples = executor.generate_final_samples(variables, samples_dir, temperature=config.monitor_temperature)
        
        # Save training statistics
        training_stats = {
            "total_duration_sec": total_duration,
            "total_duration_min": total_duration / 60,
            "avg_step_duration_sec": total_duration / config.n_steps if config.n_steps > 0 else 0,
            "steps_completed": len(rows),
            "final_temperature": rows[-1]["optimization/temperature"] if rows else float("nan"),
        }
        with open(run_dir / "training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)
        
        # Log final results
        final_metrics = rows[-1] if rows else {}
        logger.log_final_results(final_metrics, training_stats, final_samples, run_dir / "trace.csv")
        logger.finish()
        
        print(f"\nCompleted run: {config.init_strategy}|T={config.temperature}|{config.schedule}|seed={config.seed}")
        print(f"Duration: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
        print(f"Average step time: {total_duration/config.n_steps:.3f} sec/step")
        
        return {
            "run_dir": str(run_dir),
            "init_strategy": config.init_strategy,
            "temperature": config.temperature,
            "schedule": config.schedule,
            "t_final": config.t_final,
            "seed": config.seed,
            "total_duration_min": total_duration / 60,
            "steps_completed": len(rows),
            "final_metrics": final_metrics,
            "momentum_loss_mean": final_metrics.get("loss/leetcode_momentum", float("nan")),
            "syntax_loss_mean": final_metrics.get("loss/python_syntax", float("nan")),
            "final_kl_C0_Ct_mean": final_metrics.get("kl/C0_Ct", float("nan")),
            "final_kl_Ctm1_Ct_mean": final_metrics.get("kl/Ctm1_Ct", float("nan")),
            "avg_step_duration_sec": total_duration / config.n_steps if config.n_steps > 0 else 0,
        }
    
    def _save_experiment_summary(self, root_dir: Path, summary_rows: List[Dict]) -> None:
        """Save experiment summary."""
        summary_path = root_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Summary: {summary_path}")


# ================================ CLI Interface ================================ #

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="KL Divergence Experiment for Code Optimization")
    
    # Dataset arguments
    parser.add_argument("--max_items", type=int, default=10, help="Maximum number of dataset items")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--difficulty", type=str, default="Easy", choices=["Easy", "Medium", "Hard"], help="Problem difficulty")
    parser.add_argument("--max_len_tokens", type=int, default=512, help="Max tokens for starter code")
    
    # Model and optimization arguments
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--inits", type=str, nargs="+", default=["fluency", "random"], choices=["fluency", "random"], help="Initialization strategies")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 10.0, 100.0, 1000.0], help="Default initial training temperatures")
    parser.add_argument("--fluency_temperatures", type=float, nargs="+", default=None, help="Temperatures for fluency initialization (defaults to --temperatures)")
    parser.add_argument("--random_temperatures", type=float, nargs="+", default=None, help="Temperatures for random initialization (defaults to --temperatures)")
    parser.add_argument("--schedules", type=str, nargs="+", default=["constant", "linear", "cosine", "exp_decay"], choices=["constant", "linear", "cosine", "exp_decay"], help="Temperature schedules")
    parser.add_argument("--t_final", type=float, default=0.1, help="Final temperature for schedules")
    parser.add_argument("--n_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--hard_mode", action="store_true", help="Enable hard mode for Variable initialization")
    parser.add_argument("--leetcode_loss_weight", type=float, default=1.0, help="Weight for LeetCode momentum loss")
    parser.add_argument("--syntax_loss_weight", type=float, default=1.0, help="Weight for Python syntax loss")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    
    # Monitoring arguments
    parser.add_argument("--monitor_temperature", type=float, default=0.1, help="Fixed temperature for KL monitoring")
    
    # Product-of-Experts arguments
    parser.add_argument("--poe_gamma", type=float, default=0.0, help="PoE mixing coefficient (0 disables)")
    parser.add_argument("--poe_every", type=int, default=1, help="Recompute PoE distributions every N steps")
    
    # PPO-like clipping
    parser.add_argument("--kl_clip", type=float, default=None, help="KL clipping threshold")
    
    # Output arguments
    parser.add_argument("--results_dir", type=str, default="./results/momentum_kldiv_ds", help="Results directory")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="sdlm-kldiv-experiments", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/username")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="Wandb tags")
    parser.add_argument("--wandb_notes", type=str, default=None, help="Wandb run notes")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig(
        model=args.model,
        max_items=args.max_items,
        split=args.split,
        difficulty=args.difficulty,
        max_len_tokens=args.max_len_tokens,
        init_strategies=args.inits,
        temperatures=args.temperatures,
        fluency_temperatures=args.fluency_temperatures,
        random_temperatures=args.random_temperatures,
        schedules=args.schedules,
        t_final=args.t_final,
        n_steps=args.n_steps,
        lr=args.lr,
        hard_mode=args.hard_mode,
        leetcode_loss_weight=args.leetcode_loss_weight,
        syntax_loss_weight=args.syntax_loss_weight,
        seeds=args.seeds,
        monitor_temperature=args.monitor_temperature,
        poe_gamma=args.poe_gamma,
        poe_every=args.poe_every,
        kl_clip=args.kl_clip,
        results_dir=args.results_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        disable_wandb=args.disable_wandb,
    )
    
    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()