#!/usr/bin/env python3
"""
Momentum optimization over a LeetCode dataset with KL monitoring and optional Product-of-Experts (PoE).

Core design
- Optimize code variables (one per problem) using MomentumLossFunction (judge-based). This is the ONLY training loss.
- Monitor distribution shifts with KL divergences between the Variable’s own token distributions:
  - KL(C0 || Ct): from initialization C0 (fluency/random) to current Ct (monitoring temperature).
  - KL(Ct-1 || Ct): from previous step Ct-1 to current Ct (expected to spike when learning happens).
- Temperature is critical: supports per-step schedules and a fixed monitoring temperature.
- Optional Product-of-Experts (PoE) sampling: replace the variable’s relaxed distribution q with q ⊙ p_lm^gamma (renormalized),
  where p_lm is a coding-LM next-token distribution along the current code (teacher-forced on the argmax code).
  This can guide updates towards a fluent prior while keeping gradients flowing through q.

Outputs
- results/<timestamp>/init=...__temp=...__sched=...__seed=.../trace.csv with per-step dataset-averaged metrics.
- samples/ with a few decoded examples.
- summary.csv aggregating final metrics per run.
- plots/ with auto-generated figures if --auto_plot is set.

Example
  python examples/momentum_kldiv_dataset.py \
    --model gpt2 \
    --max_items 25 --split train --difficulty Easy \
    --inits fluency random \
    --temperatures 0.7 10 100 1000 \
    --schedule cosine --t_final 0.7 \
    --n_steps 1500 --lr 5e-2 \
    --monitor_temperature 0.7 \
    --poe_gamma 0.0 \
    --seeds 42 \
    --results_dir ./results/momentum_kldiv_ds \
    --auto_plot
"""

import os
import json
import math
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

try:
    import numpy as np
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from sdlm.leetcode.dataset import load_leetcode_dataset
from sdlm.leetcode.utils import build_model, clean_for_submission
from sdlm.leetcode.momentum import MomentumLossFunction
from sdlm.textgrad.variables import Variable


# ------------------------------- Config -------------------------------- #

@dataclass
class RunConfig:
    # Dataset/setup
    model: str
    max_items: int
    split: str
    difficulty: str
    max_len_tokens: int
    # Momentum/optimization
    init_strategy: str
    temperature: float
    schedule: str
    t_final: float
    n_steps: int
    lr: float
    clip_norm: float
    seed: int
    device: str
    # Monitoring (not part of training loss)
    monitor_temperature: float
    # Product-of-Experts (optional guidance)
    poe_gamma: float
    poe_every: int
    # PPO-like clipping on spikes (applied as a scalar on momentum loss)
    kl_clip: Optional[float]  # if set, scales momentum loss by min(1, kl_clip / (KL(Ct-1||Ct)+eps))


# ---------------------------- Utilities -------------------------------- #

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int):
    torch.manual_seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def schedule_temperature(step, total_steps, t0, tf, mode: str) -> float:
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


def kl_q_to_q(q_from: torch.Tensor, q_to: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    KL(q_from || q_to) for monitoring; both are relaxed distributions.
    Shapes:
      q_from: (1, L, V)
      q_to:   (1, L, V)
    Returns scalar KL summed over positions.
    """
    if q_from.size(1) != q_to.size(1):
        L = min(q_from.size(1), q_to.size(1))
        q_from = q_from[:, :L, :]
        q_to = q_to[:, :L, :]
    q_from = q_from.clamp_min(eps)
    q_to = q_to.clamp_min(eps)
    return (q_from * (q_from.log() - q_to.log())).sum(dim=-1).sum()


def random_string_like_length(tokenizer, length_tokens: int) -> str:
    V = tokenizer.vocab_size
    ids = torch.randint(low=0, high=V, size=(1, length_tokens))
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    if not text.strip():
        text = "x" * max(1, length_tokens)
    return text


def compute_lm_next_token_distributions(model, tokenizer, prefix: str, code_text: str, device: str) -> torch.Tensor:
    """
    Compute per-position next-token distributions p_lm for a code_text under a coding LM, using teacher forcing:
      p_lm[t] = softmax(logits(prefix + code[:t])[-1])

    Returns:
      p_lm: (L, V) on device, dtype matches model embeddings dtype.
    """
    with torch.no_grad():
        code_ids = tokenizer(code_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
        L = code_ids.size(0)
        pre_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        V = model.get_input_embeddings().weight.shape[0]
        p_list = []
        for t in range(L):
            if t == 0:
                ctx_ids = pre_ids
            else:
                ctx_ids = torch.cat([pre_ids, code_ids[:t].unsqueeze(0)], dim=1)
            attn = torch.ones_like(ctx_ids)
            out = model(input_ids=ctx_ids, attention_mask=attn, return_dict=True)
            logp = out.logits[:, -1, :].log_softmax(dim=-1).squeeze(0)  # (V,)
            p_list.append(logp.exp())  # convert to probs to avoid numeric issues downstream
        p_lm = torch.stack(p_list, dim=0)  # (L, V)
        # Make sure dtype/device align with embeddings
        W = model.get_input_embeddings().weight
        p_lm = p_lm.to(device=W.device, dtype=W.dtype)
        return p_lm


def product_of_experts(q: torch.Tensor, p_lm: torch.Tensor, gamma: float, eps=1e-12) -> torch.Tensor:
    """
    Compute q_poe ∝ q * (p_lm^gamma).
    q:     (1, L, V)
    p_lm:  (L, V) (no grad)
    gamma: scalar >= 0
    Returns q_poe: (1, L, V), same dtype/device as q, gradients flow w.r.t q only.
    """
    if gamma <= 0:
        return q
    # Broadcast p_lm to (1, L, V)
    p = p_lm.clamp_min(eps).unsqueeze(0)  # (1, L, V)
    mix = q * (p ** gamma)
    mix = mix.clamp_min(eps)
    mix = mix / mix.sum(dim=-1, keepdim=True)  # normalize across vocab
    return mix


# ---------------------- Auto-Visualization helpers ---------------------- #

def find_runs(root: Path) -> List[Path]:
    runs = []
    for p in root.glob("init=*__temp=*__sched=*__seed=*"):
        if (p / "config.json").exists() and (p / "trace.csv").exists():
            runs.append(p)
    return sorted(runs)


def load_run(run_dir: Path) -> Tuple[Dict, pd.DataFrame]:
    with open(run_dir / "config.json", "r") as f:
        cfg = json.load(f)
    df = pd.read_csv(run_dir / "trace.csv")
    for k, v in cfg.items():
        df[f"cfg_{k}"] = v
    return cfg, df


def ewma_smooth(y, alpha: float):
    out = [0.0] * len(y)
    if len(y) == 0:
        return out
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def auto_plot_runs(root_dir: Path,
                   metric_list: List[str],
                   group_by: str = "cfg_temperature",
                   hue: str = "cfg_init_strategy",
                   facet: Optional[str] = "cfg_schedule",
                   smooth: str = "ewma",
                   alpha: float = 0.2,
                   style: str = "seaborn-darkgrid",
                   save_pdf: bool = False):
    if not _HAS_MPL:
        print("matplotlib not installed; skipping auto-visualization.")
        return

    runs = find_runs(root_dir)
    if not runs:
        print("No run dirs found for plotting.")
        return

    # Aggregate
    dfs = []
    for rd in runs:
        try:
            _, df = load_run(rd)
            dfs.append(df)
        except Exception as e:
            print(f"Skip plot load {rd}: {e}")
    if not dfs:
        print("No data aggregated for plotting.")
        return
    df = pd.concat(dfs, ignore_index=True)

    plt_dir = ensure_dir(root_dir / "plots")
    plt.style.use(style)

    for metric in metric_list:
        if metric not in df.columns:
            print(f"Metric '{metric}' not in dataframe, skipping.")
            continue

        if facet and facet in df.columns and len(df[facet].unique()) > 1:
            facets = sorted(df[facet].unique())
            ncols = min(3, len(facets))
            nrows = int(math.ceil(len(facets) / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     figsize=(6 * ncols, 4 * nrows),
                                     squeeze=False, sharey=False)
            axes = axes.flatten()
        else:
            facets = [None]
            fig, ax = plt.subplots(figsize=(8, 5))
            axes = [ax]

        for i, fv in enumerate(facets):
            ax = axes[i]
            df_plot = df if fv is None else df[df[facet] == fv]
            if df_plot.empty:
                continue

            gb_vals = sorted(df_plot[group_by].unique()) if group_by in df_plot.columns else [None]
            hue_vals = sorted(df_plot[hue].unique()) if hue in df_plot.columns else [None]

            for g in gb_vals:
                sub = df_plot if g is None else df_plot[df_plot[group_by] == g]
                for h in hue_vals:
                    sub2 = sub if h is None else sub[sub[hue] == h]
                    if sub2.empty:
                        continue
                    agg = sub2.groupby("step")[metric].agg(["mean", "std"]).reset_index()
                    X = agg["step"].values
                    Y = agg["mean"].values
                    if smooth == "ewma":
                        Yp = np.array(ewma_smooth(Y, alpha))
                    else:
                        Yp = Y
                    lbl = f"{group_by}={g}, {hue}={h}" if (g is not None and h is not None) else \
                          f"{group_by}={g}" if g is not None else f"{hue}={h}" if h is not None else "run"
                    ax.plot(X, Yp, label=lbl)

            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            if fv is not None:
                ax.set_title(f"{facet}={fv}")
            ax.legend(loc="best", fontsize=8)

        plt.tight_layout()
        out_png = plt_dir / f"{metric}.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        if save_pdf:
            fig.savefig(plt_dir / f"{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_png}")


# --------------------------- Main experiment --------------------------- #

def main():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("--max_items", type=int, default=1)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--difficulty", type=str, default="Easy", choices=["Easy", "Medium", "Hard"])
    ap.add_argument("--max_len_tokens", type=int, default=256, help="Truncate starter code to this many tokens for monitoring")

    # Model / momentum optimization
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--inits", type=str, nargs="+", default=["fluency", "random"], choices=["fluency", "random"])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 10.0, 100.0, 1000.0], help="Initial training temperatures")
    ap.add_argument("--schedule", type=str, default="constant", choices=["constant", "linear", "cosine", "exp_decay"])
    ap.add_argument("--t_final", type=float, default=0.7)
    ap.add_argument("--n_steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])

    # Monitoring (not part of training loss)
    ap.add_argument("--monitor_temperature", type=float, default=0.7, help="Fixed T for monitoring KL(C0||Ct) and KL(Ct-1||Ct)")

    # Product-of-Experts (optional guidance)
    ap.add_argument("--poe_gamma", type=float, default=0.0, help="0 disables PoE; >0 blends variable dist with LM next-token dist")
    ap.add_argument("--poe_every", type=int, default=1, help="Recompute LM distributions every N steps for PoE (costly)")

    # PPO-like clipping on spikes
    ap.add_argument("--kl_clip", type=float, default=None, help="If set, scales momentum loss by min(1, kl_clip / KL(Ct-1||Ct))")

    # IO
    ap.add_argument("--results_dir", type=str, default="./results/momentum_kldiv_ds")
    ap.add_argument("--auto_plot", action="store_true")
    ap.add_argument("--plot_metrics", type=str, nargs="+",
                    default=["momentum_loss_mean", "kl_C0_Ct_mean", "kl_Ctm1_Ct_mean"])
    ap.add_argument("--plot_style", type=str, default="seaborn-darkgrid")
    ap.add_argument("--plot_save_pdf", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = ensure_dir(Path(args.results_dir) / ts)

    # Build a single model (used as the judge and also for PoE LM guidance)
    stgs_kwargs = dict(
        stgs_hard=False, hard=False,
        init_temperature=0.7, temperature=0.7,
        learnable_temperature=False, use_bpttoken=False, bpttoken=False,
        hidden_state_conditioning=False
    )
    judge_model, tok = build_model(args.model, device, stgs_kwargs)

    # Load dataset
    raw_ds = load_leetcode_dataset(max_items=args.max_items, split=args.split, difficulty=args.difficulty)
    print(f"Loaded dataset: {len(raw_ds)} items (split={args.split}, difficulty={args.difficulty})")

    # Prepare items for monitoring and PoE prefix
    items = []
    for i, prob in enumerate(raw_ds):
        slug = prob.get("task_id", f"item{i}")
        desc = prob.get("problem_description", "")
        starter = clean_for_submission(prob.get("starter_code", "") or "def solution():\n    pass")

        # Prefix used for monitoring PoE teacher forcing and KL teacher (if needed later).
        prefix = f"Problem:\n{desc}\n\nCode:\n"

        # Tokenize starter and truncate (monitoring only)
        starter_ids = tok(starter, add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
        if starter_ids.size(0) == 0:
            starter_ids = tok("def solution():\n    pass", add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
        if starter_ids.size(0) > args.max_len_tokens:
            starter_ids = starter_ids[: args.max_len_tokens]
        starter_text = tok.decode(starter_ids, skip_special_tokens=True)

        items.append({
            "slug": slug,
            "desc": desc,
            "starter_text": starter_text,
            "prefix": prefix,
            "length": starter_ids.size(0),
        })

    print(f"Prepared {len(items)} items.")

    # Momentum judge question template (same for all items)
    momentum_question_tmpl = (
        "#TASK_DESCRIPTION:\n {t_descr}\n\n"
        "#INPUT:\n {input}\n\n"
        "Does the above input satisfy the task description?"
    )

    summary_rows: List[Dict] = []

    for seed in args.seeds:
        set_seed(seed)
        for init_strategy in args.inits:
            for t0 in args.temperatures:
                run_dir = ensure_dir(root_dir / f"init={init_strategy}__temp={t0:g}__sched={args.schedule}__seed={seed}")
                cfg = RunConfig(
                    model=args.model,
                    max_items=args.max_items,
                    split=args.split,
                    difficulty=args.difficulty,
                    max_len_tokens=args.max_len_tokens,
                    init_strategy=init_strategy,
                    temperature=t0,
                    schedule=args.schedule,
                    t_final=args.t_final,
                    n_steps=args.n_steps,
                    lr=args.lr,
                    clip_norm=args.clip_norm,
                    seed=seed,
                    device=device,
                    monitor_temperature=args.monitor_temperature,
                    poe_gamma=args.poe_gamma,
                    poe_every=args.poe_every,
                    kl_clip=args.kl_clip,
                )
                with open(run_dir / "config.json", "w") as f:
                    json.dump(asdict(cfg), f, indent=2)

                # Build Variables and Momentum losses per item
                variables: List[Variable] = []
                momentum_losses: List[MomentumLossFunction] = []
                for it in items:
                    L = it["length"]
                    if init_strategy == "fluency":
                        init_text = it["starter_text"]
                    else:
                        init_text = random_string_like_length(tok, length_tokens=L)

                    var = Variable(
                        tokenizer=tok,
                        initial_str=init_text,
                        template="{VARIABLE}",
                        use_fluency_constraint=False,
                        temperature=t0,
                        hard=False,
                        learnable_temperature=False,
                        device=device,
                    )
                    if hasattr(var, "train"):
                        var.train()
                    variables.append(var)

                    loss_i = MomentumLossFunction(
                        critic_dlm=judge_model,
                        momentum_question=momentum_question_tmpl,
                        Momentum_variables={"t_descr": it["desc"]},
                        momentum_answer="Yes",
                        use_cot=False,
                        answer_extractor="",
                    )
                    momentum_losses.append(loss_i)

                # Store C0 and Ct-1 at monitoring temperature for KL tracking
                q_init_list: List[torch.Tensor] = []
                q_prev_list: List[torch.Tensor] = []
                with torch.no_grad():
                    for v in variables:
                        # Best-effort: set monitoring temperature if supported
                        Tm = args.monitor_temperature
                        restored = None
                        if hasattr(v, "stgs"):
                            try:
                                restored = getattr(v.stgs, "temperature", None)
                                setattr(v.stgs, "temperature", Tm)
                            except Exception:
                                pass
                        _, q0, _ = v()  # (1, L, V)
                        q0 = q0.detach().clone()
                        q_init_list.append(q0)
                        q_prev_list.append(q0.clone())
                        # Restore previous training temp if we changed it
                        if restored is not None:
                            try:
                                setattr(v.stgs, "temperature", restored)
                            except Exception:
                                pass

                # Optimizer over all Variables
                all_params = []
                for v in variables:
                    all_params.extend(list(v.parameters()))
                assert sum(p.numel() for p in all_params) > 0, "No trainable params across variables"
                optim = torch.optim.Adam(all_params, lr=args.lr)

                rows = []
                for step in tqdm(range(args.n_steps), desc=f"run[{init_strategy}|T={t0}] seed={seed}"):
                    optim.zero_grad()

                    # Global training temperature schedule
                    T_now = schedule_temperature(step, args.n_steps, t0=t0, tf=args.t_final, mode=args.schedule)
                    for v in variables:
                        if hasattr(v, "set_temperature"):
                            try:
                                v.set_temperature(T_now)
                            except Exception:
                                pass
                        elif hasattr(v, "stgs"):
                            try:
                                setattr(v.stgs, "temperature", T_now)
                            except Exception:
                                pass

                    momentum_sum = 0.0
                    kl_c0_ct_sum = 0.0
                    kl_ctm1_ct_sum = 0.0
                    ent_lm_sum = 0.0  # avg LM entropy across positions (when PoE active)
                    n_items = len(variables)

                    for idx, (v, it, loss_fn) in enumerate(zip(variables, items, momentum_losses)):
                        # Training distribution at current temperature
                        _, q_train, _ = v()  # (1, Lq, V)
                        assert q_train.requires_grad

                        # Optional PoE guidance (recomputed every poe_every steps; can be heavy)
                        q_eff = q_train
                        if args.poe_gamma > 0.0 and (step % max(1, args.poe_every) == 0):
                            # Decode current argmax code as context for LM next-token dists
                            try:
                                code_argmax = v.forward_sample(temperature=0.0)  # greedy for context
                            except Exception:
                                code_argmax = ""
                            p_lm = compute_lm_next_token_distributions(
                                model=judge_model,
                                tokenizer=tok,
                                prefix=it["prefix"],
                                code_text=code_argmax,
                                device=device,
                            )  # (L, V)
                            # Track LM entropy (branching)
                            with torch.no_grad():
                                ent_lm = (-p_lm.clamp_min(1e-8) * p_lm.clamp_min(1e-8).log()).sum(dim=-1).mean()
                                ent_lm_sum += float(ent_lm.item())
                            q_eff = product_of_experts(q_train, p_lm, gamma=args.poe_gamma)

                        # Momentum loss (training objective)
                        m_loss = loss_fn(batched_one_hot=[q_eff])
                        # PPO-like clipping on KL spikes: scale momentum loss when KL(Ct-1||Ct) is large
                        scale = 1.0

                        # Monitoring distributions at fixed temperature
                        with torch.no_grad():
                            # Temporarily set monitor temperature, sample q_mon
                            Tm = args.monitor_temperature
                            restored = None
                            if hasattr(v, "stgs"):
                                try:
                                    restored = getattr(v.stgs, "temperature", None)
                                    setattr(v.stgs, "temperature", Tm)
                                except Exception:
                                    pass
                            _, q_mon, _ = v()
                            q_mon = q_mon.detach()

                            # KL(C0 || Ct) and KL(Ct-1 || Ct)
                            kl_c0_ct = kl_q_to_q(q_init_list[idx], q_mon)
                            kl_ctm1_ct = kl_q_to_q(q_prev_list[idx], q_mon)

                            # Update Ct-1 := Ct for next step
                            q_prev_list[idx] = q_mon.clone()

                            # Restore previous training temp
                            if restored is not None:
                                try:
                                    setattr(v.stgs, "temperature", restored)
                                except Exception:
                                    pass

                        if args.kl_clip is not None and args.kl_clip > 0:
                            # scale <= 1 if spike exceeds threshold (PPO-like)
                            denom = float(kl_ctm1_ct.item()) if torch.is_tensor(kl_ctm1_ct) else float(kl_ctm1_ct)
                            denom = max(denom, 1e-8)
                            scale = min(1.0, args.kl_clip / denom)

                        momentum_sum = momentum_sum + (scale * m_loss)
                        kl_c0_ct_sum = kl_c0_ct_sum + kl_c0_ct
                        kl_ctm1_ct_sum = kl_ctm1_ct_sum + kl_ctm1_ct

                    # Average across dataset
                    momentum_mean = momentum_sum / max(1, n_items)
                    kl_c0_ct_mean = kl_c0_ct_sum / max(1, n_items)
                    kl_ctm1_ct_mean = kl_ctm1_ct_sum / max(1, n_items)
                    ent_lm_mean = ent_lm_sum / max(1, n_items) if args.poe_gamma > 0 else float("nan")

                    # Backprop ONLY the momentum loss (poe+clip already accounted for)
                    momentum_mean.backward()
                    if args.clip_norm and args.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(all_params, args.clip_norm)
                    optim.step()

                    # Decode a sample from the first variable for logging
                    text_sample = ""
                    if variables and hasattr(variables[0], "forward_sample"):
                        try:
                            text_sample = variables[0].forward_sample(temperature=max(0.3, min(1.0, T_now)))
                        except Exception:
                            pass

                    rows.append({
                        "step": step,
                        "momentum_loss_mean": float(momentum_mean.item()),
                        "kl_C0_Ct_mean": float(kl_c0_ct_mean.item()),
                        "kl_Ctm1_Ct_mean": float(kl_ctm1_ct_mean.item()),
                        "lm_entropy_mean": ent_lm_mean,
                        "temperature": float(T_now),
                        "decoded_sample0": text_sample,
                    })

                    if (step + 1) % 100 == 0 or step == args.n_steps - 1:
                        pd.DataFrame(rows).to_csv(run_dir / "trace.csv", index=False)

                # Save a few final decoded samples
                samples_dir = ensure_dir(run_dir / "samples")
                for i, v in enumerate(variables[:5]):  # limit to 5 examples
                    try:
                        txt = v.forward_sample(temperature=0.3)
                    except Exception:
                        txt = ""
                    with open(samples_dir / f"final_item{i}.txt", "w") as f:
                        f.write(txt)

                # Summarize final aggregated metrics
                final = rows[-1] if rows else {}
                summary_rows.append({
                    "run_dir": str(run_dir),
                    "init_strategy": init_strategy,
                    "temperature": t0,
                    "schedule": args.schedule,
                    "t_final": args.t_final,
                    "seed": seed,
                    "final_momentum_mean": final.get("momentum_loss_mean", float("nan")),
                    "final_kl_C0_Ct_mean": final.get("kl_C0_Ct_mean", float("nan")),
                    "final_kl_Ctm1_Ct_mean": final.get("kl_Ctm1_Ct_mean", float("nan")),
                })

    # Save overall summary
    summary_path = root_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved all runs to: {root_dir}")
    print(f"Summary: {summary_path}")

    # Auto-visualize (aggregated metrics)
    if args.auto_plot:
        print("Generating auto-plots...")
        auto_plot_runs(
            root_dir=root_dir,
            metric_list=args.plot_metrics,  # e.g. ["momentum_loss_mean", "kl_C0_Ct_mean", "kl_Ctm1_Ct_mean"]
            group_by="cfg_temperature",
            hue="cfg_init_strategy",
            facet="cfg_schedule",
            smooth="ewma",
            alpha=0.2,
            style=args.plot_style,
            save_pdf=args.plot_save_pdf,
        )
        print(f"Plots saved under: {root_dir / 'plots'}")


if __name__ == "__main__":
    main()
