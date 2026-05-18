"""
compute.py — DMFT Saddle‑Point Computation Script

Sets up all parameters, instantiates the DMFT solver and minimax
optimizer, then runs the saddle‑point search.

Each experiment corresponds to a dedicated directory.  If the
directory already exists, the script auto‑resumes from the latest
checkpoint.  Otherwise, a new experiment is created.

Usage:
    python compute.py                    (uses default exp_dir)
    python compute.py --dir exp_001      (specify experiment directory)
"""

import os
import sys
import json
import signal
import glob

import torch

from dmft_solver import (
    DMFTConfig,
    generate_dataset,
    DMFTSolver,
    DMFTMinimaxOptimizer,
)


# ============================================================
#  Experiment directory  (change this or pass --dir from CLI)
# ============================================================
EXP_DIR = "results/experiment_2"


# ============================================================
#  All hyper‑parameters — shared with main.py & algorithm.tex
# ============================================================
CONFIG = {
    # --- Network ---
    "network": {
        "N":     100,
        "g":     1,
        "dt":    0.1,
        "D_in":  10,
        "D_a":   2,
    },

    # --- Task ---
    "task": {
        "T":           30,
        "M":           2,
        "sigma_noise": 2.0,
        "gamma":       0.95,
        "c_p":         1.5,
        "c_v":         1.0,
        "beta":        10000.0,
    },

    # --- DMFT numerics ---
    "dmft_numerics": {
        "N_samples":  2000,
        "eps_reg":    1e-6,
        "dtype":      "torch.float64",
        "warm_start": True,
    },

    # --- Psi normalization ---
    "psi": {
        "method": "layer_norm_softmax",
        "tau":    0.5,
        "eps":    1e-8,
        "alpha":  2.0,
    },

    # --- Random seeds ---
    "random_seeds": {
        "solver_seed": 42,
        "data_seed":   12345,
    },

    # --- Optimisation ---
    "optimizer": {
        "scheme":          "diag",
        "max_iter":        10000,
        "tol_grad":        1e-3,
        "N_samples_start": 2000,
        "N_samples_end":   4000,
        "log_interval":    50,
        "save_interval":   200,
        "lr": {
            "C":      5e-4,
            "C_hat":  5e-4,
            "y":      5e-7,
            "y_hat":  5e-4,
            "z":      2e-3,
            "z_hat":  5e-4,
        },
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps":   1e-8,
    },
}


# ============================================================
#  Helpers
# ============================================================
def _find_latest_checkpoint(exp_dir):
    """Return path to ckpt_latest.pt if it exists, else the
    most recent ckpt_*.pt in the directory."""
    latest = os.path.join(exp_dir, "ckpt_latest.pt")
    if os.path.isfile(latest):
        return latest

    files = sorted(glob.glob(os.path.join(exp_dir, "ckpt_*.pt")))
    if files:
        return files[-1]
    return None


def save_config(exp_dir, config):
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {path}")


def load_config(exp_dir):
    path = os.path.join(exp_dir, "config.json")
    with open(path, "r") as f:
        return json.load(f)


def configs_match(saved, current):
    """Compare two config dicts; return (ok, mismatches_list)."""
    mismatches = []
    for section in sorted(set(list(saved.keys()) + list(current.keys()))):
        if section not in saved:
            mismatches.append(f"  + section '{section}' is new")
            continue
        if section not in current:
            mismatches.append(f"  - section '{section}' was removed")
            continue
        s = saved[section]
        c = current[section]
        if isinstance(s, dict) and isinstance(c, dict):
            for k in sorted(set(list(s.keys()) + list(c.keys()))):
                if k not in s:
                    mismatches.append(f"    {section}.{k}: (missing) → {c[k]}")
                elif k not in c:
                    mismatches.append(f"    {section}.{k}: {s[k]} → (removed)")
                elif s[k] != c[k]:
                    mismatches.append(f"    {section}.{k}: {s[k]} → {c[k]}")
        elif s != c:
            mismatches.append(f"  {section}: {s} → {c}")
    return len(mismatches) == 0, mismatches


# ============================================================
#  Build solver & optimizer from config dict
# ============================================================
def build_solver_and_optimizer(cfg_dict):
    nc = cfg_dict["network"]
    tc = cfg_dict["task"]
    dn = cfg_dict["dmft_numerics"]
    ps = cfg_dict.get("psi", {})
    rs = cfg_dict["random_seeds"]
    oc = cfg_dict["optimizer"]

    dtype = getattr(torch, dn["dtype"].split(".")[-1])
    torch.set_default_dtype(dtype)
    torch.manual_seed(rs["solver_seed"])

    cfg = DMFTConfig(
        N=nc["N"], g=nc["g"], dt=nc["dt"],
        D_in=nc["D_in"], D_a=nc["D_a"],
        T=tc["T"], M=tc["M"],
        sigma_noise=tc["sigma_noise"],
        gamma=tc["gamma"], c_p=tc["c_p"], c_v=tc["c_v"],
        beta=tc["beta"],
        N_samples=dn["N_samples"],
        eps_reg=dn["eps_reg"],
        dtype=dtype,
        warm_start=dn.get("warm_start", False),
        psi_method=ps.get("method", "layer_norm_softmax"),
        psi_tau=ps.get("tau", 1.0),
        psi_eps=ps.get("eps", 1e-8),
        psi_alpha=ps.get("alpha", 2.0),
    )

    input_seqs, targets = generate_dataset(cfg, seed=rs["data_seed"])
    solver = DMFTSolver(cfg, input_seqs, targets)

    optimizer = DMFTMinimaxOptimizer(
        solver,
        scheme=oc["scheme"],
        lr=oc["lr"],
        adam_beta1=oc["adam_beta1"],
        adam_beta2=oc["adam_beta2"],
        adam_eps=oc["adam_eps"],
    )

    return cfg, solver, optimizer


# ============================================================
#  Signal handler — convert SIGTERM → KeyboardInterrupt
# ============================================================
def _install_sigterm_handler():
    def _handler(signum, frame):
        print("\n  SIGTERM received, shutting down gracefully ...")
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _handler)


# ============================================================
#  Main
# ============================================================
def main():
    # Parse CLI for --dir
    exp_dir = EXP_DIR
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--dir" and i + 2 < len(sys.argv):
            exp_dir = sys.argv[i + 2]
        elif arg.startswith("--dir="):
            exp_dir = arg.split("=", 1)[1]

    print("=" * 60)
    print(f"  DMFT Solver — Compute")
    print(f"  Experiment directory: {exp_dir}")
    print("=" * 60)

    _install_sigterm_handler()

    exists = os.path.isdir(exp_dir)
    resume_ckpt = None

    if exists:
        print(f"\n  Directory '{exp_dir}' exists — attempting resume ...")

        # Load saved config and verify
        saved_cfg = load_config(exp_dir)
        ok, mismatches = configs_match(saved_cfg, CONFIG)
        if not ok:
            print("  WARNING: config mismatch detected:")
            for m in mismatches:
                print(m)
            print("  Proceeding with CURRENT config values (may cause issues).")

        # Build solver + optimizer
        cfg, solver, optimizer = build_solver_and_optimizer(CONFIG)

        # Find and load latest checkpoint
        ckpt_file = _find_latest_checkpoint(exp_dir)
        if ckpt_file is None:
            print("  No checkpoint found in directory — starting from scratch.")
        else:
            print(f"  Loading checkpoint: {ckpt_file}")
            resume_ckpt = optimizer.load_checkpoint(ckpt_file)
            print(f"  Resumed: iteration {resume_ckpt['iteration']}, "
                  f"S = {resume_ckpt['history'][-1]['S']:.6e}")

            completed = resume_ckpt['iteration'] + 1
            oc_cfg = CONFIG['optimizer']
            if completed >= oc_cfg['max_iter']:
                old_max = oc_cfg['max_iter']
                oc_cfg['max_iter'] = completed + old_max
                save_config(exp_dir, CONFIG)
                print(f"  Checkpoint reached iteration limit — "
                      f"auto-extending max_iter: {old_max} → {oc_cfg['max_iter']}")

    else:
        print(f"\n  Directory '{exp_dir}' does not exist — creating new experiment.")

        # Save config
        save_config(exp_dir, CONFIG)

        # Build solver + optimizer
        cfg, solver, optimizer = build_solver_and_optimizer(CONFIG)

    # --- Run ---
    oc = CONFIG["optimizer"]

    try:
        history = optimizer.run(
            max_iter=oc["max_iter"],
            tol_grad=oc["tol_grad"],
            N_samples_start=oc["N_samples_start"],
            N_samples_end=oc["N_samples_end"],
            verbose=True,
            log_interval=oc["log_interval"],
            checkpoint_dir=exp_dir,
            save_interval=oc["save_interval"],
            resume_checkpoint=resume_ckpt,
        )
        print("\nOptimisation finished successfully.")

    except KeyboardInterrupt:
        print("\nOptimisation interrupted.  Run again to resume.")


if __name__ == "__main__":
    main()
