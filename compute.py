"""
compute.py — DMFT Saddle‑Point Computation Script

Sets up all parameters, instantiates the DMFT solver and minimax
optimizer, then runs the saddle‑point search.

Usage:  python compute.py
"""

import torch

from dmft_solver import (
    DMFTConfig,
    generate_dataset,
    DMFTSolver,
    DMFTMinimaxOptimizer,
)


def main():
    # ============================================================
    # 1.  Parameters — shared with main.py and algorithm.tex
    # ============================================================

    # --- Network ---
    N     = 100
    g     = 1.5
    dt    = 0.1
    D_in  = 10
    D_a   = 2

    # --- Task ---
    T           = 30
    M           = 2
    sigma_noise = 2.0
    gamma       = 0.95
    c_p         = 1.0
    c_v         = 1.0
    beta        = 10000.0

    # --- DMFT numerics ---
    N_samples  = 2000
    dtype      = torch.float64
    random_seed = 42

    # --- Optimisation ---
    scheme          = 'mixed'
    max_iter        = 2000
    tol_grad        = 1e-3
    N_samples_start = 2000
    N_samples_end   = 4000
    log_interval    = 50

    # Per‑variable learning rates
    lr = {
        'C':      5e-4,
        'C_hat':  5e-4,
        'y':      5e-7,
        'y_hat':  5e-4,
        'z':      2e-3,
        'z_hat':  5e-4,
    }

    # Adam hyper‑parameters
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps   = 1e-8

    # ============================================================
    # 2.  Setup
    # ============================================================

    torch.set_default_dtype(dtype)
    torch.manual_seed(random_seed)

    cfg = DMFTConfig(
        N=N, g=g, dt=dt, D_in=D_in, D_a=D_a,
        T=T, M=M, sigma_noise=sigma_noise,
        gamma=gamma, c_p=c_p, c_v=c_v, beta=beta,
        N_samples=N_samples, dtype=dtype,
    )

    input_seqs, targets = generate_dataset(cfg, seed=12345)

    solver = DMFTSolver(cfg, input_seqs, targets)

    optimizer = DMFTMinimaxOptimizer(
        solver,
        scheme=scheme,
        lr=lr,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
    )

    # ============================================================
    # 3.  Run
    # ============================================================

    history = optimizer.run(
        max_iter=max_iter,
        tol_grad=tol_grad,
        N_samples_start=N_samples_start,
        N_samples_end=N_samples_end,
        verbose=True,
        log_interval=log_interval,
    )

    print("\nOptimisation finished.")


if __name__ == '__main__':
    main()
