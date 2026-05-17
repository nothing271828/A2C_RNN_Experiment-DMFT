# AGENTS.md — DMFT Solver for A2C RNN Experiment

## Purpose
This repo implements a **DMFT (Dynamical Mean-Field Theory) saddle-point solver** for an A2C RNN reinforcement-learning task. The theoretical framework is in `algorithm.tex`; the neural-network numerical experiment is in `main.py`; the DMFT equation solver is in `dmt_solver.py`.

## File map

| File | Role |
|------|------|
| `algorithm.tex` | Theory: DMFT action S, its decomposition, derivative formulas, and numerical algorithm |
| `main.py` | Numerical experiment: trains an A2C RNN via Adam or Langevin dynamics on a binary classification task (M=2 fixed trajectories, T=30) |
| `dmft_solver.py` | DMFT solver: computes S(C,Ĉ,y,ŷ,z,ẑ) and all gradients, with self-verification; includes `DMFTMinimaxOptimizer` with Adam-based minimax solver and checkpoint save/load |
| `compute.py` | Standalone compute entry-point: parameter setup, experiment-folder management, checkpoint auto-resume, SIGTERM handling, and auto-extend of `max_iter` on exhaustion |

**Critical invariant**: All files use the **same** hyperparameters, task, reward structure, and policy normalization (`layer_norm_softmax`, tau=1). When changing one, update the others.

## Commands

```bash
# Run the DMFT solver (includes self-verification of gradients)
python dmft_solver.py

# Run the DMFT saddle-point optimisation
python dmft_solver.py --optimize

# Standalone DMFT compute with experiment folder & checkpoint resume
python compute.py                          # uses default dir
python compute.py --dir results/my_exp     # specify experiment directory

# Run the neural network experiment (Langevin dynamics training)
python main.py
```

No package manager, no install step. Dependencies: `torch`, `numpy`, `matplotlib`.

## Numerical precision

**Float64 is mandatory for the DMFT solver.** S values are O(10⁵) while some gradients are O(10⁻⁶); float32 loses these to truncation error. The solver's `DMFTConfig` defaults to `dtype=torch.float64`.

## Gradient verification design

`dmft_solver.py` includes extensive self-tests that run automatically on `python dmft_solver.py`:

1. **Finite-difference check** for `S_quad + S_RL` (deterministic, no MC noise) — all autograd variables
2. **Self-consistency check**: `∂S/∂Ĉ = (Δt²/2)(C − ⟨φ·φᵀ⟩_W)` — validates the full MC sampling chain and -ln W gradient code (should give relative error < 1e-15 with float64)
3. **Directional steps**: S should decrease under minimax updates (minimize wrt C,y,z; maximize wrt Ĉ,ŷ,ẑ)

## DMFT variable conventions

- `K = M × T = 60` (flattened trajectory+time index)
- `C, Ĉ`: (K,K) symmetric matrices
- `y, ŷ`: (D_a, K) = (2, 60) — actor logits per action
- `z, ẑ`: (K,) — critic value
- Minimize S wrt `C, y, z`; maximize S wrt `Ĉ, ŷ, ẑ`

## Key implementation details

- **W gradients are manual**, using Price's theorem (response matrix R, batched eigendecomposition). Do NOT try to autograd through `torch.randn` for the MC samples — the formula accounts for the covariance dependence analytically.
- **`S_quad + S_RL` gradients are autograd** (PyTorch) — this is explicitly recommended in `algorithm.tex` §4.2.
- The response matrix `R` is precomputed once (block-diagonal lower-triangular, fixed by dt and T).
- `C^I` (input covariance) is computed from the **exact same input sequences** used in `main.py` (seed=12345).
- PsiNormalization uses `layer_norm_softmax` (not plain softmax) — this matters for translation/scale invariance of the policy.

## DMFT optimisation caveats

- **Warm-start is required.** Random initialization of `y` (giving π ≈ 0.5) lands in a shallow local minimum of the RL action where the exact-expectation loss increases with small policy improvements (a "barrier" effect). The `warm_start=True` option initializes `y` to strongly favor the correct action and `z` to the optimal value function. Without warm-start, accuracy degrades to ≤ 0.5 during optimisation.
- **`y, ŷ, z, ẑ` scale with β** at the saddle point (all O(β) or O(β/Δt)). The continuation approach (`β = 1 → 10 → 100 → 1000 → 10000`) with parameter rescaling between phases helps reach β=10000. Learning rates should decrease as β increases.
- **Memory management**: the `_compute_lnW_gradients` method processes MC samples in batches (BATCH=400) and must run inside `torch.no_grad()` to avoid leaking (Ns, K, K) autograd graphs. With `N_samples=2000` and float64, peak memory is ~200 MB. Never use `N_samples > 4000` without reducing BATCH.
- **Scheme recommendation**: use `'mixed'` (off‑diag for C↔Ĉ, diagonal for y↔ŷ and z↔ẑ). Pure `'off_diag'` fails for the actor‑critic pairs because ∂S/∂ŷ lacks the RL policy gradient.
- **Cholesky fallback**: C can lose positive-definiteness during optimisation. `_sample_eta` automatically falls back to eigendecomposition with eigenvalue clamping.

## Checkpoint & resume (compute.py)

`compute.py` manages a per‑experiment directory that stores all state:

```
results/<exp_name>/
├── config.json          # full hyperparameter record (human‑readable JSON)
├── ckpt_000050.pt       # periodic checkpoint
├── ckpt_000100.pt       # ...
├── ckpt_latest.pt       # always points to the latest state (fast resume)
└── ckpt_xxxxxx.pt       # final checkpoint on completion / interruption
```

**Saved in each checkpoint** (via `DMFTMinimaxOptimizer.save_checkpoint`):
- All 6 order parameters (`C, Ĉ, y, ŷ, z, ẑ`)
- Adam state (`m, v, t` per parameter — needed to exactly continue optimisation)
- Full iteration history (`self.history`)
- Internal `run()` state (`S_prev`, `max_gn_ema`, `n_stalled` — convergence tracking)

**Resume behaviour** (`compute.py` auto‑detects):
- Directory exists → loads `config.json` (verifies against current params, warns on mismatch) → loads `ckpt_latest.pt` → restores solver + optimizer state → continues from `iteration + 1`
- Directory does not exist → creates it, saves `config.json`, starts from scratch

**Auto‑extend on exhaustion**: If the resumed checkpoint has already completed `max_iter` iterations, `compute.py` automatically doubles `max_iter` (extends `completed + max_iter`) and updates `config.json`. This means a computation that hits its iteration limit will simply continue from where it left off on the next run — no manual intervention needed.

**Interrupt handling**: `compute.py` registers a SIGTERM handler that raises `KeyboardInterrupt`. The `run()` loop catches this and saves a checkpoint before exiting. `Ctrl+C` (SIGINT) also triggers the same path. Run again to resume.

**Checkpoint file size**: ~190 KB per checkpoint (6 params + 6 Adam states, float64, K=60). Periodic saves at `save_interval=50` are negligible in disk usage.
