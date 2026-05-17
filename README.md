# DMFT Solver for A2C RNN Experiment

DMFT (Dynamical Mean-Field Theory) saddle-point solver for an Advantage Actor-Critic (A2C) recurrent neural network performing binary classification on M=2 fixed trajectories of length T=30.

## Project Structure

| File | Description |
|------|-------------|
| `algorithm.tex` | Theory document: DMFT action S, its decomposition, derivative formulas, and numerical algorithm |
| `main.py` | Numerical experiment: trains an A2C RNN via Adam or Langevin dynamics on the same task |
| `dmft_solver.py` | DMFT solver core: computes S(C,Ĉ,y,ŷ,z,ẑ) and all gradients, with self-verification and an Adam-based minimax optimizer |
| `compute.py` | Standalone compute script: sets up parameters and calls `DMFTMinimaxOptimizer` to solve the saddle point |
| `AGENTS.md` | Developer notes: file map, commands, precision requirements, and optimisation caveats |

## Dependencies

```
torch, numpy, matplotlib
```

## Usage

```bash
# Run the DMFT solver with gradient verification (self-tests)
python dmft_solver.py

# Run the DMFT saddle-point optimisation (built-in)
python dmft_solver.py --optimize

# Standalone DMFT compute with checkpoint & resume support
python compute.py                          # uses default experiment dir
python compute.py --dir results/my_exp     # specify experiment directory

# Run the neural network experiment (Langevin dynamics training)
python main.py
```

## Checkpoint & Resume

Each experiment corresponds to a dedicated directory.  `compute.py` auto‑detects whether the directory exists:

- **New experiment**: creates the directory, saves a full `config.json` with all hyperparameters, then starts the minimax optimisation.
- **Resume**: loads `config.json` (verifies against current parameters), restores the latest checkpoint (`ckpt_latest.pt`), and continues from the last saved iteration.

Checkpoints are saved:
- **Periodically**: every `save_interval` iterations (default 50)
- **On interruption**: if `Ctrl+C` or `SIGTERM` is received, the current state is saved before exit
- **On completion**: final checkpoint at the last iteration

Directory structure:
```
results/my_exp/
├── config.json          # full hyperparameter record (human‑readable)
├── ckpt_000050.pt       # periodic checkpoint at iter 50
├── ckpt_000100.pt       # periodic checkpoint at iter 100
├── ckpt_latest.pt       # always points to the latest state
└── ckpt_final.pt        # final state (if completed, currently omitted; see ckpt_latest.pt)
```

## Key Hyperparameters

All three source files share the same hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 100 | Hidden size |
| g | 1.5 | Recurrent gain |
| dt | 0.1 | Time step |
| D_in | 10 | Input dimension |
| D_a | 2 | Number of actions |
| T | 30 | Steps per trajectory |
| M | 2 | Number of trajectories |
| sigma_noise | 2.0 | Input noise scale |
| gamma | 0.95 | Discount factor |
| c_p | 1.0 | Policy loss weight |
| c_v | 1.0 | Value loss weight |
| beta | 10000 | Inverse temperature |

## DMFT Variable Conventions

- `K = M × T = 60` — flattened trajectory+time index
- `C, Ĉ`: (K, K) symmetric matrices
- `y, ŷ`: (D_a, K) = (2, 60) — actor logits per action
- `z, ẑ`: (K,) — critic value
- **Minimize** S wrt `C, y, z`; **maximize** S wrt `Ĉ, ŷ, ẑ`

## Numerical Precision

**Float64 is mandatory** for the DMFT solver. S values are O(10⁵) while some gradients are O(10⁻⁶); float32 loses these to truncation error.
