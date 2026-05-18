"""
compare.py — DMFT vs Trained RNN Comparison

Loads DMFT checkpoint and a trained neural network from their respective
experiment directories, verifies hyperparameter / input consistency,
computes empirical correlation C, actor logits y, and critic value z
from the NN, then plots side‑by‑side comparisons.

Usage:
    python compare.py --dmft_dir results/experiment_1 --nn_dir results/nn_exp
"""

import os
import sys
import json
import glob

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import TrainableRNN, ActorCriticReadout, PsiNormalization, generate_trial


# ═══════════════════════════════════════════════════════════════
#  Apply PsiNormalization  (supports all methods, config‑driven)
# ═══════════════════════════════════════════════════════════════
def apply_psi(logits, psi_cfg):
    method = psi_cfg.get("method", "layer_norm_softmax")
    tau    = psi_cfg.get("tau", 1.0)
    eps    = psi_cfg.get("eps", 1e-8)
    alpha  = psi_cfg.get("alpha", 2.0)

    if method == 'power':
        m, _ = logits.min(dim=-1, keepdim=True)
        shifted = logits - m + eps
        powered = shifted.pow(alpha)
        return powered / powered.sum(dim=-1, keepdim=True)

    elif method == 'layer_norm_softmax':
        mean = logits.mean(dim=-1, keepdim=True)
        var  = ((logits - mean) ** 2).mean(dim=-1, keepdim=True)
        std  = torch.sqrt(var + eps)
        z    = (logits - mean) / std
        if tau != 1.0:
            z = z / tau
        return F.softmax(z, dim=-1)

    elif method == 'softmax':
        if tau != 1.0:
            logits = logits / tau
        return F.softmax(logits, dim=-1)
    return None


# ═══════════════════════════════════════════════════════════════
#  Config helpers
# ═══════════════════════════════════════════════════════════════
def load_json(path):
    with open(path) as f:
        return json.load(f)


def compare_configs(dmft_cfg, nn_cfg):
    """Check that network, task, and data_seed match."""
    sections = [
        ("network", ["N", "g", "dt", "D_in", "D_a"]),
        ("task",    ["T", "M", "sigma_noise", "gamma", "c_p", "c_v", "beta"]),
    ]
    ok = True
    for sec, keys in sections:
        for k in keys:
            v_dmft = dmft_cfg.get(sec, {}).get(k)
            v_nn   = nn_cfg.get(sec, {}).get(k)
            if v_dmft != v_nn:
                print(f"  ✗  {sec}.{k}:  DMFT={v_dmft}  NN={v_nn}")
                ok = False
    # data seed
    ds_dmft = dmft_cfg.get("random_seeds", {}).get("data_seed")
    ds_nn   = nn_cfg.get("random_seeds", {}).get("data_seed")
    if ds_dmft != ds_nn:
        print(f"  ✗  data_seed:  DMFT={ds_dmft}  NN={ds_nn}")
        ok = False
    # psi method
    pm_dmft = dmft_cfg.get("psi", {}).get("method", "layer_norm_softmax")
    pm_nn   = nn_cfg.get("psi", {}).get("method", "layer_norm_softmax")
    if pm_dmft != pm_nn:
        print(f"  ✗  psi.method:  DMFT={pm_dmft}  NN={pm_nn}")
        ok = False
    else:
        print(f"  ✓ Psi method: {pm_dmft}")
    if ok:
        print("  ✓ All critical hyperparameters match")
    return ok


# ═══════════════════════════════════════════════════════════════
#  Load DMFT checkpoint
# ═══════════════════════════════════════════════════════════════
def load_dmft_checkpoint(dmft_dir):
    latest = os.path.join(dmft_dir, "ckpt_latest.pt")
    if not os.path.isfile(latest):
        files = sorted(glob.glob(os.path.join(dmft_dir, "ckpt_*.pt")))
        if not files:
            raise FileNotFoundError(f"No checkpoint found in {dmft_dir}")
        latest = files[-1]
    print(f"  Loading DMFT checkpoint: {latest}")
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    C     = ckpt['params']['C'].to(torch.float64)
    y     = ckpt['params']['y'].to(torch.float64)
    z     = ckpt['params']['z'].to(torch.float64)
    it    = ckpt['iteration']
    acc   = ckpt['history'][-1].get('acc', None)
    print(f"    Iteration={it},  S={ckpt['history'][-1]['S']:.4e}"
          f"{', acc=' + f'{acc:.4f}' if acc is not None else ''}")
    return C, y, z


# ═══════════════════════════════════════════════════════════════
#  Load trained NN
# ═══════════════════════════════════════════════════════════════
def load_nn_model(nn_dir, nn_cfg, device='cpu'):
    nc = nn_cfg["network"]
    tc = nn_cfg["task"]
    pc = nn_cfg["psi"]

    rnn = TrainableRNN(nc["N"], nc["D_in"], g=nc["g"], dt=nc["dt"])
    ac  = ActorCriticReadout(nc["N"], tc["c_v"])
    psi = PsiNormalization(method=pc["method"], tau=pc["tau"])

    rnn_path = os.path.join(nn_dir, "rnn_state.pt")
    if not os.path.isfile(rnn_path):
        raise FileNotFoundError(f"RNN state not found: {rnn_path}")
    rnn.load_state_dict(torch.load(rnn_path, map_location=device, weights_only=True))
    rnn.to(device)

    ac_path = os.path.join(nn_dir, "ac_state.pt")
    if not os.path.isfile(ac_path):
        raise FileNotFoundError(f"AC state not found: {ac_path}")
    ac.load_state_dict(torch.load(ac_path, map_location=device, weights_only=True))
    ac.to(device)

    rnn.eval()
    ac.eval()

    results_path = os.path.join(nn_dir, "training_results.pt")
    final_acc = None
    if os.path.isfile(results_path):
        res = torch.load(results_path, map_location='cpu', weights_only=False)
        final_acc = res.get("final_test_acc", None)

    print(f"  Loaded NN model from {nn_dir}"
          + (f",  final test acc={final_acc:.4f}" if final_acc is not None else ""))
    return rnn, ac, psi


# ═══════════════════════════════════════════════════════════════
#  Generate dataset (same as main.py / dmft_solver.py)
# ═══════════════════════════════════════════════════════════════
def build_dataset(cfg_dict):
    tc = cfg_dict["task"]
    T, M, D_in, sigma = tc["T"], tc["M"], cfg_dict["network"]["D_in"], tc["sigma_noise"]
    data_seed = cfg_dict["random_seeds"]["data_seed"]
    torch.manual_seed(data_seed)
    signs   = [+1.0, -1.0]
    seqs    = [generate_trial(T, D_in, s, sigma) for s in signs]
    targets = [1, 0]
    return seqs, targets


# ═══════════════════════════════════════════════════════════════
#  Compute empirical observables from NN
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_empirical(rnn, ac, input_seqs, cfg_dict):
    """
    Run the trained RNN on each input trajectory and collect
    hidden states, activations, actor logits, and critic values.
    Returns C_emp (K,K), y_emp (2,K), z_emp (K,).
    """
    nc = cfg_dict["network"]
    tc = cfg_dict["task"]
    N = nc["N"]
    M = tc["M"]
    T = tc["T"]
    K = M * T
    c_v = tc["c_v"]
    device = next(rnn.parameters()).device

    x_all    = torch.zeros(K, N, dtype=torch.float64, device=device)
    phi_all  = torch.zeros(K, N, dtype=torch.float64, device=device)
    y_all    = torch.zeros(2, K, dtype=torch.float64, device=device)
    z_all    = torch.zeros(K, dtype=torch.float64, device=device)

    for mu in range(M):
        I_seq = input_seqs[mu].to(device).to(torch.float64)
        rnn.reset(batch_size=1)

        for n in range(T):
            alpha = mu * T + n
            x, phi_x = rnn.step(I_seq[n:n+1].to(rnn.J.dtype))
            x_all[alpha]    = x[0].to(torch.float64)
            phi_all[alpha]  = phi_x[0].to(torch.float64)

            logits, value = ac(x, phi_x)
            y_all[:, alpha] = logits[0].to(torch.float64)      # (2,)
            z_all[alpha]    = value[0].to(torch.float64)       # scalar

    C_emp = phi_all @ phi_all.T / N              # (K, K)

    return C_emp.cpu(), y_all.cpu(), z_all.cpu()


# ═══════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════
def make_comparison_figures(C_dmft, C_emp, y_dmft, y_emp, z_dmft, z_emp,
                             dmft_cfg, nn_cfg, dmft_dir, nn_dir, out_dir):
    M = nn_cfg["task"]["M"]
    T = nn_cfg["task"]["T"]
    K = M * T

    psi_cfg = dmft_cfg.get("psi", nn_cfg.get("psi", {"method": "layer_norm_softmax",
                                                        "tau": 1.0, "eps": 1e-8}))

    C_dmft_np = C_dmft.detach().cpu().numpy()
    C_emp_np = C_emp.detach().cpu().numpy()
    y_dmft_np = y_dmft.detach().cpu().numpy()
    y_emp_np = y_emp.detach().cpu().numpy()
    z_dmft_np = z_dmft.detach().cpu().numpy()
    z_emp_np = z_emp.detach().cpu().numpy()

    # --- compute π ---
    y_dmft_t = torch.from_numpy(y_dmft_np.T).double()   # (K, 2)
    y_emp_t  = torch.from_numpy(y_emp_np.T).double()
    pi_dmft  = apply_psi(y_dmft_t, psi_cfg).cpu().numpy()
    pi_emp   = apply_psi(y_emp_t, psi_cfg).cpu().numpy()

    dpi_dmft  = pi_dmft[:, 1]          # prob of action 1, (K,)
    dpi_emp   = pi_emp[:, 1]

    # Pearson correlations
    def pearson(a, b):
        a = a.flatten()
        b = b.flatten()
        am, bm = a - a.mean(), b - b.mean()
        return (am * bm).sum() / (np.sqrt((am**2).sum() * (bm**2).sum()) + 1e-30)

    corr_C  = pearson(C_dmft_np, C_emp_np)
    corr_y0 = pearson(y_dmft_np[0], y_emp_np[0])
    corr_y1 = pearson(y_dmft_np[1], y_emp_np[1])
    corr_z  = pearson(z_dmft_np, z_emp_np)
    corr_pi = pearson(dpi_dmft, dpi_emp)

    # ════════════════════════════════════
    #  Figure 1 — C comparison
    # ════════════════════════════════════
    vmin = min(C_dmft_np.min(), C_emp_np.min())
    vmax = max(C_dmft_np.max(), C_emp_np.max())

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    im0 = axes[0].imshow(C_dmft_np, aspect='auto', origin='lower',
                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('C — DMFT')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(C_emp_np, aspect='auto', origin='lower',
                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('C — NN empirical')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    Cd = C_dmft_np - C_emp_np
    vd = max(abs(Cd.min()), abs(Cd.max()))
    im2 = axes[2].imshow(Cd, aspect='auto', origin='lower',
                          cmap='seismic', vmin=-vd, vmax=vd)
    axes[2].set_title('C — difference (DMFT−NN)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].scatter(C_dmft_np.flatten(), C_emp_np.flatten(), s=1, alpha=0.3)
    axes[3].plot([vmin, vmax], [vmin, vmax], 'k--', lw=0.8)
    axes[3].set_xlabel('DMFT')
    axes[3].set_ylabel('NN')
    axes[3].set_title(f'C scatter  (r={corr_C:.4f})')

    fig.suptitle(f'Correlation C  (K={K})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_C.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════
    #  Figure 2 — y comparison
    # ════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    for i, ax_row in enumerate([0, 1]):
        axes[ax_row][0].plot(y_dmft_np[i], 'b-', lw=1, label='DMFT')
        axes[ax_row][0].plot(y_emp_np[i], 'r--', lw=1, label='NN')
        axes[ax_row][0].set_title(f'y action {i}  (r={corr_y0 if i==0 else corr_y1:.4f})')
        axes[ax_row][0].set_xlabel('flattened index α=(μ,n)')
        axes[ax_row][0].legend()

        axes[ax_row][1].scatter(y_dmft_np[i], y_emp_np[i], s=8, alpha=0.5)
        axes[ax_row][1].set_xlabel('DMFT')
        axes[ax_row][1].set_ylabel('NN')
        axes[ax_row][1].set_title(f'y action {i} scatter')

    fig.suptitle('Actor logits y', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_y.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════
    #  Figure 3 — z and π comparison
    # ════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    axes[0][0].plot(z_dmft_np, 'b-', lw=1, label='DMFT')
    axes[0][0].plot(z_emp_np, 'r--', lw=1, label='NN')
    axes[0][0].set_title(f'Critic value z  (r={corr_z:.4f})')
    axes[0][0].set_xlabel('flattened index α=(μ,n)')
    axes[0][0].legend()

    axes[0][1].scatter(z_dmft_np, z_emp_np, s=8, alpha=0.5)
    axes[0][1].set_xlabel('DMFT')
    axes[0][1].set_ylabel('NN')
    axes[0][1].set_title('z scatter')

    axes[1][0].plot(dpi_dmft, 'b-', lw=1, label='DMFT π(action=1)')
    axes[1][0].plot(dpi_emp, 'r--', lw=1, label='NN π(action=1)')
    axes[1][0].set_title(f'Policy prob π(action=1)  (r={corr_pi:.4f})')
    axes[1][0].set_xlabel('flattened index α=(μ,n)')
    axes[1][0].legend()

    axes[1][1].scatter(dpi_dmft, dpi_emp, s=8, alpha=0.5)
    axes[1][1].set_xlabel('DMFT')
    axes[1][1].set_ylabel('NN')
    axes[1][1].set_title('π(action=1) scatter')
    axes[1][1].plot([0, 1], [0, 1], 'k--', lw=0.8)

    fig.suptitle('Critic z  &  Policy π', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_z_pi.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════
    #  Summary
    # ════════════════════════════════════
    print(f"\n  Pearson correlations (DMFT vs NN):")
    print(f"    C:    {corr_C:+.4f}")
    print(f"    y[0]: {corr_y0:+.4f}")
    print(f"    y[1]: {corr_y1:+.4f}")
    print(f"    z:    {corr_z:+.4f}")
    print(f"    π:    {corr_pi:+.4f}")

    # save correlations
    corr_path = os.path.join(out_dir, "correlations.json")
    with open(corr_path, 'w') as f:
        json.dump({
            "C":   float(corr_C),
            "y_0": float(corr_y0),
            "y_1": float(corr_y1),
            "z":   float(corr_z),
            "pi":  float(corr_pi),
        }, f, indent=2)
    print(f"  Saved correlations to {corr_path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    dmft_dir = "results/experiment_2"
    nn_dir   = "results/experiment_2_nn"
    # for i, arg in enumerate(sys.argv[1:]):
    #     if arg == "--dmft_dir" and i + 2 < len(sys.argv):
    #         dmft_dir = sys.argv[i + 2]
    #     elif arg.startswith("--dmft_dir="):
    #         dmft_dir = arg.split("=", 1)[1]
    #     elif arg == "--nn_dir" and i + 2 < len(sys.argv):
    #         nn_dir = sys.argv[i + 2]
    #     elif arg.startswith("--nn_dir="):
    #         nn_dir = arg.split("=", 1)[1]

    if dmft_dir is None or nn_dir is None:
        print("Usage: python compare.py --dmft_dir <dir> --nn_dir <dir>")
        sys.exit(1)

    print("=" * 60)
    print("  DMFT vs NN — Comparison")
    print(f"  DMFT dir: {dmft_dir}")
    print(f"  NN   dir: {nn_dir}")
    print("=" * 60)

    # --- Load configs ---
    dmft_cfg = load_json(os.path.join(dmft_dir, "config.json"))
    nn_cfg   = load_json(os.path.join(nn_dir, "config.json"))
    print("\n  DMFT config keys:", sorted(dmft_cfg.keys()))
    print("  NN   config keys:", sorted(nn_cfg.keys()))

    # --- Verify consistency ---
    print("\n--- Hyperparameter consistency ---")
    compare_configs(dmft_cfg, nn_cfg)

    # --- Verify input sequences ---
    seqs_dmft, _ = build_dataset(dmft_cfg)
    seqs_nn,   _ = build_dataset(nn_cfg)
    mismatch = False
    for mu in range(2):
        diff = (seqs_dmft[mu].double() - seqs_nn[mu].double()).abs().max().item()
        if diff > 1e-12:
            print(f"  ✗  Input seq {mu}: max diff = {diff:.2e}")
            mismatch = True
    if not mismatch:
        print("  ✓ Input sequences are identical")

    # --- Load DMFT ---
    print("\n--- Loading DMFT checkpoint ---")
    C_dmft, y_dmft, z_dmft = load_dmft_checkpoint(dmft_dir)

    # --- Load NN ---
    print("\n--- Loading NN model ---")
    device = 'cpu'
    rnn, ac, psi = load_nn_model(nn_dir, nn_cfg, device=device)

    # --- Compute empirical observables ---
    print("\n--- Computing empirical C, y, z from NN ---")
    C_emp, y_emp, z_emp = compute_empirical(rnn, ac, seqs_nn, nn_cfg)
    print(f"  C_emp  shape: {list(C_emp.shape)},  range: [{C_emp.min():.4f}, {C_emp.max():.4f}]")
    print(f"  y_emp  shape: {list(y_emp.shape)},  range: [{y_emp.min():.4f}, {y_emp.max():.4f}]")
    print(f"  z_emp  shape: {list(z_emp.shape)},  range: [{z_emp.min():.4f}, {z_emp.max():.4f}]")

    # --- Plot ---
    out_dir = os.path.join(nn_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Generating comparison figures → {out_dir} ---")
    make_comparison_figures(C_dmft, C_emp, y_dmft, y_emp, z_dmft, z_emp,
                             dmft_cfg, nn_cfg, dmft_dir, nn_dir, out_dir)
    print(f"\n  Comparison figures saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
