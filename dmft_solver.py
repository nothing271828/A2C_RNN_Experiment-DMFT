"""
DMFT Solver for A2C RNN — computes the action S and its derivatives
w.r.t. all DMFT order parameters (C, C_hat, y, y_hat, z, z_hat).

The design follows algorithm.tex exactly, matching the experimental
setup in main.py (same environment, task, reward structure, and
PsiNormalization).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Configuration — same hyperparameters as main.py
# ============================================================
class DMFTConfig:
    def __init__(self, **kwargs):
        # Network
        self.N          = kwargs.get('N', 100)
        self.g          = kwargs.get('g', 1.5)
        self.dt         = kwargs.get('dt', 0.1)
        self.D_in       = kwargs.get('D_in', 10)
        self.D_a        = kwargs.get('D_a', 2)

        # Task
        self.T           = kwargs.get('T', 30)
        self.M           = kwargs.get('M', 2)
        self.sigma_noise = kwargs.get('sigma_noise', 2.0)
        self.gamma       = kwargs.get('gamma', 0.95)
        self.c_p         = kwargs.get('c_p', 1.0)
        self.c_v         = kwargs.get('c_v', 1.0)
        self.beta        = kwargs.get('beta', 10000.0)

        # DMFT numerics
        self.N_samples = kwargs.get('N_samples', 5000)
        self.sigma_U   = kwargs.get('sigma_U', 1.0)
        self.eps_reg   = kwargs.get('eps_reg', 1e-6)
        self.dtype     = kwargs.get('dtype', torch.float64)

        # Derived
        self.K = self.M * self.T


# ============================================================
# PsiNormalization — exactly as in main.py (layer_norm_softmax)
# ============================================================
class PsiNormalization(nn.Module):
    def __init__(self, eps=1e-8, tau=1.0):
        super().__init__()
        self.eps = eps
        self.tau = tau

    def forward(self, logits):
        mean = logits.mean(dim=-1, keepdim=True)
        var  = ((logits - mean) ** 2).mean(dim=-1, keepdim=True)
        std  = torch.sqrt(var + self.eps)
        z    = (logits - mean) / std
        if self.tau != 1.0:
            z = z / self.tau
        return F.softmax(z, dim=-1)


# ============================================================
# Data generation — same as main.py  (seed 12345)
# ============================================================
def generate_trial(T, D_in, mean_sign, sigma_noise):
    return mean_sign + sigma_noise * torch.randn(T, D_in)


def generate_dataset(cfg, seed=12345):
    torch.manual_seed(seed)
    M, T, D_in, sigma = cfg.M, cfg.T, cfg.D_in, cfg.sigma_noise
    signs   = [+1.0, -1.0]
    seqs    = [generate_trial(T, D_in, s, sigma) for s in signs]
    targets = [1, 0]                         # action 1 → +, 0 → −
    return seqs, targets


# ============================================================
# DMFT Solver
# ============================================================
class DMFTSolver(nn.Module):

    def __init__(self, cfg, input_seqs, targets):
        super().__init__()
        self.cfg = cfg
        self.K   = cfg.K
        self.M   = cfg.M
        self.Tp  = cfg.T                # time steps per trajectory
        self.Da  = cfg.D_a

        self.input_seqs = input_seqs
        self.targets    = targets

        # ---- precompute structural matrices ----
        self._precompute_input_covariance()
        self._precompute_response_matrix()

        # ---- order parameters (nn.Parameter → autograd) ----
        self._init_parameters()

        # ---- Psi normalisation (same as experiment) ----
        self.psi = PsiNormalization(eps=1e-8, tau=1.0)

    # ---------------------------------------------------------
    #  Precomputation helpers
    # ---------------------------------------------------------
    def _precompute_input_covariance(self):
        """C^I ∈ R^{K×K} :  input-induced noise covariance."""
        sigma_U = self.cfg.sigma_U
        S_flat  = torch.cat(self.input_seqs, dim=0).to(self.cfg.dtype)
        self.CI = sigma_U * sigma_U * (S_flat @ S_flat.T)

    def _precompute_response_matrix(self):
        """R ∈ R^{K×K} : block‑diagonal response matrix."""
        dt = self.cfg.dt
        T  = self.Tp
        M  = self.M
        K  = self.K
        dtype = self.cfg.dtype

        R_single = torch.zeros(T, T, dtype=dtype)
        for n in range(T):
            for m in range(n):
                R_single[n, m] = dt * (1.0 - dt) ** (n - 1 - m)

        R = torch.zeros(K, K, dtype=dtype)
        for mu in range(M):
            s = mu * T
            e = s + T
            R[s:e, s:e] = R_single

        self.R = R

    # ---------------------------------------------------------
    #  Parameters
    # ---------------------------------------------------------
    def _init_parameters(self):
        K   = self.K
        Da  = self.Da
        dt  = self.cfg.dtype

        # C   — minimise    (start as small identity for PSD)
        self.C = nn.Parameter(torch.eye(K, dtype=dt) * 0.1)

        # C_hat — maximise
        self.C_hat = nn.Parameter(torch.zeros(K, K, dtype=dt))

        # y   — minimise   (actor logits,  Da × K)
        self.y = nn.Parameter(0.01 * torch.randn(Da, K, dtype=dt))

        # y_hat — maximise
        self.y_hat = nn.Parameter(0.01 * torch.randn(Da, K, dtype=dt))

        # z   — minimise   (critic value)
        self.z = nn.Parameter(torch.zeros(K, dtype=dt))

        # z_hat — maximise
        self.z_hat = nn.Parameter(torch.zeros(K, dtype=dt))

    # =========================================================
    #  MC sampling  &  forward dynamics for W
    # =========================================================
    def _build_covariance(self):
        eps = self.cfg.eps_reg
        g   = self.cfg.g
        S   = g * g * self.C + self.CI
        S   = S + eps * torch.eye(self.K, dtype=S.dtype, device=S.device)
        return S

    def _sample_eta(self, Ns=None):
        if Ns is None:
            Ns = self.cfg.N_samples
        Sigma = self._build_covariance()
        L = torch.linalg.cholesky(Sigma)
        return torch.randn(Ns, self.K, dtype=Sigma.dtype, device=Sigma.device) @ L.T

    def _forward_dynamics(self, eta):
        """
        h  = R · η    (initial condition h_0 = 0)
        φ  = tanh(h)
        """
        h   = eta @ self.R.T
        phi = torch.tanh(h)
        return h, phi

    def _compute_O(self, h, phi):
        """
        O = (Δt²/2)  [ Σ_i (ŷ_i·h)²  +  φᵀ Ĉ φ ]
        """
        dt = self.cfg.dt
        yh = self.y_hat @ h.T                          # (Da, Ns)
        t1 = (yh ** 2).sum(dim=0)                      # (Ns,)
        t2 = torch.einsum('si,ij,sj->s', phi, self.C_hat, phi)
        return 0.5 * dt * dt * (t1 + t2)

    # =========================================================
    #  Manual gradients of  -ln W
    # =========================================================
    def _compute_lnW_gradients(self, eta, h, phi, O):
        dt  = self.cfg.dt
        g   = self.cfg.g
        Da  = self.Da
        K   = self.K
        Ns  = O.shape[0]

        # ----- softmax weights (log‑sum‑exp) -----
        O_max       = O.max()
        O_shifted   = O - O_max
        log_sum_exp = O_max + torch.log(torch.sum(torch.exp(O_shifted)) + 1e-30)
        neg_ln_W    = math.log(Ns) - log_sum_exp

        w = torch.softmax(O, dim=0)                     # (Ns,)

        # ----- grad_C_hat  = -(Δt²/2) ⟨φ φᵀ⟩_W -----
        grad_C_hat = -0.5 * dt * dt * torch.einsum('s,si,sj->ij', w, phi, phi)

        # ----- grad_y_hat  = -Δt² ⟨(ŷ_i·h) h⟩_W -----
        yh = self.y_hat @ h.T                           # (Da, Ns)
        grad_y_hat = torch.zeros(Da, K, dtype=self.y_hat.dtype)
        for i in range(Da):
            grad_y_hat[i] = -dt * dt * (h * (w * yh[i]).unsqueeze(-1)).sum(dim=0)

        # ----- grad_C  (core — uses response matrix R) -----
        phi_prime  = 1.0 - phi ** 2
        phi_dprime = -2.0 * phi * phi_prime

        C_hat_phi = (self.C_hat @ phi.T).T               # (Ns, K)

        # u = Δt² [ Σ_i ŷ_i (ŷ_i·h) + φ′ ⊙ (Ĉ φ) ]
        y_term = (self.y_hat.T @ yh).T                    # (Ns, K)
        u = dt * dt * (y_term + phi_prime * C_hat_phi)   # (Ns, K)

        # H = Δt² [ Σ_i ŷ_i ŷ_iᵀ  +  diag(φ′) Ĉ diag(φ′)  +  diag(φ″ ⊙ (Ĉ φ)) ]
        H0 = dt * dt * (self.y_hat.T @ self.y_hat)        # (K, K)

        H1 = dt * dt * (phi_prime.unsqueeze(-1) *
                        self.C_hat.unsqueeze(0) *
                        phi_prime.unsqueeze(1))           # (Ns, K, K)

        d_s  = phi_dprime * C_hat_phi
        H2   = dt * dt * torch.diag_embed(d_s)             # (Ns, K, K)

        H_s  = H0.unsqueeze(0) + H1 + H2

        # Rᵀ H_s R   (batched matmul)
        R3   = self.R.unsqueeze(0).expand(Ns, -1, -1)
        RT3  = self.R.T.unsqueeze(0).expand(Ns, -1, -1)
        step1 = torch.bmm(H_s, R3)                         # H_s @ R
        RTHR  = torch.bmm(RT3, step1)                      # Rᵀ @ (H_s @ R)

        # (Rᵀ u_s)(Rᵀ u_s)ᵀ
        RTu       = u @ self.R
        RTu_outer = torch.bmm(RTu.unsqueeze(-1), RTu.unsqueeze(1))

        per_sample = RTHR + RTu_outer
        weighted   = torch.einsum('s,sij->ij', w, per_sample)
        grad_C     = -0.5 * g * g * weighted

        return neg_ln_W, grad_C, grad_C_hat, grad_y_hat

    # =========================================================
    #  S_quad + S_RL   (autograd)
    # =========================================================
    def _compute_S_quad_RL(self):
        dt   = self.cfg.dt
        cv   = self.cfg.c_v
        cp   = self.cfg.c_p
        beta = self.cfg.beta
        gamma= self.cfg.gamma
        M    = self.M
        T    = self.Tp
        K    = self.K

        # --- S_quad ---
        S_quad = (0.5 * dt * dt * torch.sum(self.C_hat * self.C)
                  + dt * torch.sum(self.y_hat * self.y)
                  + dt * torch.dot(self.z_hat, self.z)
                  - 0.5 * dt * dt / (cv * cv) * (self.z_hat @ self.C @ self.z_hat))

        # --- S_RL ---
        # y: (Da, K) → (K, Da)   for psi
        logits    = self.y.T
        probs     = self.psi(logits)
        log_probs = torch.log(probs + 1e-12)

        # z_next  (shift within each trajectory; 0 after last step)
        z_resh = self.z.view(M, T)
        z_next = torch.zeros(M, T, dtype=self.z.dtype, device=self.z.device)
        z_next[:, :-1] = z_resh[:, 1:]
        z_next = z_next.reshape(K)

        S_RL = torch.tensor(0.0, dtype=self.z.dtype, device=self.z.device)

        for mu in range(M):
            tgt = self.targets[mu]
            for n in range(T):
                a_idx = mu * T + n
                p  = probs[a_idx]
                lp = log_probs[a_idx]
                zc = self.z[a_idx]
                zn = z_next[a_idx]

                for a in range(self.Da):
                    r = 1.0 if a == tgt else 0.0
                    A = r + gamma * zn - zc
                    S_RL = S_RL + beta * p[a] * (
                        -cp * lp[a] * A
                        + 0.5 * cv * cv * A * A
                    )

        return S_quad + S_RL

    # =========================================================
    #  Top‑level:  compute S_total and all gradients
    # =========================================================
    def zero_all_grads(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def compute_S_and_gradients(self, N_samples=None):
        """One forward pass → returns S_total (float), grads (dict)."""
        self.zero_all_grads()

        # 1. MC samples for W
        eta = self._sample_eta(N_samples)
        h, phi = self._forward_dynamics(eta)
        O = self._compute_O(h, phi)

        # 2. Manual −ln W gradients
        neg_ln_W, gC_W, gCh_W, gyh_W = self._compute_lnW_gradients(eta, h, phi, O)

        # 3. Autograd:  S_quad + S_RL
        S_qr = self._compute_S_quad_RL()
        S_qr.backward()

        # 4. Combine  (autograd already set .grad for all params)
        def _add(p, m):
            if p.grad is not None:
                p.grad = p.grad + m
            else:
                p.grad = m

        _add(self.C,      gC_W)
        _add(self.C_hat,  gCh_W)
        _add(self.y_hat,  gyh_W)

        S_total = (neg_ln_W + S_qr.detach()).item()

        grads = {}
        for n in ['C','C_hat','y','y_hat','z','z_hat']:
            p = getattr(self, n)
            grads[n] = p.grad.detach().clone() if p.grad is not None else None

        return S_total, grads

    def evaluate_S_full_at_eta(self, eta):
        """Compute S for a *given* set of noise samples  (no gradient)."""
        h, phi = self._forward_dynamics(eta)
        O = self._compute_O(h, phi)
        Ns = O.shape[0]

        O_max = O.max()
        O_shifted = O - O_max
        log_sum_exp = O_max + torch.log(torch.sum(torch.exp(O_shifted)) + 1e-30)
        neg_ln_W = math.log(Ns) - log_sum_exp

        S_qr = self._compute_S_quad_RL().detach()
        return (neg_ln_W + S_qr).item()

    # =========================================================
    #  Gradient verification
    # =========================================================
    def _perturb_param(self, name, delta):
        """Temporarily add delta to param, return context manager."""
        p = getattr(self, name)
        p.data += delta
        return p

    def verify_S_quad_RL_gradients(self, eps=1e-5):
        """
        Verify autograd gradients for S_quad + S_RL only (no MC noise).
        Compares autograd ∂(S_quad+S_RL)/∂X  with finite‑difference.
        """
        print("\n" + "="*58)
        print("  Verification: S_quad + S_RL  (autograd vs finite‑diff)")
        print("  eps =", eps)
        print("="*58)

        self.zero_all_grads()
        S0 = self._compute_S_quad_RL()
        S0.backward()

        autograd_grads = {}
        for n in ['C','C_hat','y','y_hat','z','z_hat']:
            p = getattr(self, n)
            autograd_grads[n] = p.grad.detach().clone() if p.grad is not None else None

        S0_val = S0.item()

        for name in ['C','C_hat','y','y_hat','z','z_hat']:
            g = autograd_grads[name]
            if g is None:
                print(f"  {name:>6s}: autograd grad is None")
                continue

            gn = g.norm()
            if gn < 1e-30:
                print(f"  {name:>6s}: |g| ≈ 0  (gradient is zero)")
                continue

            dirn = g / gn

            # S(+ε dir)    — must re‑build graph each time
            p = getattr(self, name)
            orig = p.data.clone()
            p.data = orig + eps * dirn
            Sp = self._compute_S_quad_RL().item()
            p.data = orig - eps * dirn
            Sm = self._compute_S_quad_RL().item()
            p.data = orig

            fd_deriv = (Sp - Sm) / (2.0 * eps)
            an_deriv = (dirn * g).sum().item()
            ratio = fd_deriv / an_deriv if abs(an_deriv) > 1e-30 else 0.0

            # also 2nd‑order check:  |Sp − S0 − ε·an|  should be O(ε²)
            first_order_err = Sp - S0_val - eps * an_deriv

            print(f"  {name:>6s}: |g|={gn:.2e}  "
                  f"FD={fd_deriv:+.4e}  an={an_deriv:+.4e}  "
                  f"ratio={ratio:.6f}  "
                  f"O(ε²)err={first_order_err:+.2e}")

        self.zero_all_grads()

    def verify_y_gradient_basic(self, eps=1e-6):
        """Isolated check of y‑gradient in S_RL (bypass W noise)."""
        print("\n--- Isolated y‑gradient check (S_RL only, no W) ---")

        self.zero_all_grads()
        S = self._compute_S_quad_RL()
        S.backward()
        g = self.y.grad.detach().clone()
        gn = g.norm()
        dirn = g / gn

        orig = self.y.data.clone()

        # finite diff
        for sgn, label in [(+1, '+ε'), (-1, '-ε')]:
            self.y.data = orig + sgn * eps * dirn
            self.zero_all_grads()
            S2 = self._compute_S_quad_RL()
            print(f"  S_quad_RL(y{label}·dir) = {S2.item():.12e}")

        # central diff
        self.y.data = orig + eps * dirn
        Sp = self._compute_S_quad_RL().item()
        self.y.data = orig - eps * dirn
        Sm = self._compute_S_quad_RL().item()
        self.y.data = orig

        fd = (Sp - Sm) / (2.0 * eps)
        an = (dirn * g).sum().item()
        print(f"  FD = {fd:.8e},  analytic = {an:.8e},  ratio = {fd/an:.6f}")
        print(f"  S(y) = {S.item():.8e}")

    def verify_full_S_directions(self):
        """
        Check that S changes in the correct direction when stepping.
        We do NOT use fixed-eta finite‑diff for C (because C changes η
        distribution).  Instead we verify:
          - Minimise vars (C,y,z):  S(x − α·g)  <  S(x)  <  S(x + α·g)
          - Maximise vars (Ĉ,ŷ,ẑ):  S(x + α·g)  >  S(x)  >  S(x − α·g)
        using self‑consistent MC evaluations at each point.
        """
        print("\n" + "="*58)
        print("  Verification: full‑S directional check")
        print("  (Each point evaluated with independent MC sampling)")
        print("="*58)

        step_scales = {
            'C':      1e-4,
            'C_hat':  1e-4,
            'y':      1e-6,
            'y_hat':  1e-4,
            'z':      1e-6,
            'z_hat':  1e-4,
        }

        minimise_vars = {'C', 'y', 'z'}
        maximise_vars = {'C_hat', 'y_hat', 'z_hat'}

        for name in ['C','C_hat','y','y_hat','z','z_hat']:
            step = step_scales[name]

            S0, grads = self.compute_S_and_gradients()
            g = grads[name]
            if g is None:
                print(f"  {name:>6s}: grad is None")
                continue

            gn = g.norm()
            if gn < 1e-30:
                print(f"  {name:>6s}: |g| ≈ 0, SKIP")
                continue

            dirn = g / gn

            p = getattr(self, name)
            orig = p.data.clone()

            # positive direction
            p.data = orig + step * dirn
            Sp, _ = self.compute_S_and_gradients()

            # negative direction
            p.data = orig - step * dirn
            Sm, _ = self.compute_S_and_gradients()

            p.data = orig

            if name in minimise_vars:
                # want  Sm < S0 < Sp   (minimise → move −g)
                # moving −g  REDUCES S;  moving +g INCREASES S
                ok = (Sm < S0) and (S0 < Sp)
            else:
                # want  Sp > S0 > Sm   (maximise → move +g)
                ok = (Sp > S0) and (S0 > Sm)

            print(f"  {name:>6s} "
                  f"S(−g)={Sm:.6e}  S(0)={S0:.6e}  S(+g)={Sp:.6e}  "
                  f"{'✓' if ok else '✗'}   |g|={gn:.4e}")

        # restore everything
        S0, _ = self.compute_S_and_gradients()


# ============================================================
#  Main:   verification
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("  DMFT Solver — Gradient Verification")
    print("="*60)

    cfg = DMFTConfig(N_samples=4000, dtype=torch.float64)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    input_seqs, targets = generate_dataset(cfg, seed=12345)

    solver = DMFTSolver(cfg, input_seqs, targets)

    print(f"\n  K = {cfg.K}  (M={cfg.M}, T={cfg.T})")
    print(f"  Hyper‑params:  g={cfg.g}, dt={cfg.dt}, γ={cfg.gamma}, "
          f"c_p={cfg.c_p}, c_v={cfg.c_v}, β={cfg.beta}")
    print(f"  MC samples = {cfg.N_samples}")

    # ---- 1. Evaluate S ----
    print("\n--- (a)  S evaluation ---")
    S0, grads = solver.compute_S_and_gradients()
    print(f"  S_total = {S0:.8e}")
    for n, g in grads.items():
        gn = g.norm().item() if g is not None else 0.0
        print(f"    |grad_{n}| = {gn:.4e}")

    # ---- 2. Verify S_quad + S_RL (deterministic, no noise) ----
    solver.verify_S_quad_RL_gradients(eps=1e-5)

    # ---- 3. Verify S_RL y‑gradient at finer resolution ----
    solver.verify_y_gradient_basic(eps=1e-7)

    # ---- 4. Full‑S directional check ----
    solver.verify_full_S_directions()

    # ---- 5. Self‑consistency: ∂S/∂Ĉ ?= (Δt²/2)(C − ⟨φ·φᵀ⟩_W) ----
    print("\n" + "="*58)
    print("  Self‑consistency check:  ∂S/∂Ĉ  vs  (Δt²/2) (C − ⟨φ·φᵀ⟩_W)")
    print("="*58)

    # re‑run MC at current parameters with a fixed seed  (no‑grad)
    with torch.no_grad():
        torch.manual_seed(777)
        eta_test = solver._sample_eta(cfg.N_samples)
        h_test, phi_test = solver._forward_dynamics(eta_test)
        O_test = solver._compute_O(h_test, phi_test)

        Ns = O_test.shape[0]
        w_test = torch.softmax(O_test, dim=0)
        phi_corr_W = torch.einsum('s,si,sj->ij', w_test, phi_test, phi_test)  # (K,K)

        # manual W gradients (these are fine inside no_grad)
        _, gC_W, Ch_W, yh_W = solver._compute_lnW_gradients(eta_test, h_test, phi_test, O_test)

    # Now compute autograd part (outside no_grad)
    solver.zero_all_grads()
    S_qr = solver._compute_S_quad_RL()
    S_qr.backward()
    # total ∂S/∂Ĉ = autograd ∂S_qr/∂Ĉ  +  manual ∂(−ln W)/∂Ĉ
    gCh_total = solver.C_hat.grad.detach().clone() + Ch_W

    dt = cfg.dt
    # Prediction:  ∂S/∂Ĉ = (Δt²/2) C − (Δt²/2) ⟨φ·φᵀ⟩_W
    pred_gCh = 0.5 * dt * dt * (solver.C.data - phi_corr_W)
    diff_norm = (gCh_total - pred_gCh).norm()
    pred_norm = pred_gCh.norm()
    rel_err = (diff_norm / (pred_norm + 1e-30)).item()

    print(f"  ‖∂S/∂Ĉ (computed) − pred‖ / ‖pred‖ = {rel_err:.2e}")
    print(f"  ‖∂S/∂Ĉ (computed)‖ = {gCh_total.norm().item():.6e}")
    print(f"  ‖∂S/∂Ĉ (predicted)‖ = {pred_gCh.norm().item():.6e}")
    print(f"  ‖(Δt²/2) C‖          = {0.5*dt*dt*solver.C.data.norm().item():.6e}")
    print(f"  ‖(Δt²/2) ⟨φ·φᵀ⟩_W‖  = {0.5*dt*dt*phi_corr_W.norm().item():.6e}")
    if rel_err < 1e-4:
        print(f"  ✓  Self‑consistency PASSED  (rel err = {rel_err:.2e})")
    else:
        print(f"  ✗  Self‑consistency FAILED  (rel err = {rel_err:.2e})")

    solver.zero_all_grads()

    # ---- 6. DMFT gradient step: check S changes correctly ----
    print("\n" + "="*58)
    print("  DMFT gradient‑step test (few iterations)")
    print("="*58)
    # Run a few update steps with small learning rates
    lr_min  = {'C': 5e-3,  'y': 5e-7,  'z': 5e-6}
    lr_max  = {'C_hat': 5e-3, 'y_hat': 5e-4, 'z_hat': 5e-4}

    S_prev = solver.compute_S_and_gradients()[0]
    print(f"  Initial S = {S_prev:.8e}")

    for step in range(5):
        S_val, grads = solver.compute_S_and_gradients()

        with torch.no_grad():
            for name in ['C','C_hat','y','y_hat','z','z_hat']:
                g = grads[name]
                if g is None:
                    continue
                p = getattr(solver, name)
                if name in lr_min:
                    p.data -= lr_min[name] * g      # minimise
                else:
                    p.data += lr_max[name] * g       # maximise

        print(f"  Step {step+1}: S = {S_val:.8e}")

    print(f"  ΔS after {step+1} steps = {S_val - S_prev:+.6e}")
    print(f"  (S should decrease for minimax:  min_C,y,z  max_Ĉ,ŷ,ẑ)")

    print("\nDone.")
