"""
PINN comparison for regularized Abel transform (CT reconstruction).

Equation:  mu(r) = f(r) + lambda * int_0^1 K(r,s) mu(s) ds
Kernel:    K(r,s) = 2s / sqrt((r+eps)^2 + s^2),  eps=0.05,  lambda=0.5
Solution:  mu*(s) = s(1-s)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import integrate
import os

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

output_folder = 'abel_ct_figures'
os.makedirs(output_folder, exist_ok=True)

print(f"Device: {device}")
print("Equation: mu(r) = f(r) + 0.5 * int_0^1 K(r,s) mu(s) ds")
print("Kernel:   K(r,s) = 2s / sqrt((r+0.05)^2 + s^2)")
print("Exact solution: mu*(s) = s(1-s)")

EPS = 0.05
LAM = 0.5


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def kernel(r, s):
    return 2 * s / np.sqrt((r + EPS) ** 2 + s ** 2)


def true_sol(s):
    return s * (1 - s)


def compute_source(r, N_quad=1000):
    """f(r) = mu*(r) - lam * int_0^1 K(r,s) mu*(s) ds."""
    s = np.linspace(0, 1, N_quad)
    integrand = kernel(r, s) * true_sol(s)
    return true_sol(r) - LAM * np.trapz(integrand, s)


def verify_solution(N_test=10):
    """Check max residual of exact solution."""
    xs = np.linspace(0.05, 0.95, N_test)
    max_res = 0.0
    for r in xs:
        integ, _ = integrate.quad(lambda s: kernel(r, s) * true_sol(s), 0, 1, limit=200)
        res = abs(true_sol(r) - compute_source(r) - LAM * integ)
        max_res = max(max_res, res)
    print(f"Verification residual: {max_res:.2e}")
    return max_res


# ---------------------------------------------------------------------------
# Kernel matrix construction
# ---------------------------------------------------------------------------

def build_kernel_matrix(N, method):
    """
    Build lambda*K matrix for PINN residual.
    method: 'discrete' | 'endpoint' | 'midpoint'
    """
    xn = np.linspace(0, 1, N)
    h  = 1.0 / (N - 1)
    K  = np.zeros((N, N))

    if method == 'discrete':
        for i in range(N):
            for j in range(N):
                K[i, j] = kernel(xn[i], xn[j]) * h

    elif method == 'endpoint':
        for i in range(N):
            for j in range(N):
                K[i, j] = kernel(xn[i], xn[j]) * h
        K[:, 0]  *= 0.5
        K[:, -1] *= 0.5

    elif method == 'midpoint':
        hm = 1.0 / N
        for i in range(N):
            for j in range(N):
                tm = min((j + 0.5) * hm, 1.0)
                K[i, j] = kernel(xn[i], tm) * hm

    K *= LAM
    return torch.tensor(K, dtype=DTYPE, device=device)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """5-layer x 512-width fully connected network with Tanh activations."""
    def __init__(self, width=512, depth=5):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(method, N=100, epochs=3000, lr=4e-4):
    xn  = np.linspace(0, 1, N)
    Km  = build_kernel_matrix(N, method)

    xpts = torch.tensor(xn, dtype=DTYPE, device=device).unsqueeze(-1)
    fv   = torch.tensor([compute_source(r) for r in xn],
                        dtype=DTYPE, device=device).unsqueeze(-1)
    tv   = torch.tensor([true_sol(r) for r in xn],
                        dtype=DTYPE, device=device).unsqueeze(-1)

    model = PINN(512, 5).double().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    loss_history, error_history = [], []
    best_state, best_error = None, float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        mu   = model(xpts)
        loss = (mu - fv - Km @ mu).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        loss_history.append(loss.item())
        with torch.no_grad():
            err = (mu - tv).abs().max().item()
            error_history.append(err)
            if err < best_error:
                best_error = err
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 500 == 0:
            print(f"  [{method:10s}] ep {ep:5d}: loss={loss.item():.3e}, err={err:.3e}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    r_test = torch.linspace(0, 1, 300, dtype=DTYPE, device=device).unsqueeze(-1)
    with torch.no_grad():
        mu_pred = model(r_test).cpu().numpy().flatten()

    r_np  = r_test.cpu().numpy().flatten()
    mu_ex = np.array([true_sol(r) for r in r_np])

    max_err  = np.max(np.abs(mu_pred - mu_ex))
    mean_err = np.mean(np.abs(mu_pred - mu_ex))
    rel_l2   = (np.sqrt(np.mean((mu_pred - mu_ex) ** 2))
                / (np.sqrt(np.mean(mu_ex ** 2)) + 1e-15))
    mse      = np.mean((mu_pred - mu_ex) ** 2)

    # PSNR (peak = max of true solution = 0.25)
    peak = np.max(np.abs(mu_ex))
    psnr = 20 * np.log10(peak / (np.sqrt(mse) + 1e-15))

    Im   = torch.eye(N, dtype=DTYPE, device=device)
    cond = torch.linalg.cond(Im - Km).item()

    print(f"  [{method:10s}] max={max_err:.3e}, rel_l2={rel_l2:.3e}, "
          f"mse={mse:.3e}, PSNR={psnr:.1f} dB, cond={cond:.2e}")

    name_map = {
        'discrete': 'Discrete Coord.',
        'endpoint': 'Endpoint',
        'midpoint': 'Midpoint',
    }
    return {
        'method':        name_map[method],
        'max_error':     max_err,
        'mean_error':    mean_err,
        'rel_l2':        rel_l2,
        'mse':           mse,
        'psnr':          psnr,
        'phys_residual': loss_history[-1],
        'cond':          cond,
        'loss_history':  loss_history,
        'error_history': error_history,
        'predictions':   mu_pred,
        'r':             r_np,
        'exact':         mu_ex,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(results):
    colors  = ['#0072B2', '#D55E00', '#009E73']
    r_test  = results[0]['r']
    exact   = results[0]['exact']

    # Reconstruction profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_test, exact, 'k-', lw=2.5, label='Exact: $s(1-s)$')
    for i, r in enumerate(results):
        ax.plot(r_test, r['predictions'], color=colors[i], ls='--', lw=2,
                label=f"{r['method']} (PSNR={r['psnr']:.1f} dB)")
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\mu(r)$')
    ax.set_title('Abel CT reconstruction ($N=100$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure5_abel_ct_compare_N100.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Training loss
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        ep = np.arange(1, len(r['loss_history']) + 1)
        ax.semilogy(ep[::10], r['loss_history'][::10],
                    color=colors[i], lw=2, label=r['method'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics residual loss')
    ax.set_title('Training loss: Abel CT ($N=100$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure6_training_loss_N100.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Summary table
    print("\nResults summary (Abel CT, N=100):")
    print(f"{'Method':<18} {'Max Err':>12} {'MAE':>12} {'MSE':>12} "
          f"{'PSNR (dB)':>12} {'Phys Res':>12}")
    print("-" * 82)
    for r in results:
        print(f"{r['method']:<18} {r['max_error']:>12.3e} {r['mean_error']:>12.3e} "
              f"{r['mse']:>12.3e} {r['psnr']:>12.1f} {r['phys_residual']:>12.3e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    verify_solution()

    N      = 100
    epochs = 3000

    results = []
    for method in ['discrete', 'endpoint', 'midpoint']:
        results.append(train(method, N=N, epochs=epochs))

    plot_results(results)
    print(f"\nFigures saved to: {output_folder}/")


if __name__ == '__main__':
    main()
