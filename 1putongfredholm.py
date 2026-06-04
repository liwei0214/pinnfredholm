"""
PINN comparison for smooth Fredholm integral equation of the second kind.

Equation:  phi(x) = f(x) + lambda * int_0^1 K(x,s) phi(s) ds
Kernel:    K(x,s) = exp(-(x-s)^2)
Solution:  phi*(x) = x(1-x),  lambda = 1
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

output_folder = 'normal_kernel_figures'
os.makedirs(output_folder, exist_ok=True)

print(f"Device: {device}")
print("Equation: phi(x) = f(x) + int_0^1 exp(-(x-s)^2) phi(s) ds")
print("Exact solution: phi*(x) = x(1-x),  lambda = 1")


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def kernel(x, s):
    return np.exp(-(x - s) ** 2)


def true_sol(x):
    return x * (1 - x)


def compute_source(x, N_quad=1000):
    """f(x) = phi*(x) - lambda * int_0^1 K(x,s) phi*(s) ds."""
    s = np.linspace(0, 1, N_quad)
    integrand = kernel(x, s) * true_sol(s)
    return true_sol(x) - np.trapz(integrand, s)


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
    lam = 1.0

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

    K *= lam
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

def train(method, N=50, epochs=3000, lr=4e-4):
    xn  = np.linspace(0, 1, N)
    Km  = build_kernel_matrix(N, method)

    xpts = torch.tensor(xn, dtype=DTYPE, device=device).unsqueeze(-1)
    fv   = torch.tensor([compute_source(x) for x in xn],
                        dtype=DTYPE, device=device).unsqueeze(-1)
    tv   = torch.tensor([true_sol(x) for x in xn],
                        dtype=DTYPE, device=device).unsqueeze(-1)

    model = PINN(512, 5).double().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    loss_history, error_history = [], []
    best_state, best_error = None, float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        phi  = model(xpts)
        loss = (phi - fv - Km @ phi).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        loss_history.append(loss.item())
        with torch.no_grad():
            err = (phi - tv).abs().max().item()
            error_history.append(err)
            if err < best_error:
                best_error = err
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 500 == 0:
            print(f"  [{method:10s}] ep {ep:5d}: loss={loss.item():.3e}, err={err:.3e}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    x_test = torch.linspace(0, 1, 200, dtype=DTYPE, device=device).unsqueeze(-1)
    with torch.no_grad():
        phi_pred = model(x_test).cpu().numpy().flatten()

    x_np  = x_test.cpu().numpy().flatten()
    phi_ex = np.array([true_sol(x) for x in x_np])

    max_err  = np.max(np.abs(phi_pred - phi_ex))
    mean_err = np.mean(np.abs(phi_pred - phi_ex))
    rel_l2   = (np.sqrt(np.mean((phi_pred - phi_ex) ** 2))
                / (np.sqrt(np.mean(phi_ex ** 2)) + 1e-15))
    mse      = np.mean((phi_pred - phi_ex) ** 2)

    Im   = torch.eye(N, dtype=DTYPE, device=device)
    cond = torch.linalg.cond(Im - Km).item()

    print(f"  [{method:10s}] max={max_err:.3e}, rel_l2={rel_l2:.3e}, "
          f"mse={mse:.3e}, cond={cond:.2e}")

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
        'phys_residual': loss_history[-1],
        'cond':          cond,
        'loss_history':  loss_history,
        'error_history': error_history,
        'predictions':   phi_pred,
        'x':             x_np,
        'exact':         phi_ex,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(results):
    colors  = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', 's', '^']
    x_test  = results[0]['x']
    exact   = results[0]['exact']

    # Solution profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_test, exact, 'k-', lw=2.5, label='Exact: $x(1-x)$')
    for i, r in enumerate(results):
        ax.plot(x_test[::5], r['predictions'][::5],
                markers[i], color=colors[i], ms=4,
                label=f"{r['method']} (rel={r['rel_l2']:.1e})")
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\varphi(x)$')
    ax.set_title('Smooth kernel: $K(x,s)=e^{-(x-s)^2}$,  $N=50$')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure1_normal_kernel_compare_N50.png'),
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
    ax.set_title('Training loss: smooth kernel ($N=50$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure2_training_loss_N50.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Summary table
    print("\nResults summary (smooth kernel, N=50):")
    print(f"{'Method':<18} {'Max Err':>12} {'Rel L2':>12} {'MSE':>12} "
          f"{'Phys Res':>12} {'Cond':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<18} {r['max_error']:>12.3e} {r['rel_l2']:>12.3e} "
              f"{r['mse']:>12.3e} {r['phys_residual']:>12.3e} {r['cond']:>12.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N      = 50
    epochs = 3000

    results = []
    for method in ['discrete', 'endpoint', 'midpoint']:
        results.append(train(method, N=N, epochs=epochs))

    plot_results(results)
    print(f"\nFigures saved to: {output_folder}/")


if __name__ == '__main__':
    main()
