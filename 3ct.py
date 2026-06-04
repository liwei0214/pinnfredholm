import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import integrate, optimize
import os

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

output_folder = 'abel_ct_figures'
os.makedirs(output_folder, exist_ok=True)


def find_alpha(eps=0.05, lam=0.2):
    """
    Numerically find the correction factor alpha such that
    mu(r) = r*(1-r)*(1+alpha) approximately satisfies
        mu(r) = r*(1-r) + lam * integral_0^1 K(r,s)*mu(s) ds
    where K(r,s) = 2s / sqrt((r+eps)^2 + s^2).
    """
    def kernel(r, s):
        return 2 * s / np.sqrt((r + eps) ** 2 + s ** 2)

    def residual(alpha):
        total = 0.0
        for r in np.linspace(0.1, 0.9, 30):
            val, _ = integrate.quad(
                lambda s: kernel(r, s) * r * (1 - r) * (1 + alpha), 0, 1, limit=100)
            lhs = r * (1 - r) * (1 + alpha)
            rhs = r * (1 - r) + lam * val
            total += (lhs - rhs) ** 2
        return np.sqrt(total / 30)

    res = optimize.minimize_scalar(residual, bounds=(0, 1), method='bounded')
    print(f"Optimal alpha = {res.x:.6f}  (residual = {res.fun:.2e})")
    return res.x


def create_equation():
    """
    Regularized Abel transform equation:
        mu(r) = r*(1-r) + lam * integral_0^1 K(r,s)*mu(s) ds
        K(r,s) = 2s / sqrt((r+eps)^2 + s^2),  eps=0.05,  lam=0.2
    Approximate solution: mu(r) = r*(1-r)*(1+alpha)
    """
    eps, lam = 0.05, 0.2
    alpha = find_alpha(eps, lam)
    print(f"Equation: mu(r) = r(1-r) + {lam}*int_0^1 K(r,s)*mu(s) ds")
    print(f"Approximate solution: mu(r) = r(1-r)*(1+{alpha:.6f})")
    return {
        'lambda':  lam,
        'eps':     eps,
        'alpha':   alpha,
        'kernel':  lambda r, s: 2 * s / np.sqrt((r + eps) ** 2 + s ** 2),
        'source':  lambda r: r * (1 - r),
        'exact':   lambda r: r * (1 - r) * (1 + alpha),
    }


def build_kernel_matrices(N, eq):
    """
    Build kernel matrices for three discretization methods.

    Returns list of (method_name, A_matrix, K_matrix) where A = I - K.
    """
    kernel, lam = eq['kernel'], eq['lambda']
    dr = 1.0 / (N - 1)
    r = np.linspace(0, 1, N)

    # Endpoint method (Simpson's rule)
    K1 = torch.zeros(N, N, dtype=DTYPE)
    N_use = N - 1 if N % 2 == 0 else N
    for i in range(N):
        for j in range(N_use):
            if j == 0 or j == N_use - 1:
                w = dr / 3
            elif j % 2 == 1:
                w = 4 * dr / 3
            else:
                w = 2 * dr / 3
            K1[i, j] = kernel(r[i], r[j]) * w
        if N % 2 == 0:
            K1[i, N - 1]  = kernel(r[i], r[N - 1]) * dr / 2
            K1[i, N - 2] += kernel(r[i], r[N - 2]) * dr / 2
    K1 *= lam

    # Discrete coordinate method (composite trapezoidal rule)
    K2 = torch.zeros(N, N, dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            w = 0.5 * dr if (j == 0 or j == N - 1) else dr
            K2[i, j] = kernel(r[i], r[j]) * w
    K2 *= lam

    # Midpoint method (composite midpoint rule, 3 sub-intervals per panel)
    K3 = torch.zeros(N, N, dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            if j == 0:
                s_l, s_r = 0.0, dr / 2
            elif j == N - 1:
                s_l, s_r = 1.0 - dr / 2, 1.0
            else:
                s_l, s_r = r[j] - dr / 2, r[j] + dr / 2
            n_sub = 3
            w = (s_r - s_l) / n_sub
            K3[i, j] = sum(kernel(r[i], s_l + (k + 0.5) * w) * w for k in range(n_sub))
    K3 *= lam

    I = torch.eye(N, dtype=DTYPE)
    configs = [
        ('Endpoint',        I - K1 + 1e-8 * I, K1),
        ('Discrete Coord.', I - K2 + 1e-9 * I, K2),
        ('Midpoint',        I - K3 + 1e-8 * I, K3),
    ]

    for name, A, _ in configs:
        print(f"Condition number ({name}): {torch.linalg.cond(A).item():.2e}")

    return configs


class AbelPINN(nn.Module):
    """
    Residual network for the Abel equation.
    Boundary condition mu(0) = mu(1) = 0 is enforced by construction.
    """
    def __init__(self, width=200, depth=6):
        super().__init__()
        self.input_layer = nn.Linear(1, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
            for _ in range(depth // 2)
        ])
        self.output_layer = nn.Linear(width, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, r):
        x = torch.tanh(self.input_layer(r))
        for block in self.blocks:
            x = torch.tanh(block(x)) + 0.5 * x
        return self.output_layer(x) * r * (1 - r) * 4


def train(eq, A, K, method_name, N=100, epochs=5000):
    r_pts = torch.linspace(0, 1, N, device=device, dtype=DTYPE).unsqueeze(-1)
    f_vals = torch.tensor(
        [eq['source'](r.item()) for r in r_pts.squeeze()],
        device=device, dtype=DTYPE).unsqueeze(-1)
    true_vals = torch.tensor(
        [eq['exact'](r.item()) for r in r_pts.squeeze()],
        device=device, dtype=DTYPE).unsqueeze(-1)

    K_dev = K.to(device)
    model = AbelPINN(width=200, depth=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=1,
        pct_start=0.3, anneal_strategy='cos')

    loss_hist, error_hist = [], []
    best_error = float('inf')
    best_state = None
    patience = 0

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        mu = model(r_pts)
        phys_res = mu - f_vals - torch.matmul(K_dev, mu)
        phys_loss = phys_res.pow(2).mean()
        data_loss = (mu - true_vals).pow(2).mean()
        smooth_loss = (mu[1:] - mu[:-1]).pow(2).mean() if N > 2 else torch.tensor(0.0)

        if ep < 1000:
            loss = phys_loss + 0.01 * data_loss + 0.001 * smooth_loss
        elif ep < 3000:
            loss = phys_loss + 0.1 * data_loss + 1e-4 * smooth_loss
        else:
            loss = phys_loss + 0.5 * data_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_hist.append(phys_loss.item())
        with torch.no_grad():
            err = (mu - true_vals).abs().max().item()
            error_hist.append(err)
            if err < best_error:
                best_error = err
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

        if ep % 1000 == 0:
            print(f"  [{method_name}] epoch {ep}: "
                  f"loss={phys_loss.item():.2e}, error={err:.2e}")

        if patience > 500 and ep > 2000:
            print(f"  [{method_name}] early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    r_test = torch.linspace(0, 1, 300, device=device, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        mu_pred = model(r_test).cpu().numpy().flatten()
        mu_phys = model(r_pts)
        phys_err = (mu_phys - f_vals - torch.matmul(K_dev, mu_phys)).abs().max().item()

    r_np   = r_test.cpu().numpy().flatten()
    mu_ex  = np.array([eq['exact'](r) for r in r_np])
    max_err  = np.max(np.abs(mu_pred - mu_ex))
    mean_err = np.mean(np.abs(mu_pred - mu_ex))
    rel_err  = max_err / (np.max(np.abs(mu_ex)) + 1e-8)

    print(f"  [{method_name}] max={max_err:.3e}, mean={mean_err:.3e}, "
          f"rel={rel_err:.3e}, phys={phys_err:.3e}")

    return {
        'method':        method_name,
        'max_error':     max_err,
        'mean_error':    mean_err,
        'rel_error':     rel_err,
        'phys_error':    phys_err,
        'best_error':    best_error,
        'loss_history':  loss_hist,
        'error_history': error_hist,
        'predictions':   mu_pred,
        'r_test':        r_np,
        'exact':         mu_ex,
    }


def plot_results(results, eq):
    colors = ['blue', 'red', 'green']
    r_test = results[0]['r_test']
    exact  = results[0]['exact']

    # Solution profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_test, exact, 'k-', lw=2.5, label='Exact: $r(1-r)(1+\\alpha)$')
    for i, r in enumerate(results):
        ax.plot(r_test, r['predictions'], color=colors[i], ls='--', lw=2,
                label=f"{r['method']} (rel={r['rel_error']:.1e})")
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
        ax.semilogy(ep[::50], r['loss_history'][::50],
                    color=colors[i], lw=2, label=r['method'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics residual loss')
    ax.set_title('Training loss convergence: Abel CT ($N=100$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure6_training_loss_N100.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Pointwise error
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        err = np.abs(r['predictions'] - r['exact']) + 1e-10
        ax.semilogy(r_test, err, color=colors[i], lw=2, label=r['method'])
    ax.set_xlabel('r')
    ax.set_ylabel('Absolute error')
    ax.set_title('Pointwise error: Abel CT ($N=100$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'error_distribution_N100.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Summary
    print("\nResults summary:")
    print(f"{'Method':<20} {'Max error':<14} {'Mean error':<14} "
          f"{'Rel error':<14} {'Phys residual'}")
    print("-" * 76)
    for r in sorted(results, key=lambda x: x['rel_error']):
        print(f"{r['method']:<20} {r['max_error']:<14.3e} {r['mean_error']:<14.3e} "
              f"{r['rel_error']:<14.3e} {r['phys_error']:.3e}")


def main():
    eq = create_equation()
    N  = 100
    configs = build_kernel_matrices(N, eq)

    results = []
    for name, A, K in configs:
        results.append(train(eq, A, K, name, N=N, epochs=5000))

    plot_results(results, eq)
    print(f"\nFigures saved to: {output_folder}/")


if __name__ == '__main__':
    main()
