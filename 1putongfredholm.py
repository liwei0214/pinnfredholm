import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

output_folder = 'normal_kernel_figures'
os.makedirs(output_folder, exist_ok=True)


def create_equation():
    """
    Fredholm integral equation of the second kind:
        phi(s) + integral_0^1 s*exp(t)*phi(t) dt = exp(-s)
    Standard form: phi(s) = f(s) + lambda * integral_0^1 K(s,t)*phi(t) dt
        f(s) = exp(-s),  K(s,t) = s*exp(t),  lambda = -1
    Exact solution: phi(s) = exp(-s) - s/2
    """
    print("Equation: phi(s) + int_0^1 s*exp(t)*phi(t) dt = exp(-s)")
    print("Exact solution: phi(s) = exp(-s) - s/2")

    # Verify exact solution at s = 0.5
    s = 0.5
    phi = np.exp(-s) - s / 2
    integral = s * (1 - 0.5 + (np.e - 1) / 2)
    residual = phi + integral - np.exp(-s)
    print(f"Verification at s=0.5: residual = {abs(residual):.2e}")

    return {
        'lambda': -1.0,
        'kernel': lambda s, t: s * np.exp(t),
        'source': lambda s: np.exp(-s),
        'exact':  lambda s: np.exp(-s) - s / 2,
    }


def build_kernel_matrices(N, kernel, lam):
    """
    Build kernel matrices for three discretization methods.

    Returns K1 (endpoint/trapezoidal), K2 (discrete coordinate/rectangular),
    K3 (midpoint).
    """
    dx = 1.0 / (N - 1)
    s = np.linspace(0, 1, N)

    # Endpoint method (composite trapezoidal rule)
    K1 = torch.zeros(N, N, dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            K1[i, j] = kernel(s[i], s[j]) * dx
    K1[:, 0]  *= 0.5
    K1[:, -1] *= 0.5
    K1 *= lam

    # Discrete coordinate method (rectangular rule)
    K2 = torch.zeros(N, N, dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            K2[i, j] = kernel(s[i], s[j]) * dx
    K2 *= lam

    # Midpoint method
    K3 = torch.zeros(N, N, dtype=DTYPE)
    dx_mid = 1.0 / N
    for i in range(N):
        for j in range(N):
            if j < N - 1:
                t_mid = (j + 0.5) * dx_mid
                K3[i, j] = kernel(s[i], t_mid) * dx_mid
            else:
                t_end = min(1.0, (j + 0.5) * dx_mid)
                K3[i, j] = kernel(s[i], t_end) * dx_mid * 0.5
    K3 *= lam

    I = torch.eye(N, dtype=DTYPE)
    print(f"Condition numbers (N={N}): "
          f"endpoint={torch.linalg.cond(I-K1).item():.2e}, "
          f"discrete={torch.linalg.cond(I-K2).item():.2e}, "
          f"midpoint={torch.linalg.cond(I-K3).item():.2e}")

    return K1, K2, K3


class PINN(nn.Module):
    def __init__(self, width=128, depth=4):
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

    def forward(self, s):
        return self.net(s)


def train(eq, K, method_name, N=50, epochs=2400):
    s_pts = torch.linspace(0, 1, N, device=device, dtype=DTYPE).unsqueeze(-1)
    f_vals = torch.tensor(
        [eq['source'](s.item()) for s in s_pts.squeeze()],
        device=device, dtype=DTYPE).unsqueeze(-1)
    true_vals = torch.tensor(
        [eq['exact'](s.item()) for s in s_pts.squeeze()],
        device=device, dtype=DTYPE).unsqueeze(-1)

    K_dev = K.to(device)
    model = PINN(width=128, depth=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    loss_hist, error_hist = [], []
    best_error = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        phi = model(s_pts)
        residual = phi - f_vals - torch.matmul(K_dev, phi)
        loss = residual.pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            err = (phi - true_vals).abs().max().item()
            error_hist.append(err)
            best_error = min(best_error, err)

        if ep % 600 == 0:
            print(f"  [{method_name}] epoch {ep}: loss={loss.item():.2e}, error={err:.2e}")

    # Evaluate on dense grid
    model.eval()
    s_test = torch.linspace(0, 1, 200, device=device, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        phi_pred = model(s_test).cpu().numpy().flatten()

    s_np   = s_test.cpu().numpy().flatten()
    phi_ex = np.array([eq['exact'](s) for s in s_np])

    max_err = np.max(np.abs(phi_pred - phi_ex))
    mean_err = np.mean(np.abs(phi_pred - phi_ex))
    rel_err  = max_err / (np.max(np.abs(phi_ex)) + 1e-8)

    print(f"  [{method_name}] max={max_err:.3e}, mean={mean_err:.3e}, rel={rel_err:.3e}")

    return {
        'method':       method_name,
        'max_error':    max_err,
        'mean_error':   mean_err,
        'rel_error':    rel_err,
        'best_error':   best_error,
        'loss_history': loss_hist,
        'error_history':error_hist,
        'predictions':  phi_pred,
        's_test':       s_np,
        'exact':        phi_ex,
    }


def plot_results(results):
    colors  = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    s_test  = results[0]['s_test']
    exact   = results[0]['exact']

    # Solution profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_test, exact, 'k-', lw=2.5, label='Exact: $e^{-s} - s/2$')
    for i, r in enumerate(results):
        ax.plot(s_test[::5], r['predictions'][::5], markers[i],
                color=colors[i], ms=4,
                label=f"{r['method']} (rel={r['rel_error']:.1e})")
    ax.set_xlabel('s')
    ax.set_ylabel(r'$\varphi(s)$')
    ax.set_title('Solution comparison: smooth kernel ($K(s,t)=se^t$, $N=50$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure1_normal_kernel_compare_N50.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Training loss
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        epochs = np.arange(1, len(r['loss_history']) + 1)
        ax.semilogy(epochs[::10], r['loss_history'][::10],
                    color=colors[i], lw=2, label=r['method'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.set_title('Training loss convergence: smooth kernel ($N=50$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'figure2_training_loss_N50.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Pointwise error
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        err = np.abs(r['predictions'] - r['exact']) + 1e-15
        ax.semilogy(s_test, err, color=colors[i], lw=2, label=r['method'])
    ax.set_xlabel('s')
    ax.set_ylabel('Absolute error')
    ax.set_title('Pointwise error distribution: smooth kernel ($N=50$)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'error_distribution_N50.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Summary table
    print("\nResults summary:")
    print(f"{'Method':<20} {'Max error':<14} {'Mean error':<14} {'Rel error':<14}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: x['rel_error']):
        print(f"{r['method']:<20} {r['max_error']:<14.3e} "
              f"{r['mean_error']:<14.3e} {r['rel_error']:<14.3e}")


def main():
    eq = create_equation()
    N  = 50
    K1, K2, K3 = build_kernel_matrices(N, eq['kernel'], eq['lambda'])

    methods = [
        ('Endpoint',          K1),
        ('Discrete Coord.',   K2),
        ('Midpoint',          K3),
    ]

    results = []
    for name, K in methods:
        results.append(train(eq, K, name, N=N, epochs=2400))

    plot_results(results)
    print(f"\nFigures saved to: {output_folder}/")


if __name__ == '__main__':
    main()
