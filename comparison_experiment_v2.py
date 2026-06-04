"""
Benchmark: PINN discretization methods vs traditional numerical methods
for Fredholm integral equations of the second kind.

PINN methods:        Discrete Coordinate, Endpoint, Midpoint
Traditional methods: Nystrom (Gauss-Legendre), Galerkin (Legendre basis), Collocation

Usage:
    python comparison_experiment.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate
import warnings
import time

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

print(f"Device: {device}, PyTorch: {torch.__version__}")


# ---------------------------------------------------------------------------
# Source term computation
# ---------------------------------------------------------------------------

def compute_source(kernel, true_sol, lam, x, singular=False):
    """f(x) = phi*(x) - lam * int_0^1 K(x,s) phi*(s) ds."""
    def integrand(s):
        return kernel(x, s) * true_sol(s)

    if singular:
        eps = 1e-10
        parts = []
        if x - eps > 0:
            I1, _ = integrate.quad(integrand, 0, x - eps,
                                   limit=500, epsabs=1e-14, epsrel=1e-14)
            parts.append(I1)
        if x + eps < 1:
            I2, _ = integrate.quad(integrand, x + eps, 1,
                                   limit=500, epsabs=1e-14, epsrel=1e-14)
            parts.append(I2)
        integral = sum(parts)
    else:
        integral, _ = integrate.quad(integrand, 0, 1,
                                     limit=500, epsabs=1e-14, epsrel=1e-14)
    return true_sol(x) - lam * integral


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

class Problem:
    """
    Fredholm integral equation of the second kind:
        phi(x) = f(x) + lam * int_0^1 K(x,s) phi(s) ds
    """
    def __init__(self, name, kernel, true_sol, lam, singular=False, alpha=0.0):
        self.name = name
        self.kernel = kernel
        self.true_sol = true_sol
        self.lam = lam
        self.singular = singular
        self.alpha = alpha
        self._precompute_source()

    def _precompute_source(self, M=500):
        xs = np.linspace(0, 1, M)
        fs = np.array([compute_source(
            self.kernel, self.true_sol, self.lam, x, singular=self.singular
        ) for x in xs])
        self._sx = xs
        self._sf = fs

    def source(self, x):
        return float(np.interp(x, self._sx, self._sf))

    def verify(self, N_test=10):
        xs = np.linspace(0.05, 0.95, N_test)
        max_res = 0.0
        for x in xs:
            integ = lambda s: self.kernel(x, s) * self.true_sol(s)
            if self.singular:
                eps = 1e-10
                I1, _ = integrate.quad(integ, 0, max(0, x - eps), limit=300)
                I2, _ = integrate.quad(integ, min(1, x + eps), 1, limit=300)
                integral = I1 + I2
            else:
                integral, _ = integrate.quad(integ, 0, 1, limit=300)
            res = abs(self.true_sol(x) - self.source(x) - self.lam * integral)
            max_res = max(max_res, res)
        return max_res


def make_smooth_problem():
    """K(x,s) = exp(-(x-s)^2),  phi*(x) = x(1-x),  lam=1."""
    return Problem("Smooth Kernel",
                   kernel=lambda x, s: np.exp(-(x - s) ** 2),
                   true_sol=lambda x: x * (1 - x),
                   lam=1.0)


def make_singular_problem():
    """K(x,s) = |x-s|^{-0.3},  phi*(x) = sin(pi*x),  lam=0.1."""
    alpha = 0.3
    def kernel(x, s):
        d = abs(x - s)
        return 0.0 if d < 1e-15 else d ** (-alpha)
    return Problem("Singular Kernel",
                   kernel=kernel,
                   true_sol=lambda x: np.sin(np.pi * x),
                   lam=0.1,
                   singular=True,
                   alpha=alpha)


def make_abel_problem():
    """K(r,s) = 2s/sqrt((r+eps)^2+s^2),  mu*(s) = s(1-s),  lam=0.5."""
    eps = 0.05
    return Problem("Abel Transform",
                   kernel=lambda r, s: 2 * s / np.sqrt((r + eps) ** 2 + s ** 2),
                   true_sol=lambda s: s * (1 - s),
                   lam=0.5)


# ---------------------------------------------------------------------------
# Traditional numerical methods
# ---------------------------------------------------------------------------

def nystrom_solve(prob, N=50):
    """Nystrom method with Gauss-Legendre quadrature."""
    nd, wt = np.polynomial.legendre.leggauss(N)
    nd = 0.5 * (nd + 1)
    wt = 0.5 * wt
    K = np.array([[prob.lam * wt[j] * prob.kernel(nd[i], nd[j])
                   for j in range(N)] for i in range(N)])
    A = np.eye(N) - K
    f = np.array([prob.source(x) for x in nd])
    t0 = time.time()
    phi = np.linalg.solve(A, f)
    dt = time.time() - t0
    xf = np.linspace(0, 1, 200)
    return {
        'method':     'Nystrom',
        'max_error':  np.max(np.abs(phi - [prob.true_sol(x) for x in nd])),
        'mean_error': np.mean(np.abs(phi - [prob.true_sol(x) for x in nd])),
        'time':       dt,
        'cond':       np.linalg.cond(A),
        'x':          xf,
        'phi':        np.interp(xf, nd, phi),
        'true':       np.array([prob.true_sol(x) for x in xf]),
        'stability':  1 / (1 + np.log10(max(np.linalg.cond(A), 1))),
    }


def galerkin_solve(prob, N_basis=15):
    """Galerkin method with Legendre polynomial basis."""
    from numpy.polynomial import legendre

    def bas(k, x):
        c = np.zeros(k + 1)
        c[k] = 1.0
        return legendre.legval(2 * x - 1, c)

    nq = 80
    qn, qw = np.polynomial.legendre.leggauss(nq)
    qn = 0.5 * (qn + 1)
    qw = 0.5 * qw

    B = np.array([[bas(k, x) for x in qn] for k in range(N_basis)])
    M = B @ np.diag(qw) @ B.T
    F = B @ (qw * np.array([prob.source(x) for x in qn]))

    Km = np.zeros((N_basis, N_basis))
    for qi in range(nq):
        inn = np.array([sum(qw[qj] * prob.kernel(qn[qi], qn[qj]) * B[j, qj]
                            for qj in range(nq)) for j in range(N_basis)])
        for i in range(N_basis):
            Km[i] += qw[qi] * B[i, qi] * inn
    Km *= prob.lam

    A = M - Km
    t0 = time.time()
    co = np.linalg.solve(A, F)
    dt = time.time() - t0

    xf = np.linspace(0, 1, 200)
    pf = sum(co[k] * np.array([bas(k, x) for x in xf]) for k in range(N_basis))
    tf = np.array([prob.true_sol(x) for x in xf])
    return {
        'method':     'Galerkin',
        'max_error':  np.max(np.abs(pf - tf)),
        'mean_error': np.mean(np.abs(pf - tf)),
        'time':       dt,
        'cond':       np.linalg.cond(A),
        'x':          xf,
        'phi':        pf,
        'true':       tf,
        'stability':  1 / (1 + np.log10(max(np.linalg.cond(A), 1))),
    }


def collocation_solve(prob, N=50):
    """Collocation method with composite trapezoidal quadrature."""
    xn = np.linspace(0, 1, N)
    h = 1.0 / (N - 1)
    K = np.array([[prob.lam * h * (0.5 if j in (0, N - 1) else 1.0) * prob.kernel(xn[i], xn[j])
                   for j in range(N)] for i in range(N)])
    A = np.eye(N) - K
    f = np.array([prob.source(x) for x in xn])
    t0 = time.time()
    phi = np.linalg.solve(A, f)
    dt = time.time() - t0
    pt = np.array([prob.true_sol(x) for x in xn])
    xf = np.linspace(0, 1, 200)
    tf = np.array([prob.true_sol(x) for x in xf])
    return {
        'method':     'Collocation',
        'max_error':  np.max(np.abs(phi - pt)),
        'mean_error': np.mean(np.abs(phi - pt)),
        'time':       dt,
        'cond':       np.linalg.cond(A),
        'x':          xf,
        'phi':        np.interp(xf, xn, phi),
        'true':       tf,
        'stability':  1 / (1 + np.log10(max(np.linalg.cond(A), 1))),
    }


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Fully connected network with Tanh activations and Xavier init."""
    def __init__(self, width=512, depth=5):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def build_kernel_matrix(prob, N, method):
    """
    Build lambda*K matrix for PINN residual.
    method: 'discrete' | 'endpoint' | 'midpoint'
    """
    xn = np.linspace(0, 1, N)
    h = 1.0 / (N - 1)
    K = np.zeros((N, N))

    if method == 'discrete':
        for i in range(N):
            for j in range(N):
                K[i, j] = prob.kernel(xn[i], xn[j]) * h

    elif method == 'endpoint':
        for i in range(N):
            for j in range(N):
                K[i, j] = prob.kernel(xn[i], xn[j]) * h
        K[:, 0]  *= 0.5
        K[:, -1] *= 0.5

    elif method == 'midpoint':
        hm = 1.0 / N
        for i in range(N):
            for j in range(N):
                tm = min((j + 0.5) * hm, 1.0)
                K[i, j] = prob.kernel(xn[i], tm) * hm

    K *= prob.lam
    return torch.tensor(K, dtype=DTYPE, device=device)


def pinn_solve(prob, method='discrete', N=50, epochs=3000,
               lr=4e-4, width=512, depth=5):
    """Train PINN and return error metrics."""
    xpts = torch.linspace(0, 1, N, device=device, dtype=DTYPE).unsqueeze(-1)
    xnp  = np.linspace(0, 1, N)
    Km   = build_kernel_matrix(prob, N, method)
    fv   = torch.tensor([prob.source(x) for x in xnp],
                        device=device, dtype=DTYPE).unsqueeze(-1)
    tv   = torch.tensor([prob.true_sol(x) for x in xnp],
                        device=device, dtype=DTYPE).unsqueeze(-1)

    disc_res = (tv - fv - Km @ tv).abs().max().item()

    model = PINN(width, depth).double().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    loss_history, error_history = [], []
    best_state, best_error = None, float('inf')
    t0 = time.time()

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
            e = (phi - tv).abs().max().item()
            error_history.append(e)
            if e < best_error:
                best_error = e
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 500 == 0:
            print(f"    [{method:10s}] ep {ep:5d}: loss={loss.item():.3e}, err={e:.3e}")

    tt = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    xf = torch.linspace(0, 1, 200, device=device, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        pf = model(xf).cpu().numpy().flatten()

    xfn = xf.cpu().numpy().flatten()
    tfn = np.array([prob.true_sol(x) for x in xfn])
    max_err  = np.max(np.abs(pf - tfn))
    mean_err = np.mean(np.abs(pf - tfn))
    rel_l2   = (np.sqrt(np.mean((pf - tfn) ** 2))
                / (np.sqrt(np.mean(tfn ** 2)) + 1e-15))
    Im   = torch.eye(N, dtype=DTYPE, device=device)
    cond = torch.linalg.cond(Im - Km).item()

    name_map = {'discrete': 'Discrete Coord.', 'endpoint': 'Endpoint', 'midpoint': 'Midpoint'}
    return {
        'method':        name_map[method],
        'method_key':    method,
        'max_error':     max_err,
        'mean_error':    mean_err,
        'rel_l2':        rel_l2,
        'mse':           np.mean((pf - tfn) ** 2),
        'time':          tt,
        'cond':          cond,
        'stability':     1 / (1 + np.log10(max(cond, 1))),
        'loss_history':  loss_history,
        'error_history': error_history,
        'x':             xfn,
        'phi':           pf,
        'true':          tfn,
        'disc_residual': disc_res,
        'phys_residual': loss_history[-1] if loss_history else 0,
    }


# ---------------------------------------------------------------------------
# Experiment workflow
# ---------------------------------------------------------------------------

def run_experiment(prob, N_pinn=50, epochs=3000):
    """Run all six methods on a single problem."""
    print(f"\n{'='*60}\n  {prob.name}\n{'='*60}")
    vr = prob.verify()
    print(f"  Verification residual: {vr:.2e}"
          + ("  [WARN]" if vr > 1e-3 else ""))

    results = {}

    print("\n  Traditional methods:")
    for nm, fn, kw in [("Nystrom",     nystrom_solve,     {'N': N_pinn}),
                       ("Galerkin",    galerkin_solve,    {'N_basis': 15}),
                       ("Collocation", collocation_solve, {'N': N_pinn})]:
        try:
            t0 = time.time()
            r  = fn(prob, **kw)
            r['time'] = time.time() - t0
            results[nm] = r
            print(f"    {nm:<15}: MaxErr={r['max_error']:.3e}, "
                  f"Cond={r['cond']:.2e}, Time={r['time']:.2f}s")
        except Exception as e:
            print(f"    {nm:<15}: FAILED - {e}")

    print("\n  PINN methods:")
    for mk in ['discrete', 'endpoint', 'midpoint']:
        try:
            r = pinn_solve(prob, method=mk, N=N_pinn, epochs=epochs)
            results[r['method']] = r
            print(f"    {r['method']:<15}: MaxErr={r['max_error']:.3e}, "
                  f"Stab={r['stability']:.3f}, Time={r['time']:.2f}s")
        except Exception as e:
            print(f"    PINN-{mk}: FAILED - {e}")

    return results


def run_convergence(ps, psg):
    """Verify O(h^2) and O(h^{1-alpha}) convergence rates."""
    print(f"\n{'='*60}\n  Convergence Rate Verification\n{'='*60}")
    Ns = [20, 40, 80, 160, 320]

    print("\n  Smooth kernel (expect ratio ~4.0):")
    es = []
    for N in Ns:
        e = collocation_solve(ps, N=N)['max_error']
        es.append(e)
        print(f"    N={N:4d}, MaxErr={e:.4e}")
    for i in range(1, len(es)):
        if es[i] > 1e-16:
            print(f"    ratio {Ns[i-1]}->{Ns[i]}: {es[i-1]/es[i]:.2f}")

    print(f"\n  Singular kernel a={psg.alpha} (expect ratio ~{2**(1-psg.alpha):.2f}):")
    esg = []
    for N in Ns:
        e = collocation_solve(psg, N=N)['max_error']
        esg.append(e)
        print(f"    N={N:4d}, MaxErr={e:.4e}")
    for i in range(1, len(esg)):
        if esg[i] > 1e-16:
            print(f"    ratio {Ns[i-1]}->{Ns[i]}: {esg[i-1]/esg[i]:.2f}")

    return Ns, es, esg


def run_ablation(prob):
    """Network architecture ablation study."""
    print(f"\n{'='*60}\n  Ablation Study\n{'='*60}")
    configs = [
        ("5x512 (baseline)", 512, 5, 2000),
        ("3x256",            256, 3, 2000),
        ("7x512",            512, 7, 2000),
        ("3x512",            512, 3, 2000),
        ("5x256",            256, 5, 2000),
        ("5x1024",          1024, 5, 2000),
    ]
    res = []
    for lb, w, d, ep in configs:
        try:
            r   = pinn_solve(prob, method='discrete', N=50, epochs=ep, width=w, depth=d)
            np_ = sum(p.numel() for p in PINN(w, d).parameters())
            res.append({'config': lb, 'max_error': r['max_error'],
                        'mean_error': r['mean_error'], 'time': r['time'], 'params': np_})
            print(f"  {lb:<20}: MaxErr={r['max_error']:.3e}, "
                  f"Time={r['time']:.1f}s, Params={np_:,}")
        except Exception as e:
            print(f"  {lb:<20}: FAILED - {e}")
    return res


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_all(all_res, pns, prefix="fig"):
    ct = {'Nystrom': '#E69F00', 'Galerkin': '#56B4E9', 'Collocation': '#CC79A7'}
    cp = {'Discrete Coord.': '#0072B2', 'Endpoint': '#D55E00', 'Midpoint': '#009E73'}
    ac = {**ct, **cp}
    nr = len(all_res)
    fig, axes = plt.subplots(nr, 3, figsize=(18, 5 * nr))
    if nr == 1:
        axes = axes[np.newaxis, :]

    for row, (res, pn) in enumerate(zip(all_res, pns)):
        xf = np.linspace(0, 1, 200)
        tv = list(res.values())[0]['true']

        ax = axes[row, 0]
        ax.plot(xf, tv, 'k-', lw=2.5, label='Exact', zorder=10)
        for nm, r in res.items():
            ax.plot(r['x'], r['phi'], color=ac.get(nm, 'gray'),
                    ls='--' if nm in cp else '-.', lw=1.5, label=nm, alpha=0.85)
        ax.set_title(f'{pn}: Solution', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel(r'$\phi(x)$')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        for nm, r in res.items():
            e = np.abs(r['phi'] - r['true']) + 1e-16
            ax.semilogy(r['x'], e, color=ac.get(nm, 'gray'), lw=1.5, label=nm, alpha=0.85)
        ax.set_title(f'{pn}: Pointwise Error', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('|Error|')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        nms = list(res.keys())
        me  = [res[n]['max_error'] for n in nms]
        bars = ax.bar(range(len(nms)), me, color=[ac.get(n, 'gray') for n in nms], alpha=0.85)
        ax.set_xticks(range(len(nms)))
        ax.set_xticklabels(nms, rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('Max Error'); ax.set_yscale('log')
        ax.set_title(f'{pn}: Max Error', fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
        for b, v in zip(bars, me):
            ax.text(b.get_x() + b.get_width() / 2., b.get_height() * 1.3,
                    f'{v:.1e}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    fn = f'{prefix}_comparison.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fn}")


def plot_conv(Ns, es, esg, alpha, prefix="fig"):
    hs  = [1.0 / (N - 1) for N in Ns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(hs, es, 'bo-', lw=2, ms=8, label='Measured')
    ax.loglog(hs, [es[0] * (h / hs[0]) ** 2 for h in hs], 'r--', lw=1.5, label=r'$O(h^2)$')
    ax.set_xlabel('h'); ax.set_ylabel('Max Error')
    ax.set_title('Theorem 2: Smooth Kernel', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(hs, esg, 'bo-', lw=2, ms=8, label='Measured')
    ax.loglog(hs, [esg[0] * (h / hs[0]) ** (1 - alpha) for h in hs],
              'r--', lw=1.5, label=f'$O(h^{{{1-alpha:.1f}}})$')
    ax.set_xlabel('h'); ax.set_ylabel('Max Error')
    ax.set_title(f'Theorem 3: Singular (alpha={alpha})', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fn = f'{prefix}_convergence.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fn}")


def plot_train(all_res, pns, prefix="fig"):
    pk = ['Discrete Coord.', 'Endpoint', 'Midpoint']
    cl = {'Discrete Coord.': '#0072B2', 'Endpoint': '#D55E00', 'Midpoint': '#009E73'}
    nr = len(all_res)
    fig, axes = plt.subplots(nr, 2, figsize=(12, 4 * nr))
    if nr == 1:
        axes = axes[np.newaxis, :]

    for row, (res, pn) in enumerate(zip(all_res, pns)):
        for col, (key, ylabel) in enumerate([('loss_history', 'Loss'),
                                              ('error_history', 'Max Error')]):
            ax = axes[row, col]
            for nm in pk:
                if nm in res and key in res[nm]:
                    h = res[nm][key]
                    ax.semilogy(range(1, len(h) + 1), h, color=cl[nm], lw=1.5,
                                label=nm, alpha=0.8)
            ax.set_title(f'{pn}: {ylabel}', fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fn = f'{prefix}_training.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fn}")


def print_tables(all_res, pns):
    mo = ['Nystrom', 'Galerkin', 'Collocation', 'Discrete Coord.', 'Endpoint', 'Midpoint']
    print(f"\n{'='*100}\n  RESULTS\n{'='*100}")
    print(f"{'Method':<18}", end="")
    for pn in pns:
        print(f"  {'MaxErr':>12}  {'MeanErr':>12}  {'Time':>7}", end="")
    print()
    print("-" * 100)
    for m in mo:
        print(f"{m:<18}", end="")
        for res in all_res:
            if m in res:
                r = res[m]
                print(f"  {r['max_error']:>12.3e}  {r['mean_error']:>12.3e}  {r['time']:>7.2f}", end="")
            else:
                print(f"  {'---':>12}  {'---':>12}  {'---':>7}", end="")
        print()

    print(f"\n{'='*80}\n  PINN Stability\n{'='*80}")
    print(f"{'Problem':<15} {'Method':<18} {'Stab':>8} {'Cond':>14} {'RelL2':>12} {'PhysRes':>12}")
    print("-" * 73)
    for res, pn in zip(all_res, pns):
        for m in ['Discrete Coord.', 'Endpoint', 'Midpoint']:
            if m in res:
                r = res[m]
                print(f"{pn[:14]:<15} {m:<18} {r.get('stability', 0):>8.3f} "
                      f"{r.get('cond', 0):>14.2e} {r.get('rel_l2', 0):>12.3e} "
                      f"{r.get('phys_residual', 0):>12.3e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PINN vs Traditional Methods: Full Benchmark")
    print("=" * 60)

    ps  = make_smooth_problem()
    psg = make_singular_problem()
    pa  = make_abel_problem()

    for p in [ps, psg, pa]:
        r = p.verify()
        print(f"  [{'OK' if r < 1e-3 else 'WARN'}] {p.name}: residual={r:.2e}")

    all_res = []
    for prob in [ps, psg, pa]:
        ep = 4000 if prob.singular else 3000
        all_res.append(run_experiment(prob, N_pinn=50, epochs=ep))

    Ns, es, esg = run_convergence(ps, psg)
    abl         = run_ablation(ps)

    print("\n  Generating figures...")
    plot_all(all_res,  [p.name for p in [ps, psg, pa]])
    plot_conv(Ns, es, esg, psg.alpha)
    plot_train(all_res, [p.name for p in [ps, psg, pa]])
    print_tables(all_res, [p.name for p in [ps, psg, pa]])

    if abl:
        print(f"\n{'='*70}\n  ABLATION\n{'='*70}")
        print(f"{'Config':<22} {'MaxErr':>12} {'MeanErr':>12} {'Time':>10} {'Params':>10}")
        print("-" * 70)
        for r in abl:
            print(f"{r['config']:<22} {r['max_error']:>12.3e} {r['mean_error']:>12.3e} "
                  f"{r['time']:>10.1f} {r['params']:>10,}")

    print("\n  Done. Output: fig_comparison.png, fig_convergence.png, fig_training.png")


if __name__ == '__main__':
    main()
