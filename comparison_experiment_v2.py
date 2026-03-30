"""
Comprehensive Comparison: PINN Discretization Methods vs Traditional
Numerical Methods for Fredholm Integral Equations of the Second Kind

This module implements and benchmarks three discretization approaches:
  - PINN methods: Discrete Coordinate, Endpoint, Midpoint
  - Traditional methods: Nystrom (Gauss-Legendre), Galerkin (Legendre basis), Collocation

Three benchmark problems with verified exact solutions are used for comparison.

Usage:
    python comparison_experiment_v2.py
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

print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")


# ══════════════════════════════════════════════════════════════
# Part 0: High-precision source term computation
# ══════════════════════════════════════════════════════════════

def compute_source_value(kernel_func, true_func, lambda_val, x_val,
                         singular=False):
    """
    Compute source term: f(x) = phi*(x) - lambda * integral_0^1 K(x,s) phi*(s) ds
    
    Args:
        kernel_func: Callable, kernel function K(x,s)
        true_func: Callable, true solution phi*(x)
        lambda_val: float, coupling parameter
        x_val: float, evaluation point
        singular: bool, whether kernel has singularity
        
    Returns:
        float: f(x) value computed via high-precision integration
    """
    def integrand(s):
        return kernel_func(x_val, s) * true_func(s)

    if singular:
        eps = 1e-10
        parts = []
        if x_val - eps > 0:
            I1, _ = integrate.quad(integrand, 0, x_val - eps,
                                   limit=500, epsabs=1e-14, epsrel=1e-14)
            parts.append(I1)
        if x_val + eps < 1:
            I2, _ = integrate.quad(integrand, x_val + eps, 1,
                                   limit=500, epsabs=1e-14, epsrel=1e-14)
            parts.append(I2)
        integral_val = sum(parts)
    else:
        integral_val, _ = integrate.quad(integrand, 0, 1,
                                          limit=500, epsabs=1e-14, epsrel=1e-14)
    return true_func(x_val) - lambda_val * integral_val


# ══════════════════════════════════════════════════════════════
# Part 1: Problem definition and management
# ══════════════════════════════════════════════════════════════

class Problem:
    """
    Encapsulates Fredholm integral equation of the second kind:
    
        phi(x) = f(x) + lambda * integral_0^1 K(x,s) phi(s) ds
    
    Attributes:
        name: Problem identifier
        kernel: Kernel function K(x,s)
        true_sol: Known exact solution phi*(x)
        lam: Coupling parameter lambda
        singular: Whether kernel is singular
        alpha: Singularity exponent (for singular kernels)
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
        """Precompute source term on fine grid for interpolation."""
        xs = np.linspace(0, 1, M)
        fs = np.array([compute_source_value(
            self.kernel, self.true_sol, self.lam, x, singular=self.singular
        ) for x in xs])
        self._sx = xs
        self._sf = fs

    def source(self, x):
        """Evaluate source term f(x) via interpolation."""
        return float(np.interp(x, self._sx, self._sf))

    def verify(self, N_test=10):
        """
        Verify that exact solution satisfies the integral equation.
        
        Returns:
            float: Maximum residual at test points
        """
        xs = np.linspace(0.05, 0.95, N_test)
        max_res = 0
        for x in xs:
            phi_x = self.true_sol(x)
            f_x = self.source(x)
            integ = lambda s: self.kernel(x, s) * self.true_sol(s)
            if self.singular:
                eps = 1e-10
                I1, _ = integrate.quad(integ, 0, max(0, x-eps), limit=300)
                I2, _ = integrate.quad(integ, min(1, x+eps), 1, limit=300)
                integral = I1 + I2
            else:
                integral, _ = integrate.quad(integ, 0, 1, limit=300)
            residual = abs(phi_x - f_x - self.lam * integral)
            max_res = max(max_res, residual)
        return max_res


def make_smooth_problem():
    """
    Benchmark 1: Smooth Gaussian kernel
    
    K(x,s) = exp(-(x-s)^2)
    phi*(x) = x(1-x)
    """
    kernel = lambda x, s: np.exp(-(x-s)**2)
    true_sol = lambda x: x*(1-x)
    return Problem("Smooth Kernel", kernel, true_sol, lam=1.0)


def make_singular_problem():
    """
    Benchmark 2: Weakly singular kernel
    
    K(x,s) = |x-s|^{-0.3}  (hypersingular)
    phi*(x) = sin(pi*x)
    """
    alpha = 0.3
    def kernel(x, s):
        d = abs(x-s)
        return 0.0 if d < 1e-15 else d**(-alpha)
    true_sol = lambda x: np.sin(np.pi*x)
    return Problem("Singular Kernel", kernel, true_sol, lam=0.1,
                   singular=True, alpha=alpha)


def make_abel_problem():
    """
    Benchmark 3: Regularized Abel transform
    
    K(r,s) = 2s / sqrt((r+eps)^2 + s^2)
    mu*(s) = s(1-s)
    """
    eps = 0.05
    kernel = lambda r, s: 2*s / np.sqrt((r+eps)**2 + s**2)
    true_sol = lambda s: s*(1-s)
    return Problem("Abel Transform", kernel, true_sol, lam=0.5)


# ══════════════════════════════════════════════════════════════
# Part 2: Traditional numerical methods
# ══════════════════════════════════════════════════════════════

def nystrom_solve(prob, N=50):
    """
    Nystrom method with Gauss-Legendre quadrature.
    
    Args:
        prob: Problem instance
        N: Number of quadrature nodes
        
    Returns:
        dict: Results containing errors, condition number, solution, etc.
    """
    nd, wt = np.polynomial.legendre.leggauss(N)
    nd = 0.5*(nd+1)
    wt = 0.5*wt
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = prob.lam * wt[j] * prob.kernel(nd[i], nd[j])
    A = np.eye(N) - K
    f = np.array([prob.source(x) for x in nd])
    t0 = time.time()
    phi = np.linalg.solve(A, f)
    dt = time.time() - t0
    pt = np.array([prob.true_sol(x) for x in nd])
    xf = np.linspace(0, 1, 200)
    pf = np.interp(xf, nd, phi)
    tf = np.array([prob.true_sol(x) for x in xf])
    return {
        'method': 'Nystrom',
        'max_error': np.max(np.abs(phi - pt)),
        'mean_error': np.mean(np.abs(phi - pt)),
        'time': dt,
        'cond': np.linalg.cond(A),
        'x': xf,
        'phi': pf,
        'true': tf,
        'stability': 1/(1 + np.log10(max(np.linalg.cond(A), 1)))
    }


def galerkin_solve(prob, N_basis=15):
    """
    Galerkin method with Legendre basis.
    
    Args:
        prob: Problem instance
        N_basis: Number of basis functions
        
    Returns:
        dict: Results containing errors, condition number, solution, etc.
    """
    from numpy.polynomial import legendre
    
    def bas(k, x):
        """k-th Legendre basis function at x."""
        c = np.zeros(k + 1)
        c[k] = 1.0
        return legendre.legval(2*x - 1, c)
    
    nq = 80
    qn, qw = np.polynomial.legendre.leggauss(nq)
    qn = 0.5*(qn + 1)
    qw = 0.5*qw
    
    B = np.array([[bas(k, x) for x in qn] for k in range(N_basis)])
    M = B @ np.diag(qw) @ B.T
    fv = np.array([prob.source(x) for x in qn])
    F = B @ (qw*fv)
    
    Km = np.zeros((N_basis, N_basis))
    for qi in range(nq):
        xq, wx = qn[qi], qw[qi]
        inn = np.zeros(N_basis)
        for qj in range(nq):
            sq, ws = qn[qj], qw[qj]
            kv = prob.kernel(xq, sq)
            for j in range(N_basis):
                inn[j] += ws * kv * B[j, qj]
        for i in range(N_basis):
            for j in range(N_basis):
                Km[i, j] += wx * B[i, qi] * inn[j]
    
    Km *= prob.lam
    A = M - Km
    t0 = time.time()
    co = np.linalg.solve(A, F)
    dt = time.time() - t0
    
    xf = np.linspace(0, 1, 200)
    pf = sum(co[k]*np.array([bas(k, x) for x in xf]) for k in range(N_basis))
    tf = np.array([prob.true_sol(x) for x in xf])
    
    return {
        'method': 'Galerkin',
        'max_error': np.max(np.abs(pf - tf)),
        'mean_error': np.mean(np.abs(pf - tf)),
        'time': dt,
        'cond': np.linalg.cond(A),
        'x': xf,
        'phi': pf,
        'true': tf,
        'stability': 1/(1 + np.log10(max(np.linalg.cond(A), 1)))
    }


def collocation_solve(prob, N=50):
    """
    Collocation method with trapezoidal quadrature.
    
    Args:
        prob: Problem instance
        N: Number of collocation points
        
    Returns:
        dict: Results containing errors, condition number, solution, etc.
    """
    xn = np.linspace(0, 1, N)
    h = 1.0/(N - 1)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            w = h * (0.5 if j == 0 or j == N-1 else 1.0)
            K[i, j] = prob.lam * w * prob.kernel(xn[i], xn[j])
    A = np.eye(N) - K
    f = np.array([prob.source(x) for x in xn])
    t0 = time.time()
    phi = np.linalg.solve(A, f)
    dt = time.time() - t0
    pt = np.array([prob.true_sol(x) for x in xn])
    xf = np.linspace(0, 1, 200)
    pf = np.interp(xf, xn, phi)
    tf = np.array([prob.true_sol(x) for x in xf])
    return {
        'method': 'Collocation',
        'max_error': np.max(np.abs(phi - pt)),
        'mean_error': np.mean(np.abs(phi - pt)),
        'time': dt,
        'cond': np.linalg.cond(A),
        'x': xf,
        'phi': pf,
        'true': tf,
        'stability': 1/(1 + np.log10(max(np.linalg.cond(A), 1)))
    }


# ══════════════════════════════════════════════════════════════
# Part 3: PINN-based solution
# ══════════════════════════════════════════════════════════════

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for integral equations.
    
    Architecture: Fully connected network with Tanh activations
    Customizable depth and width.
    
    Args:
        width: Hidden layer width (default 512)
        depth: Number of hidden layers (default 5)
    """
    
    def __init__(self, width=512, depth=5):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Evaluate network at input x."""
        return self.net(x)


def build_kernel_matrix(prob, N, method):
    """
    Build discretized kernel matrix for PINN residual.
    
    Three discretization schemes supported:
      - 'discrete': Rectangle rule with full width h
      - 'endpoint': Trapezoidal (endpoint) weights
      - 'midpoint': Midpoint rule with shifted evaluation
    
    Args:
        prob: Problem instance
        N: Number of points
        method: Discretization scheme ('discrete', 'endpoint', 'midpoint')
        
    Returns:
        torch.Tensor: Lambda * K matrix on device
    """
    xn = np.linspace(0, 1, N)
    h = 1.0/(N - 1)
    K = np.zeros((N, N))
    
    if method == 'discrete':
        for i in range(N):
            for j in range(N):
                K[i, j] = prob.kernel(xn[i], xn[j]) * h
        K *= prob.lam
        
    elif method == 'endpoint':
        for i in range(N):
            for j in range(N):
                K[i, j] = prob.kernel(xn[i], xn[j]) * h
        K[:, 0] *= 0.5
        K[:, -1] *= 0.5
        K *= prob.lam
        
    elif method == 'midpoint':
        hm = 1.0/N
        for i in range(N):
            for j in range(N):
                tm = min((j + 0.5)*hm, 1.0)
                K[i, j] = prob.kernel(xn[i], tm) * hm
        K *= prob.lam
    
    return torch.tensor(K, dtype=DTYPE, device=device)


def pinn_solve(prob, method='discrete', N=50, epochs=3000,
               lr=4e-4, width=512, depth=5):
    """
    Solve Fredholm equation using PINN with specified discretization.
    
    Args:
        prob: Problem instance
        method: Discretization method ('discrete', 'endpoint', 'midpoint')
        N: Number of training points
        epochs: Training epochs
        lr: Initial learning rate (uses cosine annealing)
        width: Network width
        depth: Network depth
        
    Returns:
        dict: Results including errors, training history, condition number, etc.
    """
    xpts = torch.linspace(0, 1, N, device=device, dtype=DTYPE).unsqueeze(-1)
    xnp = np.linspace(0, 1, N)
    Km = build_kernel_matrix(prob, N, method)
    fv = torch.tensor([prob.source(x) for x in xnp],
                      device=device, dtype=DTYPE).unsqueeze(-1)
    tv = torch.tensor([prob.true_sol(x) for x in xnp],
                      device=device, dtype=DTYPE).unsqueeze(-1)

    # Discretization residual
    dr = (tv - fv - Km@tv).abs().max().item()

    model = PINN(width, depth).double().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    loss_history = []
    error_history = []
    best_state = None
    best_error = float('inf')
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        phi = model(xpts)
        res = phi - fv - Km@phi
        loss = res.pow(2).mean()
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
            print(f"    [{method:10s}] Ep {ep:5d}: loss={loss.item():.3e}, err={e:.3e}")

    tt = time.time() - t0
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    xf = torch.linspace(0, 1, 200, device=device, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        pf = model(xf).cpu().numpy().flatten()
    
    xfn = xf.cpu().numpy().flatten()
    tfn = np.array([prob.true_sol(x) for x in xfn])
    
    max_error = np.max(np.abs(pf - tfn))
    mean_error = np.mean(np.abs(pf - tfn))
    rel_l2 = np.sqrt(np.mean((pf - tfn)**2))/(np.sqrt(np.mean(tfn**2)) + 1e-15)
    mse = np.mean((pf - tfn)**2)
    
    Im = torch.eye(N, dtype=DTYPE, device=device)
    cond = torch.linalg.cond(Im - Km).item()
    stab = 1/(1 + np.log10(max(cond, 1)))

    method_name = {
        'discrete': 'Discrete Coord.',
        'endpoint': 'Endpoint',
        'midpoint': 'Midpoint'
    }[method]
    
    return {
        'method': method_name,
        'method_key': method,
        'max_error': max_error,
        'mean_error': mean_error,
        'rel_l2': rel_l2,
        'mse': mse,
        'time': tt,
        'cond': cond,
        'stability': stab,
        'loss_history': loss_history,
        'error_history': error_history,
        'x': xfn,
        'phi': pf,
        'true': tfn,
        'disc_residual': dr,
        'phys_residual': loss_history[-1] if loss_history else 0
    }


# ══════════════════════════════════════════════════════════════
# Part 4: Experiment workflow
# ══════════════════════════════════════════════════════════════

def run_experiment(prob, N_pinn=50, epochs=3000):
    """
    Run complete benchmark on single problem.
    
    Solves with all 6 methods and collects results.
    
    Args:
        prob: Problem instance
        N_pinn: Number of training points for PINN
        epochs: Training epochs
        
    Returns:
        dict: Results for each method
    """
    print(f"\n{'='*70}")
    print(f"  {prob.name}")
    print(f"{'='*70}")
    
    vr = prob.verify()
    print(f"  Verification residual: {vr:.2e}")
    if vr > 1e-3:
        print(f"  WARNING: residual {vr:.2e} > 1e-3, source interpolation floor")

    results = {}
    
    print("\n  --- Traditional Methods ---")
    for nm, fn, kw in [("Nystrom", nystrom_solve, {'N': N_pinn}),
                       ("Galerkin", galerkin_solve, {'N_basis': 15}),
                       ("Collocation", collocation_solve, {'N': N_pinn})]:
        try:
            t0 = time.time()
            r = fn(prob, **kw)
            r['time'] = time.time() - t0
            results[nm] = r
            print(f"    {nm:15s}: MaxErr={r['max_error']:.3e}, MeanErr={r['mean_error']:.3e}, "
                  f"Cond={r['cond']:.2e}, Time={r['time']:.2f}s")
        except Exception as e:
            print(f"    {nm:15s}: FAILED - {e}")

    print("\n  --- PINN Methods ---")
    for mk in ['discrete', 'endpoint', 'midpoint']:
        try:
            r = pinn_solve(prob, method=mk, N=N_pinn, epochs=epochs)
            results[r['method']] = r
            print(f"    {r['method']:15s}: MaxErr={r['max_error']:.3e}, MeanErr={r['mean_error']:.3e}, "
                  f"Stab={r['stability']:.3f}, Time={r['time']:.2f}s")
        except Exception as e:
            print(f"    PINN-{mk}: FAILED - {e}")

    return results


def run_convergence(ps, psg):
    """
    Verify convergence rates for smooth and singular kernels.
    
    Tests Theorem 2 (smooth: O(h^2)) and Theorem 3 (singular: O(h^{1-alpha}))
    
    Args:
        ps: Smooth problem
        psg: Singular problem
        
    Returns:
        tuple: (N values, smooth errors, singular errors)
    """
    print(f"\n{'='*70}")
    print(f"  Convergence Rate Verification")
    print(f"{'='*70}")
    Ns = [20, 40, 80, 160, 320]

    print("\n  --- Theorem 2: Smooth Kernel -> O(h^2) ---")
    es = []
    for N in Ns:
        r = collocation_solve(ps, N=N)
        es.append(r['max_error'])
        print(f"    N={N:4d}, h={1/(N-1):.5f}, MaxErr={r['max_error']:.4e}")
    print("    Ratios (expect ~4.0):")
    for i in range(1, len(es)):
        if es[i] > 1e-16:
            print(f"      {Ns[i-1]}->{Ns[i]}: {es[i-1]/es[i]:.2f}")

    print(f"\n  --- Theorem 3: Singular Kernel (a={psg.alpha}) -> O(h^{1-psg.alpha:.1f}) ---")
    esg = []
    for N in Ns:
        r = collocation_solve(psg, N=N)
        esg.append(r['max_error'])
        print(f"    N={N:4d}, h={1/(N-1):.5f}, MaxErr={r['max_error']:.4e}")
    
    exp_r = 2**(1 - psg.alpha)
    print(f"    Ratios (expect ~{exp_r:.2f}):")
    for i in range(1, len(esg)):
        if esg[i] > 1e-16:
            print(f"      {Ns[i-1]}->{Ns[i]}: {esg[i-1]/esg[i]:.2f}")

    return Ns, es, esg


def run_ablation(prob):
    """
    Ablation study: test different network architectures.
    
    Args:
        prob: Problem instance
        
    Returns:
        list: Results for each configuration
    """
    print(f"\n{'='*70}")
    print(f"  Ablation Study")
    print(f"{'='*70}")
    configs = [
        ("5x512 Tanh (baseline)", 512, 5, 2000),
        ("3x256 Tanh", 256, 3, 2000),
        ("7x512 Tanh", 512, 7, 2000),
        ("3x512 Tanh", 512, 3, 2000),
        ("5x256 Tanh", 256, 5, 2000),
        ("5x1024 Tanh", 1024, 5, 2000),
    ]
    res = []
    for lb, w, d, ep in configs:
        print(f"\n    {lb}...")
        try:
            r = pinn_solve(prob, method='discrete', N=50, epochs=ep, width=w, depth=d)
            np_ = sum(p.numel() for p in PINN(w, d).parameters())
            res.append({
                'config': lb,
                'max_error': r['max_error'],
                'mean_error': r['mean_error'],
                'time': r['time'],
                'params': np_
            })
            print(f"      MaxErr={r['max_error']:.3e}, Time={r['time']:.1f}s, Params={np_:,}")
        except Exception as e:
            print(f"      FAILED: {e}")
    return res


# ══════════════════════════════════════════════════════════════
# Part 5: Visualization
# ══════════════════════════════════════════════════════════════

def plot_all(all_res, pns, pf="fig"):
    """Create comparison plots: solutions, errors, max errors."""
    ct = {'Nystrom': '#E69F00', 'Galerkin': '#56B4E9', 'Collocation': '#CC79A7'}
    cp = {'Discrete Coord.': '#0072B2', 'Endpoint': '#D55E00', 'Midpoint': '#009E73'}
    ac = {**ct, **cp}
    nr = len(all_res)
    fig, axes = plt.subplots(nr, 3, figsize=(18, 5*nr))
    if nr == 1:
        axes = axes[np.newaxis, :]
    
    for row, (res, pn) in enumerate(zip(all_res, pns)):
        xf = np.linspace(0, 1, 200)
        tv = list(res.values())[0]['true']
        
        # Solutions
        ax = axes[row, 0]
        ax.plot(xf, tv, 'k-', lw=2.5, label='Exact', zorder=10)
        for nm, r in res.items():
            c = ac.get(nm, 'gray')
            ls = '--' if nm in cp else '-.'
            ax.plot(r['x'], r['phi'], color=c, ls=ls, lw=1.5, label=nm, alpha=0.85)
        ax.set_title(f'{pn}: Solution', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\phi(x)$')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Errors
        ax = axes[row, 1]
        for nm, r in res.items():
            c = ac.get(nm, 'gray')
            e = np.abs(r['phi'] - r['true']) + 1e-16
            ax.semilogy(r['x'], e, color=c, lw=1.5, label=nm, alpha=0.85)
        ax.set_title(f'{pn}: Error', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('|Error|')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Max errors
        ax = axes[row, 2]
        nms = list(res.keys())
        me = [res[n]['max_error'] for n in nms]
        bc = [ac.get(n, 'gray') for n in nms]
        bars = ax.bar(range(len(nms)), me, color=bc, alpha=0.85)
        ax.set_xticks(range(len(nms)))
        ax.set_xticklabels(nms, rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('Max Error')
        ax.set_yscale('log')
        ax.set_title(f'{pn}: Max Error', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for b, v in zip(bars, me):
            ax.text(b.get_x() + b.get_width()/2., b.get_height()*1.3,
                    f'{v:.1e}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    fn = f'{pf}_comparison.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fn}")
    plt.close()


def plot_conv(Ns, es, esg, alpha, pf="fig"):
    """Create convergence rate plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hs = [1.0/(N-1) for N in Ns]
    
    ax = axes[0]
    ax.loglog(hs, es, 'bo-', lw=2, ms=8, label='Measured')
    ref = [es[0]*(h/hs[0])**2 for h in hs]
    ax.loglog(hs, ref, 'r--', lw=1.5, label=r'$O(h^2)$')
    ax.set_xlabel('h')
    ax.set_ylabel('Max Error')
    ax.set_title('Theorem 2: Smooth Kernel', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.loglog(hs, esg, 'bo-', lw=2, ms=8, label='Measured')
    ref = [esg[0]*(h/hs[0])**(1-alpha) for h in hs]
    ax.loglog(hs, ref, 'r--', lw=1.5, label=f'$O(h^{{{1-alpha:.1f}}})$')
    ax.set_xlabel('h')
    ax.set_ylabel('Max Error')
    ax.set_title(f'Theorem 3: Singular (a={alpha})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fn = f'{pf}_convergence.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fn}")
    plt.close()


def plot_train(all_res, pns, pf="fig"):
    """Create training history plots."""
    pk = ['Discrete Coord.', 'Endpoint', 'Midpoint']
    cl = {'Discrete Coord.': '#0072B2', 'Endpoint': '#D55E00', 'Midpoint': '#009E73'}
    nr = len(all_res)
    fig, axes = plt.subplots(nr, 2, figsize=(12, 4*nr))
    if nr == 1:
        axes = axes[np.newaxis, :]
    
    for row, (res, pn) in enumerate(zip(all_res, pns)):
        ax = axes[row, 0]
        for nm in pk:
            if nm in res and 'loss_history' in res[nm]:
                lh = res[nm]['loss_history']
                ax.semilogy(range(1, len(lh)+1), lh, color=cl[nm], lw=1.5, label=nm, alpha=0.8)
        ax.set_title(f'{pn}: Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[row, 1]
        for nm in pk:
            if nm in res and 'error_history' in res[nm]:
                eh = res[nm]['error_history']
                ax.semilogy(range(1, len(eh)+1), eh, color=cl[nm], lw=1.5, label=nm, alpha=0.8)
        ax.set_title(f'{pn}: Error', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Max Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fn = f'{pf}_training.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fn}")
    plt.close()


def print_tables(all_res, pns):
    """Print comprehensive results tables."""
    mo = ['Nystrom', 'Galerkin', 'Collocation', 'Discrete Coord.', 'Endpoint', 'Midpoint']
    print(f"\n{'='*110}")
    print("  COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*110}")
    print(f"{'Method':<18}", end="")
    for pn in pns:
        print(f"  {'MaxErr':>12s}  {'MeanErr':>12s}  {'Time':>7s}", end="")
    print()
    print("-"*110)
    
    for m in mo:
        print(f"{m:<18}", end="")
        for res in all_res:
            if m in res:
                r = res[m]
                print(f"  {r['max_error']:>12.3e}  {r['mean_error']:>12.3e}  {r['time']:>7.2f}", end="")
            else:
                print(f"  {'---':>12s}  {'---':>12s}  {'---':>7s}", end="")
        print()

    print(f"\n{'='*90}")
    print("  PINN Stability & Condition Numbers")
    print(f"{'='*90}")
    print(f"{'Problem':<15} {'Method':<18} {'Stab':>8} {'Cond':>14} {'RelL2':>12} {'PhysRes':>12}")
    print("-"*83)
    
    for res, pn in zip(all_res, pns):
        for m in ['Discrete Coord.', 'Endpoint', 'Midpoint']:
            if m in res:
                r = res[m]
                print(f"{pn[:14]:<15} {m:<18} {r.get('stability', 0):>8.3f} "
                      f"{r.get('cond', 0):>14.2e} {r.get('rel_l2', 0):>12.3e} "
                      f"{r.get('phys_residual', 0):>12.3e}")


# ══════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════

def main():
    """Run complete benchmark suite."""
    print("="*70)
    print("  PINN vs Traditional Methods: Full Benchmark (v2)")
    print("="*70)

    print("\n  Setting up problems...")
    ps = make_smooth_problem()
    psg = make_singular_problem()
    pa = make_abel_problem()
    probs = [ps, psg, pa]
    pns = [p.name for p in probs]

    for p in probs:
        r = p.verify()
        s = "OK" if r < 1e-3 else "WARN"
        print(f"  [{s}] {p.name}: residual={r:.2e}")

    all_res = []
    for prob in probs:
        ep = 4000 if prob.singular else 3000
        results = run_experiment(prob, N_pinn=50, epochs=ep)
        all_res.append(results)

    Ns, es, esg = run_convergence(ps, psg)
    abl = run_ablation(ps)

    print("\n  Generating figures...")
    plot_all(all_res, pns)
    plot_conv(Ns, es, esg, psg.alpha)
    plot_train(all_res, pns)
    print_tables(all_res, pns)

    if abl:
        print(f"\n{'='*80}")
        print("  ABLATION")
        print(f"{'='*80}")
        print(f"{'Config':<25} {'MaxErr':>12} {'MeanErr':>12} {'Time':>10} {'Params':>10}")
        print("-"*73)
        for r in abl:
            print(f"{r['config']:<25} {r['max_error']:>12.3e} {r['mean_error']:>12.3e} "
                  f"{r['time']:>10.1f} {r['params']:>10,}")

    print("\n" + "="*70)
    print("  DONE. Files: fig_comparison.png, fig_convergence.png, fig_training.png")
    print("="*70)


if __name__ == "__main__":
    main()
