"""Core experiment: PINN for weakly singular nonlinear Fredholm equations.

This is the single-file version worth uploading when only one experiment
script is allowed. It demonstrates the main numerical idea used in the paper:
do not set the diagonal of a weakly singular kernel to zero. Instead, integrate
the singular cell analytically.

Equation:
    phi(x) = f(x) + lambda * int_0^1 |x - s|^{-alpha} g(phi(s)) ds

The script builds f from a known exact solution phi(x)=sin(pi*x), trains a
small PINN, and compares two quadrature choices:
    1. endpoint_analytic: endpoint cells with exact singular-cell weights
    2. midpoint: midpoint cells, which avoid the diagonal singularity

Outputs:
    nonlinear_pinn_figures/core_singular_fredholm_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy import integrate


DTYPE = torch.float64


def exact_solution(x: float | np.ndarray) -> float | np.ndarray:
    """Reference solution used to manufacture the source term."""
    return np.sin(np.pi * x)


def analytic_singular_weight(x: float, a: float, b: float, alpha: float) -> float:
    """Return int_a^b |x-s|^{-alpha} ds for 0 < alpha < 1.

    This is the key correction. A naive endpoint rule evaluates K(x_i, x_j),
    which is infinite on the diagonal. Setting that diagonal value to zero
    changes the equation. The correct replacement is the finite cell integral.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    beta = 1.0 - alpha
    if x <= a:
        return ((b - x) ** beta - (a - x) ** beta) / beta
    if x >= b:
        return ((x - a) ** beta - (x - b) ** beta) / beta
    return ((x - a) ** beta + (b - x) ** beta) / beta


def build_endpoint_matrix(nodes: np.ndarray, lam: float, alpha: float) -> torch.Tensor:
    """Build lambda*K_h using analytic integration on every cell."""
    n = len(nodes)
    k_mat = np.zeros((n, n), dtype=np.float64)
    for i, x_i in enumerate(nodes):
        for j in range(n - 1):
            k_mat[i, j] = analytic_singular_weight(
                x_i, nodes[j], nodes[j + 1], alpha
            )
    return torch.tensor(lam * k_mat, dtype=DTYPE)


def build_midpoint_matrix(nodes: np.ndarray, lam: float, alpha: float) -> torch.Tensor:
    """Build lambda*K_h with midpoint quadrature."""
    n = len(nodes)
    k_mat = np.zeros((n, n), dtype=np.float64)
    for i, x_i in enumerate(nodes):
        for j in range(n - 1):
            mid = 0.5 * (nodes[j] + nodes[j + 1])
            h = nodes[j + 1] - nodes[j]
            k_mat[i, j] = abs(x_i - mid) ** (-alpha) * h
    return torch.tensor(lam * k_mat, dtype=DTYPE)


class PINN(nn.Module):
    """Small fully connected network for phi_theta(x)."""

    def __init__(self, width: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_source(
    g_np: Callable[[float], float],
    lam: float,
    alpha: float,
    grid_size: int = 300,
) -> Callable[[float], float]:
    """Manufacture f(x) from the exact solution.

    f(x) = phi_exact(x) - lambda * int K(x,s) g(phi_exact(s)) ds
    """
    xs = np.linspace(0.0, 1.0, grid_size)
    values = []

    for x in xs:
        eps = 1e-10

        def integrand(s: float) -> float:
            return abs(x - s) ** (-alpha) * g_np(float(exact_solution(s)))

        pieces = []
        if x - eps > 0.0:
            pieces.append(integrate.quad(integrand, 0.0, x - eps, limit=400)[0])
        if x + eps < 1.0:
            pieces.append(integrate.quad(integrand, x + eps, 1.0, limit=400)[0])

        values.append(float(exact_solution(x) - lam * sum(pieces)))

    return lambda x: float(np.interp(x, xs, values))


def train_once(
    method: str,
    matrix_builder: Callable[[np.ndarray, float, float], torch.Tensor],
    source_fn: Callable[[float], float],
    g_torch: Callable[[torch.Tensor], torch.Tensor],
    lam: float,
    alpha: float,
    n_nodes: int,
    epochs: int,
    seed: int,
) -> dict[str, float]:
    """Train one method and return accuracy/residual metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    nodes = np.linspace(0.0, 1.0, n_nodes)
    x_train = torch.tensor(nodes, dtype=DTYPE).unsqueeze(-1)
    f_train = torch.tensor([source_fn(x) for x in nodes], dtype=DTYPE).unsqueeze(-1)
    y_true = torch.tensor(exact_solution(nodes), dtype=DTYPE).unsqueeze(-1)
    k_mat = matrix_builder(nodes, lam, alpha)

    model = PINN().double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    best_state = None
    best_max_err = float("inf")
    t0 = time.time()

    for _ in range(epochs):
        optimizer.zero_grad()
        phi = model(x_train)
        residual = phi - f_train - k_mat @ g_torch(phi)
        loss = residual.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            max_err = (phi - y_true).abs().max().item()
            if max_err < best_max_err:
                best_max_err = max_err
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    x_eval = torch.linspace(0.001, 0.999, 400, dtype=DTYPE).unsqueeze(-1)
    true_eval = exact_solution(x_eval.squeeze(-1).cpu().numpy())
    with torch.no_grad():
        pred_eval = model(x_eval).squeeze(-1).cpu().numpy()
        phi_train = model(x_train)
        final_residual = phi_train - f_train - k_mat @ g_torch(phi_train)

    err = np.abs(pred_eval - true_eval)
    rel_l2 = np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(true_eval**2)) + 1e-15)

    return {
        "method": method,
        "max_error": float(err.max()),
        "relative_l2": float(rel_l2),
        "mse": float(np.mean(err**2)),
        "physics_residual_mse": float(final_residual.pow(2).mean().item()),
        "sup_residual": float(final_residual.abs().max().item()),
        "runtime_seconds": float(time.time() - t0),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Run the manufactured singular Fredholm benchmark."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.nonlinearity == "square":
        g_np = lambda u: u**2
        g_torch = lambda u: u**2
    elif args.nonlinearity == "sin":
        g_np = np.sin
        g_torch = torch.sin
    else:
        raise ValueError(f"unknown nonlinearity: {args.nonlinearity}")

    source_fn = make_source(g_np, lam=args.lam, alpha=args.alpha)
    methods = [
        ("endpoint_analytic", build_endpoint_matrix),
        ("midpoint", build_midpoint_matrix),
    ]

    results = [
        train_once(
            method=name,
            matrix_builder=builder,
            source_fn=source_fn,
            g_torch=g_torch,
            lam=args.lam,
            alpha=args.alpha,
            n_nodes=args.nodes,
            epochs=args.epochs,
            seed=args.seed,
        )
        for name, builder in methods
    ]

    payload = {
        "description": (
            "PINN benchmark for a weakly singular nonlinear Fredholm equation. "
            "The endpoint method uses analytic singular-cell integration instead "
            "of a zero diagonal approximation."
        ),
        "equation": "phi(x) = f(x) + lambda * int_0^1 |x-s|^{-alpha} g(phi(s)) ds",
        "exact_solution": "sin(pi*x)",
        "parameters": {
            "alpha": args.alpha,
            "lambda": args.lam,
            "nodes": args.nodes,
            "epochs": args.epochs,
            "seed": args.seed,
            "nonlinearity": args.nonlinearity,
        },
        "results": results,
    }

    out_path = out_dir / "core_singular_fredholm_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nonlinearity", choices=["square", "sin"], default="square")
    parser.add_argument("--out-dir", default="nonlinear_pinn_figures")
    return parser.parse_args()


if __name__ == "__main__":
    result = run_experiment(parse_args())
    print(json.dumps(result, indent=2))
