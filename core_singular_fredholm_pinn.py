"""Core experiment: PINN for weakly singular nonlinear Fredholm equations.

This is the single-file version worth uploading when only one experiment
script is allowed. It demonstrates the main numerical idea used in the paper:
do not set the diagonal of a weakly singular kernel to zero. Instead, integrate
the singular cell analytically.

Equation:
    phi(x) = f(x) + lambda * int_0^1 |x - s|^{-alpha} g(phi(s)) ds

The script builds f from a known exact solution phi(x)=sin(pi*x), trains a
small PINN, and compares three residual assembly choices:
    1. endpoint_zero: naive endpoint rule with zero diagonal (incorrect)
    2. endpoint_analytic: endpoint cells with exact singular-cell weights
    3. midpoint_kernel: simplified midpoint-kernel baseline

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
    # N nodes define N-1 cells, so the last column is unused and remains zero.
    # It is kept only so the matrix can multiply g(phi(nodes)) directly.
    for i, x_i in enumerate(nodes):
        for j in range(n - 1):
            k_mat[i, j] = analytic_singular_weight(
                x_i, nodes[j], nodes[j + 1], alpha
            )
    return torch.tensor(lam * k_mat, dtype=DTYPE)


def build_endpoint_zero_matrix(nodes: np.ndarray, lam: float, alpha: float) -> torch.Tensor:
    """Naive endpoint rule with zero diagonal.

    This is the intentionally incorrect baseline used to expose the
    zero-diagonal pitfall for weakly singular kernels.
    """
    n = len(nodes)
    h = nodes[1] - nodes[0]
    k_mat = np.zeros((n, n), dtype=np.float64)

    for i, x_i in enumerate(nodes):
        for j, s_j in enumerate(nodes):
            if i == j:
                k_mat[i, j] = 0.0
            else:
                k_mat[i, j] = abs(x_i - s_j) ** (-alpha) * h

    return torch.tensor(lam * k_mat, dtype=DTYPE)


def build_midpoint_kernel_matrix(
    nodes: np.ndarray, lam: float, alpha: float
) -> torch.Tensor:
    """Build lambda*K_h with kernel values evaluated at cell midpoints.

    This simplified midpoint baseline evaluates the singular kernel at cell
    midpoints while still using endpoint samples of g(phi). A full midpoint
