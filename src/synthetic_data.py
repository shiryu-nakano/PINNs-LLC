"""Synthetic stand-in dataset generator.

The original benchmark data (data.mat for Case 1-4) was distributed via
Google Drive links embedded in PINNs_SVD.ipynb. As of 2026-09, all four
files return "file does not exist" from Google Drive, and the upstream
repo (github.com/ArezooArdekani/spatially_varying_diffusion) has no
alternative archive linked. This module reconstructs a *physically
consistent stand-in* for that data from the source paper:

    Thakur, Esmaili, Libring, Solorio, Ardekani, "Inverse resolution of
    spatially varying diffusion coefficient using Physics-Informed neural
    networks", Physics of Fluids 36, 081915 (2024). arXiv:2403.03970.

The paper (eq. 3) gives the governing PDE:

    dC/dt = D_x*C_x + D_y*C_y + D*(C_xx + C_yy)

on the unit square x,y in (0,1), t in (0,1), and four exact D_true(x,y)
definitions (Case I-IV below). It does NOT state (in the text available
to us) the initial condition or boundary condition used to produce the
ground truth (which was generated with OpenFOAM). So this module is NOT a
reproduction of the paper's benchmark data -- only the domain, D(x,y), and
PDE are from the paper. The initial condition (a centered Gaussian blob)
and boundary condition (no-flux / zero-gradient) below are our own
reasonable choice, made solely so run/train.py's pipeline (data loading,
training loop, checkpointing, paths) can be exercised end-to-end while the
real data is unavailable. Swap in the real data.mat if/when it resurfaces.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.io


def d_true(case: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """D_true(x, y) for Case 1-4, as given in the paper (eq. after Table/Fig
    describing the benchmarks)."""
    if case == 1:
        return 0.05 + 0.1 * (x * (1 - x) + y * (1 - y))
    if case == 2:
        return 0.25 + 0.1 * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))
    if case == 3:
        return 0.25 + 0.1 * (np.sin(4 * np.pi * x) + np.sin(4 * np.pi * y))
    if case == 4:
        return 0.3 + 0.15 * np.tanh(20 * (x - 0.5)) + 0.1 * np.tanh(20 * (y - 0.35))
    raise ValueError(f"unknown case {case}")


def _grad_no_flux(C: np.ndarray, dx: float, dy: float):
    Cp = np.pad(C, 1, mode="edge")  # zero-gradient (Neumann) boundary
    c_x = (Cp[1:-1, 2:] - Cp[1:-1, :-2]) / (2 * dx)
    c_y = (Cp[2:, 1:-1] - Cp[:-2, 1:-1]) / (2 * dy)
    return c_x, c_y


def _laplacian_no_flux(C: np.ndarray, dx: float, dy: float):
    Cp = np.pad(C, 1, mode="edge")
    c_xx = (Cp[1:-1, 2:] - 2 * Cp[1:-1, 1:-1] + Cp[1:-1, :-2]) / dx ** 2
    c_yy = (Cp[2:, 1:-1] - 2 * Cp[1:-1, 1:-1] + Cp[:-2, 1:-1]) / dy ** 2
    return c_xx, c_yy


def generate_case(
    case: int,
    *,
    nx: int = 128,
    ny: int = 128,
    nt: int = 100,
    t_end: float = 1.0,
    cfl: float = 0.2,
) -> dict:
    """Forward-simulate the paper's PDE with the paper's D(x,y) for `case`,
    from a centered Gaussian initial blob, with no-flux boundaries.

    Returns a dict with the same keys/shapes PINNs_SVD.ipynb expects to
    scipy.io.loadmat: C_star (N,T), Diff_star (N,T), X_star (N,2), t (1,T).
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    D = d_true(case, X, Y)
    D_x, D_y = _grad_no_flux(D, dx, dy)

    C = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.08 ** 2))

    dt_stable = cfl * min(dx, dy) ** 2 / D.max()
    n_substeps = max(1, int(np.ceil(t_end / dt_stable)))
    dt = t_end / n_substeps
    snapshot_times = np.linspace(0.0, t_end, nt)

    snapshots = np.empty((nt, ny, nx), dtype=np.float64)
    snapshots[0] = C
    t = 0.0
    next_snap = 1
    for _ in range(1, n_substeps + 1):
        c_x, c_y = _grad_no_flux(C, dx, dy)
        c_xx, c_yy = _laplacian_no_flux(C, dx, dy)
        C = C + dt * (D_x * c_x + D_y * c_y + D * (c_xx + c_yy))
        t += dt
        while next_snap < nt and t >= snapshot_times[next_snap] - 1e-9:
            snapshots[next_snap] = C
            next_snap += 1
    while next_snap < nt:
        snapshots[next_snap] = C
        next_snap += 1

    N = nx * ny
    X_star = np.column_stack([X.ravel(order="C"), Y.ravel(order="C")])
    C_star = snapshots.reshape(nt, N).T  # N x T
    Diff_star = np.tile(D.ravel(order="C")[:, None], (1, nt))  # N x T
    t_star = snapshot_times[None, :]  # 1 x T

    return {"C_star": C_star, "Diff_star": Diff_star, "X_star": X_star, "t": t_star}


def save_case(case: int, dest: str, **kwargs) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    scipy.io.savemat(dest, generate_case(case, **kwargs))
    return dest
