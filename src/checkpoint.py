"""Checkpoint saving for post-hoc LLC (Local Learning Coefficient) estimation.

Design rationale is recorded in docs/20260907.md. Summary:
  - Weights are saved as plain npz (framework-agnostic; tf.train.Checkpoint would
    tangle a future PyTorch port and doesn't track NeuralNet.X_mean/X_std, which
    are numpy arrays, not tf.Variables).
  - context.npz captures everything that is NOT reproducible after the fact
    (the dataset itself, idx_test/idx_x, dt, scale_c, scale_d, normalization
    stats) so the loss L_n used for LLC can be reconstructed later.
  - Collocation points for the LLC evaluation set are frozen separately from
    the training eqns pipeline, since training draws them online and does not
    define a fixed n.
"""
from __future__ import annotations

import os

import numpy as np


def save_context(
    ckpt_dir: str,
    *,
    C_star: np.ndarray,
    Diff_star,
    X_star: np.ndarray,
    t_star: np.ndarray,
    idx_test: np.ndarray,
    idx_x: np.ndarray,
    dt,
    scale_c: float,
    scale_d,
    c_net,
    D_net,
    layers,
    layers_d,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "context.npz")
    np.savez_compressed(
        path,
        C_star=C_star,
        Diff_star=Diff_star if Diff_star is not None else np.array([]),
        X_star=X_star,
        t_star=t_star,
        idx_test=idx_test,
        idx_x=idx_x,
        dt=np.float32(dt),
        scale_c=np.float32(scale_c),
        scale_d=np.float32(scale_d) if scale_d is not None else np.float32(np.nan),
        c_X_mean=c_net.X_mean,
        c_X_std=c_net.X_std,
        D_X_mean=D_net.X_mean,
        D_X_std=D_net.X_std,
        layers=np.array(layers),
        layers_d=np.array(layers_d),
    )
    return path


def save_llc_eval_set(
    ckpt_dir: str,
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    t_data: np.ndarray,
    c_data: np.ndarray,
    d_data: np.ndarray,
    dt,
    scale_c: float,
    scale_d,
    n_llc_data: int,
    n_llc_eqns: int,
    seed: int = 2024,
) -> str:
    """Freeze the fixed (data, collocation) set used to evaluate L_n for LLC."""
    os.makedirs(ckpt_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    x_eqns = np.float32(rng.uniform(x_data.min(), x_data.max(), (n_llc_eqns, 1)))
    y_eqns = np.float32(rng.uniform(y_data.min(), y_data.max(), (n_llc_eqns, 1)))
    t_eqns = np.float32(rng.uniform(t_data.min(), t_data.max(), (n_llc_eqns, 1)))

    idx_llc = rng.choice(x_data.shape[0], n_llc_data, replace=False)

    path = os.path.join(ckpt_dir, "llc_eval_set.npz")
    np.savez_compressed(
        path,
        x_eqns=x_eqns, y_eqns=y_eqns, t_eqns=t_eqns,
        x_data=x_data[idx_llc], y_data=y_data[idx_llc], t_data=t_data[idx_llc],
        c_data=c_data[idx_llc], d_data=d_data[idx_llc],
        dt=np.float32(dt),
        scale_c=np.float32(scale_c),
        scale_d=np.float32(scale_d) if scale_d is not None else np.float32(np.nan),
        n=np.int64(n_llc_data + n_llc_eqns),
    )
    return path


def save_ckpt(ckpt_dir: str, it: int, c_net, D_net) -> str:
    d = {}
    for name, net in [("c", c_net), ("D", D_net)]:
        for l in range(net.num_layers - 1):
            d[f"{name}_W{l}"] = net.weights[l].numpy()
            d[f"{name}_b{l}"] = net.biases[l].numpy()
            d[f"{name}_g{l}"] = net.gammas[l].numpy()
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{it:07d}.npz")
    np.savez_compressed(path, **d)
    return path


def build_ckpt_steps(n_iter: int, ckpt_every: int, extra_steps) -> set:
    """Log-dense-ish early on, then every `ckpt_every` steps.

    Matches the agreed minimal schedule in docs/20260907.md:
    `it % ckpt_every == 0 or it in extra_steps`.
    """
    linear = set(range(0, n_iter + 1, ckpt_every))
    return linear | set(extra_steps)
