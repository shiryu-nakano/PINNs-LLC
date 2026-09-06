"""Data download and preprocessing for the spatially-varying-diffusion PINN.

download_data() always writes under a caller-supplied data_dir (see src/config.py
for the default, which points at the shared data server, not the repo checkout).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import gdown
import numpy as np
import scipy.io

# Google Drive file ids for the four cases shipped with the original notebook.
CASE_FILE_IDS = {
    1: "1akSDShRp5w5iryi_pYWerg3QqvdMGoiG",
    2: "1wuRcFT82sKLlBvtcHdOhv0xWdAfWCvTx",
    3: "1zEfx76C67EE351ZzGWNXv4-Yfn-6B3P8",
    4: "10Xdxi8Hbi8RsKKuMxsBqJxNr-FOdCqr-",
}


def download_data(data_dir: str, case: int, filename: str = "data.mat") -> str:
    """Download (once) the .mat file for `case` into data_dir/case{case}/filename."""
    case_dir = os.path.join(data_dir, f"case{case}")
    os.makedirs(case_dir, exist_ok=True)
    dest = os.path.join(case_dir, filename)
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?id={CASE_FILE_IDS[case]}"
    gdown.download(url, dest, quiet=False)
    return dest


@dataclass
class Dataset:
    x_data: np.ndarray
    y_data: np.ndarray
    t_data: np.ndarray
    c_data: np.ndarray
    d_data: np.ndarray
    x_eqns: np.ndarray
    y_eqns: np.ndarray
    t_eqns: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    t_test: np.ndarray
    c_test: np.ndarray
    d_test: np.ndarray
    dt: np.ndarray
    idx_x: np.ndarray
    idx_test: np.ndarray
    C_star: np.ndarray
    Diff_star: Optional[np.ndarray]
    X_star: np.ndarray
    t_star: np.ndarray
    known_diffusion: bool


def load_dataset(
    data_path: str,
    *,
    n_eqns: int,
    test_fraction: float,
    seed: int,
    known_diffusion: bool = True,
) -> Dataset:
    """Load a .mat file and reproduce the preprocessing from PINNs_SVD.ipynb.

    When known_diffusion is False, D-related arrays are zero placeholders
    (kept so downstream tf.data pipelines have a stable shape) and the
    corresponding loss/eval terms are skipped by the caller.
    """
    np.random.seed(seed)

    mat = scipy.io.loadmat(data_path)
    C_star = mat["C_star"]
    t_star = mat["t"].T
    X_star = mat["X_star"]
    Diff_star = mat["Diff_star"] if known_diffusion else None

    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    N = x_star.shape[0]
    T = t_star.shape[0]

    x_mesh = np.tile(x_star, (1, T)).flatten()[:, None]
    y_mesh = np.tile(y_star, (1, T)).flatten()[:, None]
    t_mesh = np.tile(t_star, (1, N)).T.flatten()[:, None]

    c_mesh = C_star.flatten()[:, None]
    d_mesh = Diff_star.flatten()[:, None] if known_diffusion else np.zeros_like(c_mesh)

    idx_x = np.random.choice(x_mesh.shape[0], x_mesh.shape[0], replace=False)

    x_data = np.float32(x_mesh[idx_x, :])
    y_data = np.float32(y_mesh[idx_x, :])
    t_data = np.float32(t_mesh[idx_x, :])
    c_data = np.float32(c_mesh[idx_x, :])
    d_data = np.float32(d_mesh[idx_x, :])

    dt = t_star[1] - t_star[0]

    idx_test = np.random.choice(N * T, int(test_fraction * N * T), replace=False)

    t_eqns = np.float32(np.random.uniform(t_data.min(), t_data.max(), size=(n_eqns, 1)))
    x_eqns = np.float32(np.random.uniform(x_data.min(), x_data.max(), size=(n_eqns, 1)))
    y_eqns = np.float32(np.random.uniform(y_data.min(), y_data.max(), size=(n_eqns, 1)))

    t_test = np.float32(t_mesh[idx_test, :])
    x_test = np.float32(x_mesh[idx_test, :])
    y_test = np.float32(y_mesh[idx_test, :])
    c_test = np.float32(c_mesh[idx_test, :])
    d_test = np.float32(d_mesh[idx_test, :])

    return Dataset(
        x_data=x_data, y_data=y_data, t_data=t_data, c_data=c_data, d_data=d_data,
        x_eqns=x_eqns, y_eqns=y_eqns, t_eqns=t_eqns,
        x_test=x_test, y_test=y_test, t_test=t_test, c_test=c_test, d_test=d_test,
        dt=dt, idx_x=idx_x, idx_test=idx_test,
        C_star=C_star, Diff_star=Diff_star, X_star=X_star, t_star=t_star,
        known_diffusion=known_diffusion,
    )
