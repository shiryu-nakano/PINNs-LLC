"""Run configuration for training.

Precedence (highest wins): CLI args > --config YAML file > defaults below.
No environment variables are read on purpose (per project preference) --
paths are always explicit, either as CLI flags or in a config file.

Defaults point at the shared data server path, which is identical on both
GPU hosts (s29, s23) -- see the "infra_data_storage" memory / README.md.
The repo checkout itself should never receive downloaded data or checkpoints.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_DATA_DIR = "/home/nakano/server/PINNs-LLC/data"
DEFAULT_CKPT_DIR = "/home/nakano/server/PINNs-LLC/ckpts"


@dataclass
class TrainConfig:
    case: int = 1
    run_name: Optional[str] = None

    data_dir: str = DEFAULT_DATA_DIR
    ckpt_dir: str = DEFAULT_CKPT_DIR

    seed: int = 1234
    n_iter: int = 320_000
    batch_size: int = 256
    eta_min: float = 2.5e-6
    eta_max: float = 2.5e-3
    n_eqns: int = 256_000_000
    test_fraction: float = 0.1

    layers: List[int] = field(default_factory=lambda: [3] + 8 * [64] + [1])
    layers_d: List[int] = field(default_factory=lambda: [2] + 2 * [8] + [1])

    known_diffusion: bool = True

    log_every: int = 10
    eval_every: int = 1000

    ckpt_every: int = 5000
    ckpt_extra_steps: List[int] = field(default_factory=lambda: [1, 10, 100, 500, 1000, 2000])

    n_llc_data: int = 100_000
    n_llc_eqns: int = 100_000

    def resolved_run_name(self) -> str:
        return self.run_name or f"case{self.case}"

    def run_ckpt_dir(self) -> str:
        return str(Path(self.ckpt_dir) / self.resolved_run_name())


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_config(argv: Optional[List[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the spatially-varying-diffusion PINN")
    parser.add_argument("--config", type=str, default=None, help="YAML file overriding defaults")

    parser.add_argument("--case", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help=f"default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--ckpt-dir", type=str, default=None, help=f"default: {DEFAULT_CKPT_DIR}")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--eta-max", type=float, default=None)
    parser.add_argument("--n-eqns", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)

    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)

    parser.add_argument("--known-diffusion", dest="known_diffusion", action="store_true", default=None)
    parser.add_argument("--no-known-diffusion", dest="known_diffusion", action="store_false")

    args = parser.parse_args(argv)

    cfg_dict = dataclasses.asdict(TrainConfig())
    if args.config:
        cfg_dict.update(_load_yaml(args.config))

    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg_dict.update(cli_overrides)

    return TrainConfig(**cfg_dict)
