"""Generate a synthetic stand-in data.mat for a case (see src/synthetic_data.py
for what this is and is not, and why it exists).

Writes to the exact path src/data.py's download_data() expects
(data_dir/case{N}/data.mat), so run/train.py will find it and skip the
(currently broken) Google Drive download entirely -- no other changes needed.

Usage:
    python run/make_synthetic_data.py --case 1
    python run/make_synthetic_data.py --case 4 --nx 64 --ny 64 --nt 50  # faster/coarser
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_DATA_DIR  # noqa: E402
from src.synthetic_data import save_case  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nt", type=int, default=100)
    args = parser.parse_args(argv)

    dest = Path(args.data_dir) / f"case{args.case}" / "data.mat"
    path = save_case(args.case, str(dest), nx=args.nx, ny=args.ny, nt=args.nt)
    print(f"wrote synthetic data for case {args.case} to {path}")


if __name__ == "__main__":
    main()
