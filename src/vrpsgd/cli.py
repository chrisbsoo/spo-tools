"""Command-line entry point: `vrpsgd optimize AAPL MSFT GOOGL --start 2020-01-01 --end 2025-01-01`."""

from __future__ import annotations

import argparse

from vrpsgd.data import load_return_matrix
from vrpsgd.objective import ObjectiveConfig
from vrpsgd.optimizers import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vrpsgd")
    sub = parser.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize", help="Run one or all algorithms on a set of tickers.")
    opt.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT GOOGL")
    opt.add_argument("--start", required=True)
    opt.add_argument("--end", required=True)
    opt.add_argument("--algorithm", choices=[*REGISTRY.keys(), "all"], default="all")
    opt.add_argument("--gamma", type=float, default=2.0)
    opt.add_argument("--lam", type=float, default=0.01)
    opt.add_argument("--t-max", type=int, default=3000)
    opt.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "optimize":
        R, tickers = load_return_matrix(args.tickers, args.start, args.end)
        cfg = ObjectiveConfig(gamma=args.gamma, lam=args.lam)
        names = list(REGISTRY.keys()) if args.algorithm == "all" else [args.algorithm]

        print(f"{R.shape[0]} trading days x {R.shape[1]} assets\n")
        for name in names:
            result = REGISTRY[name](cfg, seed=args.seed).fit(R, T_max=args.t_max)
            top = sorted(zip(tickers, result.weights, strict=True), key=lambda x: -x[1])[:5]
            print(f"[{result.algorithm}] objective={result.history.objective[-1]:.6f} "
                  f"non-zero={result.n_nonzero}/{len(tickers)} time={result.wall_time_seconds:.2f}s")
            print("  top holdings:", ", ".join(f"{t}={w:.3f}" for t, w in top if w > 1e-6))
        print()


if __name__ == "__main__":
    main()
