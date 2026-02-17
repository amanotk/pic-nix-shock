#!/usr/bin/env python
"""Compare CG and MGCG for shock.ohm."""

import argparse
import pathlib
import sys
import time

import numpy as np
import pyamg

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shock import ohm


def _build_case(nx, ny, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(nx)
    y = np.arange(ny)
    xx = np.broadcast_to(x, (ny, nx))
    yy = np.broadcast_to(y[:, None], (ny, nx))

    L = 1.0 + 0.1 * np.cos(2.0 * np.pi * xx / nx) + 0.1 * np.sin(2.0 * np.pi * yy / ny)
    S = rng.normal(size=(ny, nx, 3))
    return L, S


def _run_solver(nx, ny, c, delta, solver, repeats, seed):
    elapsed_total = []
    elapsed_a1 = []
    elapsed_a2 = []
    iter_1 = []
    iter_2 = []
    status_1 = []
    status_2 = []
    precond_setup_time = 0.0

    c2 = c * c
    c2_dx2 = c2 / (delta * delta)
    c2_dx4 = c2 / (4.0 * delta * delta)

    def solve_block(A, b, M=None):
        if solver in ("cg", "mgcg"):
            return ohm._solve_with_cg(A, b, M=M, maxiter=1000, rtol=1.0e-12)
        raise ValueError(f"Unsupported solver: {solver}")

    M1_cached = None
    M2_cached = None

    for i in range(repeats):
        L, S = _build_case(nx, ny, seed + i)
        base1, base2 = ohm.build_ohm_bases(nx, ny, c2_dx2, c2_dx4)
        A_1 = ohm.assemble_matrix_1(nx, ny, L, c2_dx2, c2_dx4, base=base1)
        A_2 = ohm.assemble_matrix_2(nx, ny, L, c2_dx2, base=base2)

        b1 = np.concatenate([S[..., 0].flatten(order="C"), S[..., 1].flatten(order="C")])
        b2 = S[..., 2].flatten(order="C")

        if solver == "mgcg" and (M1_cached is None or M2_cached is None):
            t_pre0 = time.perf_counter()
            ml1 = pyamg.smoothed_aggregation_solver(A_1)
            ml2 = pyamg.smoothed_aggregation_solver(A_2)
            M1_cached = ml1.aspreconditioner(cycle="V")
            M2_cached = ml2.aspreconditioner(cycle="V")
            precond_setup_time += time.perf_counter() - t_pre0

        t0 = time.perf_counter()
        t1 = time.perf_counter()
        _, st1, r1 = solve_block(A_1, b1, M=M1_cached)
        t2 = time.perf_counter()
        _, st2, r2 = solve_block(A_2, b2, M=M2_cached)
        t3 = time.perf_counter()

        elapsed_total.append(t3 - t0)
        elapsed_a1.append(t2 - t1)
        elapsed_a2.append(t3 - t2)
        iter_1.append(max(len(r1) - 1, 0))
        iter_2.append(max(len(r2) - 1, 0))
        status_1.append(st1)
        status_2.append(st2)

    return {
        "solver": solver,
        "time_total_mean": float(np.mean(elapsed_total)),
        "time_total_std": float(np.std(elapsed_total)),
        "time_a1_mean": float(np.mean(elapsed_a1)),
        "time_a2_mean": float(np.mean(elapsed_a2)),
        "niter1_mean": float(np.mean(iter_1)),
        "niter2_mean": float(np.mean(iter_2)),
        "status_1": status_1,
        "status_2": status_2,
        "precond_setup_time": float(precond_setup_time),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ohm solver backends")
    parser.add_argument("--nx", type=int, default=96, help="Grid size in x")
    parser.add_argument("--ny", type=int, default=96, help="Grid size in y")
    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repeats")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--c", type=float, default=1.0, help="Speed of light")
    parser.add_argument("--delta", type=float, default=1.0, help="Grid spacing")
    args = parser.parse_args()

    print(f"Benchmarking Ohm solver on grid {args.nx}x{args.ny}, repeats={args.repeats}")

    res_cg = _run_solver(
        nx=args.nx,
        ny=args.ny,
        c=args.c,
        delta=args.delta,
        solver="cg",
        repeats=args.repeats,
        seed=args.seed,
    )
    res_mgcg = _run_solver(
        nx=args.nx,
        ny=args.ny,
        c=args.c,
        delta=args.delta,
        solver="mgcg",
        repeats=args.repeats,
        seed=args.seed,
    )

    results = (res_cg, res_mgcg)

    print()
    print(
        f"{'solver':<8} {'total [s]':>10} {'std [s]':>10} {'A1 [s]':>10} {'A2 [s]':>10} "
        f"{'iter A1':>9} {'iter A2':>9} {'setup [s]':>10}"
    )
    print("-" * 86)

    for r in results:
        setup = r["precond_setup_time"]
        setup_str = f"{setup:.4f}" if setup > 0.0 else "-"
        print(
            f"{r['solver']:<8} "
            f"{r['time_total_mean']:>10.4f} "
            f"{r['time_total_std']:>10.4f} "
            f"{r['time_a1_mean']:>10.4f} "
            f"{r['time_a2_mean']:>10.4f} "
            f"{r['niter1_mean']:>9.1f} "
            f"{r['niter2_mean']:>9.1f} "
            f"{setup_str:>10}"
        )

    print()
    print("Convergence Summary:")
    for r in results:
        ok1 = sum(s == 0 for s in r["status_1"])
        ok2 = sum(s == 0 for s in r["status_2"])
        uniq1 = sorted(set(r["status_1"]))
        uniq2 = sorted(set(r["status_2"]))
        print(
            f"  {r['solver']:<8} "
            f"A1 ok={ok1}/{len(r['status_1'])} codes={uniq1}, "
            f"A2 ok={ok2}/{len(r['status_2'])} codes={uniq2}"
        )


if __name__ == "__main__":
    main()
