"""
Simple simulation before real Tello experiment.

This is not a full quadrotor model. It is only a safe tuning sandbox for the
outer-loop controller.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import math
import random

from rbfnn_controller import RBFNNAltitudeController


def run_sim(args) -> Path:
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"sim_{args.mode}.csv"

    ctrl = RBFNNAltitudeController(
        zd=args.zd,
        lam=args.lam,
        ks=args.ks,
        gamma=args.gamma,
        sigma=args.sigma,
        ku=args.ku,
        cmax=args.cmax,
    )

    # Tello-like vertical dynamics approximation:
    # z_dot = vz
    # vz_dot = -a*vz + b*c/100 + disturbance
    z = args.z0
    vz = 0.0
    a = 1.4
    b = 1.2

    fields = ["time", "mode", "z", "zd", "vz", "e1", "e2", "s", "c", "c_float", "uf", "dhat", "W_norm", "d"]

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        t = 0.0
        while t < args.duration:
            # Reference step after 10 s
            if args.step and t > args.duration / 2:
                ctrl.zd = args.zd2

            use_nn = args.mode == "nn"
            diag = ctrl.step(z, vz, args.dt, use_nn=use_nn)

            if args.mode == "baseline":
                c = 0
                diag["c"] = 0
                diag["dhat"] = 0.0
                diag["W_norm"] = 0.0
            else:
                c = diag["c"]

            d = 0.20 * math.sin(1.8 * t) + 0.08 * math.cos(4.2 * t)
            d += random.gauss(0.0, 0.015)

            vz_dot = -a * vz + b * (c / 100.0) + d
            vz += vz_dot * args.dt
            z += vz * args.dt

            # Floor/ceiling approximation
            if z < 0.2:
                z = 0.2
                vz = max(0.0, vz)
            if z > 1.5:
                z = 1.5
                vz = min(0.0, vz)

            row = {k: diag.get(k, "") for k in fields}
            row["time"] = t
            row["mode"] = args.mode
            row["d"] = d
            writer.writerow(row)

            t += args.dt

    print(f"[INFO] Saved {log_path}")
    return log_path


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "pd", "nn"], default="pd")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--z0", type=float, default=0.55)
    p.add_argument("--zd", type=float, default=0.8)
    p.add_argument("--zd2", type=float, default=0.95)
    p.add_argument("--step", action="store_true")

    p.add_argument("--lam", type=float, default=1.5)
    p.add_argument("--ks", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--ku", type=float, default=20.0)
    p.add_argument("--cmax", type=int, default=15)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_sim(args)
