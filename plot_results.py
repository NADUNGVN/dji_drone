"""
Plot altitude experiment logs and compute simple metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import math
from typing import List, Dict

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def metric(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"rmse": float("nan"), "max_abs_e": float("nan"), "iae": float("nan"), "energy": float("nan")}

    ts = [float(r["time"]) for r in rows]
    e = [float(r.get("e1", 0.0)) for r in rows]
    c = [float(r.get("c", 0.0)) for r in rows]

    rmse = math.sqrt(sum(x*x for x in e) / max(1, len(e)))
    max_abs_e = max(abs(x) for x in e)
    iae = 0.0
    energy = 0.0
    for i in range(1, len(rows)):
        dt = max(0.0, ts[i] - ts[i-1])
        iae += abs(e[i]) * dt
        energy += c[i] * c[i] * dt

    return {"rmse": rmse, "max_abs_e": max_abs_e, "iae": iae, "energy": energy}


def plot_logs(paths: List[Path]) -> None:
    Path("figures").mkdir(exist_ok=True)

    datasets = []
    for p in paths:
        rows = read_csv(p)
        if rows:
            datasets.append((p, rows))

    if not datasets:
        print("[WARN] No valid datasets.")
        return

    # Figure 1: altitude
    plt.figure()
    for path, rows in datasets:
        t = [r["time"] for r in rows]
        z = [r["z"] for r in rows]
        zd = [r["zd"] for r in rows]
        label = path.stem
        plt.plot(t, z, label=f"{label}: z")
        plt.plot(t, zd, linestyle="--", label=f"{label}: zd")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Altitude tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/altitude_tracking.png", dpi=200)

    # Figure 2: error
    plt.figure()
    for path, rows in datasets:
        t = [r["time"] for r in rows]
        e = [r["e1"] for r in rows]
        plt.plot(t, e, label=path.stem)
    plt.xlabel("Time [s]")
    plt.ylabel("e1 = z - zd [m]")
    plt.title("Altitude error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/altitude_error.png", dpi=200)

    # Figure 3: command
    plt.figure()
    for path, rows in datasets:
        t = [r["time"] for r in rows]
        c = [r["c"] for r in rows]
        plt.plot(t, c, label=path.stem)
    plt.xlabel("Time [s]")
    plt.ylabel("Tello vertical rc command c")
    plt.title("Control command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/control_command.png", dpi=200)

    # Figure 4: NN diagnostics
    plt.figure()
    for path, rows in datasets:
        t = [r["time"] for r in rows]
        dhat = [r.get("dhat", 0.0) for r in rows]
        plt.plot(t, dhat, label=path.stem)
    plt.xlabel("Time [s]")
    plt.ylabel("dhat")
    plt.title("RBFNN disturbance estimate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/dhat.png", dpi=200)

    # Figure 5: W norm
    plt.figure()
    for path, rows in datasets:
        t = [r["time"] for r in rows]
        wn = [r.get("W_norm", 0.0) for r in rows]
        plt.plot(t, wn, label=path.stem)
    plt.xlabel("Time [s]")
    plt.ylabel("||W_hat||")
    plt.title("NN weight norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/W_norm.png", dpi=200)

    metrics_path = Path("figures") / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "rmse", "max_abs_e", "iae", "energy"])
        writer.writeheader()
        for path, rows in datasets:
            m = metric(rows)
            writer.writerow({"file": path.name, **m})
            print(
                f"{path.name}: RMSE={m['rmse']:.4f}, "
                f"Max|e|={m['max_abs_e']:.4f}, IAE={m['iae']:.4f}, "
                f"Energy={m['energy']:.2f}"
            )

    print("[INFO] Figures saved to figures/")
    print(f"[INFO] Metrics saved to {metrics_path}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="CSV log files.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    plot_logs([Path(x) for x in args.logs])
