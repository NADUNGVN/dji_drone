"""
Run altitude outer-loop experiments on DJI/Ryze Tello.

Modes:
    baseline: Tello takeoff + hover, log only, send rc zero
    pd:       outer-loop PD/sliding-error command
    nn:       PD + RBFNN adaptive disturbance compensation

Dry-run is default. Add --fly to send takeoff/land and rc commands to real Tello.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

from tello_io import TelloIO
from rbfnn_controller import RBFNNAltitudeController


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def lowpass(prev, value, alpha: float):
    if prev is None:
        return value
    return alpha * value + (1.0 - alpha) * prev


def make_log_path(mode: str) -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"{mode}_{now_stamp()}.csv"


def run_experiment(args) -> Path:
    mode = args.mode.lower()
    if mode not in {"baseline", "pd", "nn"}:
        raise ValueError("mode must be baseline, pd, or nn")

    log_path = make_log_path(mode)
    controller = RBFNNAltitudeController(
        zd=args.zd,
        lam=args.lam,
        ks=args.ks,
        gamma=args.gamma,
        sigma=args.sigma,
        ku=args.ku,
        cmax=args.cmax,
    )

    fields = [
        "time",
        "mode",
        "z",
        "zd",
        "vz",
        "e1",
        "e2",
        "s",
        "c",
        "c_float",
        "uf",
        "dhat",
        "W_norm",
        "bat",
        "h_raw",
        "tof_raw",
        "baro_raw",
        "state_age",
    ]

    print(f"[INFO] Log file: {log_path}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Fly mode: {args.fly}")

    tello = None

    try:
        if args.fly:
            tello = TelloIO()
            tello.open()
            print("[INFO] Enter SDK command mode...")
            resp = tello.send_command("command", timeout=5, retries=5, retry_delay=2.0)
            print(f"[TELLO] command -> {resp}")

            if not tello.wait_for_state(timeout=5):
                raise RuntimeError("No state packets received from Tello.")

            state = tello.get_state()
            bat = state.get("bat", -1)
            print(f"[INFO] Battery: {bat}%")
            if bat < args.min_bat:
                raise RuntimeError(f"Battery too low: {bat}% < {args.min_bat}%")

            print("[INFO] Takeoff...")
            resp = tello.send_command("takeoff", timeout=20, retries=3, retry_delay=2.0)
            print(f"[TELLO] takeoff -> {resp}")
            time.sleep(args.takeoff_wait)

        z_f = None
        vz_f = None
        last_time = time.time()
        t0 = last_time

        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while True:
                t = time.time() - t0
                if t >= args.duration:
                    break

                current = time.time()
                dt = current - last_time
                last_time = current

                if args.fly and tello is not None:
                    state = tello.get_state()
                    age = tello.state_age()
                    z_meas = tello.height_m_from_state(state)
                    vz_meas = tello.vertical_velocity_mps_from_state(state)
                    if z_meas is None:
                        print("[WARN] No valid height. Sending rc zero.")
                        tello.send_rc(0, 0, 0, 0)
                        time.sleep(args.period)
                        continue
                    if vz_meas is None:
                        vz_meas = 0.0
                else:
                    # Dry-run placeholder: pretend at reference altitude.
                    state = {}
                    age = 0.0
                    z_meas = args.zd
                    vz_meas = 0.0

                z_f = lowpass(z_f, z_meas, args.alpha_z)
                vz_f = lowpass(vz_f, vz_meas, args.alpha_vz)

                if mode == "baseline":
                    diag = controller.step(z_f, vz_f, dt, use_nn=False)
                    diag["c"] = 0
                    diag["c_float"] = 0.0
                    diag["uf"] = 0.0
                    diag["dhat"] = 0.0
                    diag["W_norm"] = 0.0
                elif mode == "pd":
                    diag = controller.step(z_f, vz_f, dt, use_nn=False)
                else:
                    diag = controller.step(z_f, vz_f, dt, use_nn=True)

                # Safety checks.
                unsafe = False
                reason = ""
                bat = state.get("bat", -1)
                if args.fly:
                    if bat >= 0 and bat < args.min_bat:
                        unsafe, reason = True, f"low battery {bat}%"
                    elif z_f < args.z_min:
                        unsafe, reason = True, f"z too low {z_f:.2f} m"
                    elif z_f > args.z_max:
                        unsafe, reason = True, f"z too high {z_f:.2f} m"
                    elif abs(diag["e1"]) > args.max_error:
                        unsafe, reason = True, f"altitude error too large {diag['e1']:.2f} m"

                row = {
                    "time": t,
                    "mode": mode,
                    "z": diag["z"],
                    "zd": diag["zd"],
                    "vz": diag["vz"],
                    "e1": diag["e1"],
                    "e2": diag["e2"],
                    "s": diag["s"],
                    "c": diag["c"],
                    "c_float": diag["c_float"],
                    "uf": diag["uf"],
                    "dhat": diag["dhat"],
                    "W_norm": diag["W_norm"],
                    "bat": bat,
                    "h_raw": state.get("h", ""),
                    "tof_raw": state.get("tof", ""),
                    "baro_raw": state.get("baro", ""),
                    "state_age": age,
                }
                writer.writerow(row)
                f.flush()

                if args.fly and tello is not None:
                    if unsafe:
                        print(f"[SAFETY] {reason}. Landing.")
                        tello.send_rc(0, 0, 0, 0)
                        time.sleep(0.2)
                        tello.send_command("land", timeout=10)
                        break

                    tello.send_rc(0, 0, int(diag["c"]), 0)

                print(
                    f"t={t:5.2f}s z={diag['z']:.2f} zd={diag['zd']:.2f} "
                    f"e={diag['e1']:+.2f} vz={diag['vz']:+.2f} "
                    f"c={diag['c']:+d} dhat={diag['dhat']:+.3f} "
                    f"W={diag['W_norm']:.3f}",
                    end="\r",
                )
                time.sleep(args.period)

        print("\n[INFO] Experiment complete.")

    finally:
        if args.fly and tello is not None:
            try:
                print("[INFO] Sending rc zero and land...")
                tello.send_rc(0, 0, 0, 0)
                time.sleep(0.5)
                tello.send_command("land", timeout=10)
            except Exception as exc:
                print(f"[WARN] Could not land cleanly: {exc}")
            tello.close()

    return log_path


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "pd", "nn"], default="baseline")
    p.add_argument("--fly", action="store_true", help="Actually fly the Tello.")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--period", type=float, default=0.1, help="Control period in seconds.")
    p.add_argument("--takeoff-wait", type=float, default=3.0)

    # Controller
    p.add_argument("--zd", type=float, default=0.8)
    p.add_argument("--lam", type=float, default=1.5)
    p.add_argument("--ks", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--ku", type=float, default=20.0)
    p.add_argument("--cmax", type=int, default=15)

    # Filtering
    p.add_argument("--alpha-z", type=float, default=0.35)
    p.add_argument("--alpha-vz", type=float, default=0.35)

    # Safety
    p.add_argument("--min-bat", type=float, default=40.0)
    p.add_argument("--z-min", type=float, default=0.35)
    p.add_argument("--z-max", type=float, default=1.30)
    p.add_argument("--max-error", type=float, default=0.50)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_experiment(args)
