"""
Lyapunov-inspired RBFNN outer-loop altitude controller.

The controller computes a vertical command c for Tello SDK:
    rc 0 0 c 0

The adaptive law uses sigma modification:
    W_dot = gamma * s * xi - sigma * W
where:
    s = e2 + lambda * e1

This is not raw thrust control. It is an outer-loop command shaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class RBFNNAltitudeController:
    zd: float = 0.8
    lam: float = 1.5
    ks: float = 0.6
    gamma: float = 0.05
    sigma: float = 0.02
    ku: float = 20.0
    cmax: int = 15

    # RBF structure
    e1_centers: tuple = (-0.5, -0.25, 0.0, 0.25, 0.5)
    e2_centers: tuple = (-0.6, -0.3, 0.0, 0.3, 0.6)
    width: float = 0.35
    w_clip: float = 5.0

    W: np.ndarray = field(init=False)
    centers: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.centers = np.array(
            [(a, b) for a in self.e1_centers for b in self.e2_centers],
            dtype=float,
        )
        self.W = np.zeros(len(self.centers), dtype=float)

    def reset(self) -> None:
        self.W[:] = 0.0

    def features(self, e1: float, e2: float) -> np.ndarray:
        x = np.array([e1, e2], dtype=float)
        diff = self.centers - x
        phi = np.exp(-np.sum(diff * diff, axis=1) / (2.0 * self.width * self.width))
        denom = np.sum(phi) + 1e-9
        return phi / denom

    def step(self, z: float, vz: float, dt: float, use_nn: bool = True) -> dict:
        """
        Return a dict with c command and diagnostic variables.
        """
        dt = max(1e-3, float(dt))
        e1 = z - self.zd
        e2 = vz
        s = e2 + self.lam * e1

        xi = self.features(e1, e2)
        dhat = float(np.dot(self.W, xi)) if use_nn else 0.0

        # Outer-loop control signal. Positive means "go up".
        uf = -self.ks * s - dhat

        if use_nn:
            W_dot = self.gamma * s * xi - self.sigma * self.W
            self.W += W_dot * dt
            self.W = np.clip(self.W, -self.w_clip, self.w_clip)

        c_float = self.ku * uf
        c = int(round(clip(c_float, -self.cmax, self.cmax)))

        return {
            "z": z,
            "zd": self.zd,
            "vz": vz,
            "e1": e1,
            "e2": e2,
            "s": s,
            "dhat": dhat,
            "uf": uf,
            "c": c,
            "c_float": c_float,
            "W_norm": float(np.linalg.norm(self.W)),
        }
