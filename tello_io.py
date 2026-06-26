"""
Minimal DJI/Ryze Tello UDP interface.

This file intentionally avoids bypassing any internal safety controller.
It only uses public SDK-style text commands.

Architecture mirrors djitellopy: a background thread receives *all* UDP
responses on the command port, and send_command() polls the response list
rather than doing a synchronous recvfrom().  This avoids race conditions
where stale data or state packets corrupt the response.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


TELLO_IP = "192.168.10.1"
TELLO_CMD_PORT = 8889
LOCAL_STATE_PORT = 8890


def parse_state_packet(packet: str) -> Dict[str, float]:
    """
    Parse Tello state string:
    pitch:0;roll:0;yaw:0;vgx:0;...;h:80;bat:90;baro:12.34;...
    """
    out: Dict[str, float] = {}
    for item in packet.strip().split(";"):
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


@dataclass
class TelloIO:
    tello_ip: str = TELLO_IP
    tello_port: int = TELLO_CMD_PORT
    local_state_port: int = LOCAL_STATE_PORT
    command_timeout: float = 7.0

    _cmd_socket: Optional[socket.socket] = field(default=None, init=False)
    _state_socket: Optional[socket.socket] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _state_thread: Optional[threading.Thread] = field(default=None, init=False)
    _response_thread: Optional[threading.Thread] = field(default=None, init=False)
    _latest_state: Dict[str, float] = field(default_factory=dict, init=False)
    _latest_state_time: float = field(default=0.0, init=False)
    _responses: List[str] = field(default_factory=list, init=False)
    _response_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def open(self) -> None:
        # Command socket — bind to port 8889 (same as djitellopy).
        # Tello sends command responses back to this port.
        self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cmd_socket.bind(("", self.tello_port))

        # State socket — Tello broadcasts state to port 8890.
        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._state_socket.settimeout(1.0)
        self._state_socket.bind(("", self.local_state_port))

        self._running = True

        self._response_thread = threading.Thread(
            target=self._response_loop, daemon=True
        )
        self._response_thread.start()

        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def close(self) -> None:
        self._running = False
        try:
            self.send_rc(0, 0, 0, 0)
        except Exception:
            pass

        for sock in (self._cmd_socket, self._state_socket):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    #  Background receivers                                               #
    # ------------------------------------------------------------------ #

    def _response_loop(self) -> None:
        """Receive command responses on port 8889 in background."""
        assert self._cmd_socket is not None
        self._cmd_socket.settimeout(1.0)
        while self._running:
            try:
                data, addr = self._cmd_socket.recvfrom(4096)
                if addr[0] != self.tello_ip:
                    continue
                text = data.decode("utf-8", errors="ignore").strip()
                with self._response_lock:
                    self._responses.append(text)
            except socket.timeout:
                continue
            except OSError:
                break

    def _state_loop(self) -> None:
        """Receive state telemetry on port 8890 in background."""
        assert self._state_socket is not None
        while self._running:
            try:
                data, _ = self._state_socket.recvfrom(4096)
                msg = data.decode("utf-8", errors="ignore")
                st = parse_state_packet(msg)
                if st:
                    self._latest_state = st
                    self._latest_state_time = time.time()
            except socket.timeout:
                continue
            except OSError:
                break

    # ------------------------------------------------------------------ #
    #  Command interface                                                  #
    # ------------------------------------------------------------------ #

    def send_command(
        self,
        command: str,
        timeout: Optional[float] = None,
        retries: int = 1,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Send a command and wait for response (polled from background thread).

        Parameters
        ----------
        timeout : float | None
            Per-attempt timeout in seconds (default: self.command_timeout).
        retries : int
            Total number of attempts (default 1 = no retry).
        retry_delay : float
            Seconds to wait between retries.
        """
        if self._cmd_socket is None:
            raise RuntimeError("Socket not opened. Call open() first.")

        per_attempt = timeout if timeout is not None else self.command_timeout

        for attempt in range(1, retries + 1):
            # Drain any stale responses before sending.
            with self._response_lock:
                self._responses.clear()

            try:
                self._cmd_socket.sendto(
                    command.encode("utf-8"), (self.tello_ip, self.tello_port)
                )
            except OSError as exc:
                # e.g. WinError 10051 "network unreachable" — treat as timeout.
                if attempt < retries:
                    print(
                        f"[TELLO] '{command}' send failed ({exc}) "
                        f"(attempt {attempt}/{retries}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue
                raise TimeoutError(
                    f"Tello unreachable for '{command}' after {retries} attempt(s): {exc}"
                ) from exc

            # Poll for a response from the background thread.
            t0 = time.time()
            while time.time() - t0 < per_attempt:
                with self._response_lock:
                    if self._responses:
                        return self._responses.pop(0)
                time.sleep(0.1)

            # Timeout — retry if allowed.
            if attempt < retries:
                print(
                    f"[TELLO] '{command}' timeout (attempt {attempt}/{retries}), "
                    f"retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)

        raise TimeoutError(
            f"Tello did not respond to '{command}' after {retries} attempt(s)."
        )

    def send_command_no_wait(self, command: str) -> None:
        if self._cmd_socket is None:
            raise RuntimeError("Socket not opened. Call open() first.")
        try:
            self._cmd_socket.sendto(
                command.encode("utf-8"), (self.tello_ip, self.tello_port)
            )
        except OSError:
            pass  # Best-effort for fire-and-forget commands like rc.

    def send_rc(self, a: int, b: int, c: int, d: int) -> None:
        """
        rc a b c d:
        a: left/right, b: forward/back, c: up/down, d: yaw
        Each channel is clipped to [-100, 100].
        """
        vals = [int(max(-100, min(100, v))) for v in (a, b, c, d)]
        self.send_command_no_wait(f"rc {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

    # ------------------------------------------------------------------ #
    #  State helpers                                                      #
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, float]:
        return dict(self._latest_state)

    def state_age(self) -> float:
        if self._latest_state_time <= 0:
            return float("inf")
        return time.time() - self._latest_state_time

    def wait_for_state(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._latest_state:
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def height_m_from_state(state: Dict[str, float]) -> Optional[float]:
        """
        Prefer ToF if available and plausible. Fall back to h.
        Tello state usually reports h in cm and tof in cm.
        """
        tof = state.get("tof", None)
        h = state.get("h", None)

        # Use ToF only if plausible for indoor altitude experiment.
        if tof is not None and 20 <= tof <= 300:
            return tof / 100.0
        if h is not None and 20 <= h <= 300:
            return h / 100.0
        return None

    @staticmethod
    def vertical_velocity_mps_from_state(state: Dict[str, float]) -> Optional[float]:
        vgz = state.get("vgz", None)
        if vgz is None:
            return None
        return vgz / 100.0
