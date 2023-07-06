from typing import Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


airfoil_fn = Path("")


def lift_drag(aoa):
    Cl = 2 * np.pi * aoa
    Cd = np.zeros_like(aoa)

    return Cl, Cd


def fixedpointiteration(
    f: Callable[[np.ndarray, any], np.ndarray],
    x0: np.ndarray,
    args=(),
    eps=0.000001,
    maxiter=100,
) -> np.ndarray:
    """
    Performs fixed-point iteration on function f until residuals converge or max
    iterations is reached.

    Args:
        f (Callable): residual function of form f(x, *args) -> np.ndarray
        x0 (np.ndarray): Initial guess
        args (tuple): arguments to pass to residual function. Defaults to ().
        eps (float): Convergence tolerance. Defaults to 0.000001.
        maxiter (int): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: Max iterations reached.

    Returns:
        np.ndarray: Solution to residual function.
    """
    for c in range(maxiter):
        residuals = f(x0, *args)

        x0 += residuals
        if np.abs(residuals).max() < eps:
            break
    else:
        raise ValueError("max iterations reached.")

    print(f"niter: {c}")
    return x0


class Rotor:
    def __init__(self, chord, twist, lift_drag_func, N_theta, N_r, B=3):
        self.N_theta, self.N_r = N_theta, N_r
        self.chord = chord
        self.twist = twist
        self.lift_drag_func = lift_drag_func
        self.B = B

        thetas = np.linspace(0, 2 * np.pi, N_theta)
        Rs = np.linspace(0, 1, N_r)

        self.r_mesh, self.theta_mesh = np.meshgrid(Rs, thetas)

    def induction(self, pitch, omega, yaw):
        a = fixedpointiteration(self.residual, x0=1 / 3, args=(pitch, omega, yaw))
        return a

    def residual(self, _a, pitch, omega, yaw):
        a = _a * np.ones((self.N_theta, self.N_r))

        # local wind direction
        gamma = yaw * np.ones((self.N_theta, self.N_r))  # + veer
        U = np.ones((self.N_theta, self.N_r))
        vx = (
            (1 - a)
            * np.cos(gamma * np.cos(self.theta_mesh))
            * U
            * np.cos(gamma * np.sin(self.theta_mesh))
        )

        vt = (
            -np.sin(gamma * np.cos(self.theta_mesh))
            * U
            * np.cos(gamma * np.sin(self.theta_mesh))
        )

        # inflow angle
        phi = np.arctan2(vx, (omega * self.r_mesh + (1 - a) * vt))

        Cl, Cd = lift_drag(phi - self.twist - pitch)

        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ct = Cl * np.sin(phi) - Cd * np.cos(phi)

        Ct = (
            self.B
            / np.pi**2
            * np.trapz(np.trapz(self.r_mesh * self.chord * vx**2 * Cn, axis=-1))
        )

        a_new = (2 * Ct - 4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)) / (
            -4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)
        )

        return a_new - _a


if __name__ == "__main__":
    rotor = Rotor(
        chord=0.05,
        twist=0,
        lift_drag_func=lift_drag,
        N_theta=5,
        N_r=6,
    )

    a = rotor.induction(pitch=0, omega=2, yaw=np.deg2rad(0))
    print(a)
