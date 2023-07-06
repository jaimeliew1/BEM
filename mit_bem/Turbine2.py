from typing import Callable
from pathlib import Path
import yaml
from rich import print
import numpy as np
from scipy import interpolate

fn_IEA15MW = Path(__file__).parent / "IEA-15-240-RWT.yaml"

RHO = 1.293


def PrandtlTiploss(mu, phi, B=3):
    f = B / 2 * (1 - mu) / (mu * np.sin(phi))
    F = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f, -100, 100)), -1, 1))
    return F


def NoTiploss(mu, phi):
    return np.ones_like(mu)


def Ct2a_HAWC2(Ct):
    return 0.0883 * Ct**3 + 0.0586 * Ct**2 + 0.246 * Ct


def fixedpointiteration(
    f: Callable[[np.ndarray, any], np.ndarray],
    x0: np.ndarray,
    args=(),
    eps=0.000001,
    maxiter=100,
    relax=0,
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

        x0 += (1 - relax) * residuals
        if np.abs(residuals).max() < eps:
            break
    else:
        raise ValueError("max iterations reached.")

    # print(f"niter: {c}")
    return x0


class Airfoil:
    @classmethod
    def from_windio_airfoil(cls, airfoil: dict):
        assert len(airfoil["polars"]) == 1
        return cls(
            airfoil["name"],
            airfoil["polars"][0]["c_l"]["grid"],
            airfoil["polars"][0]["c_l"]["values"],
            airfoil["polars"][0]["c_d"]["values"],
        )

    def __init__(self, name, grid, cl, cd):
        self.name = name
        self.Cl_interp = interpolate.interp1d(grid, cl, fill_value="extrapolate")
        self.Cd_interp = interpolate.interp1d(grid, cd, fill_value="extrapolate")

    def __repr__(self):
        return f"Airfoil: {self.name}"

    def Cl(self, angle):
        return self.Cl_interp(angle)

    def Cd(self, angle):
        return self.Cd_interp(angle)


class BladeAirfoils:
    @classmethod
    def from_windio(cls, windio: dict, N=120):
        blade = windio["components"]["blade"]
        airfoils = windio["airfoils"]
        D = windio["assembly"]["rotor_diameter"]

        airfoil_grid = np.array(blade["outer_shape_bem"]["airfoil_position"]["grid"])
        airfoil_order = blade["outer_shape_bem"]["airfoil_position"]["labels"]

        airfoils = {
            x["name"]: Airfoil.from_windio_airfoil(x) for x in windio["airfoils"]
        }

        return cls(D, airfoil_grid, airfoil_order, airfoils, N=N)

    def __init__(self, D, airfoil_grid, airfoil_order, airfoils, N=120):
        self.D = D

        aoa_grid = np.linspace(-np.pi, np.pi, N)
        cl = np.array([airfoils[name].Cl(aoa_grid) for name in airfoil_order])
        cd = np.array([airfoils[name].Cd(aoa_grid) for name in airfoil_order])

        self.cl_interp = interpolate.RegularGridInterpolator(
            (airfoil_grid, aoa_grid), cl, fill_value=None
        )
        self.cd_interp = interpolate.RegularGridInterpolator(
            (airfoil_grid, aoa_grid), cd, fill_value=None
        )

    def Cl(self, x, inflow):
        return self.cl_interp((x, inflow))

    def Cd(self, x, inflow):
        return self.cd_interp((x, inflow))

    def __call__(self, x, inflow):
        return self.Cl(x, inflow), self.Cd(x, inflow)


class GenericRotor:
    def __init__(self, twist, solidity, clcd, tiploss, Ct2a, N_r=100, N_theta=31):
        self.twist = twist
        self.solidity = solidity
        self.clcd = clcd
        self.tiploss = tiploss
        self.Ct2a = Ct2a

        self.N_r = N_r
        self.N_theta = N_theta

        self.mus = np.linspace(0.01, 0.99, N_r)

    def bem(self, a, pitch, tsr, yaw, return_data=False):
        vx = 1 - a
        vt = tsr * self.mus

        # inflow angle
        phi = np.arctan2(vx, vt)
        aoa = phi - self.twist(self.mus) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = self.clcd(self.mus, aoa)

        Cn, Ctan = np.zeros_like(Cl), np.zeros_like(Cl)
        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ctan = Cl * np.sin(phi) - Cd * np.cos(phi)

        sigma = self.solidity(self.mus)
        dCt = (1 - a) ** 2 * sigma * Cn / np.sin(phi) ** 2

        a_ring = self.Ct2a(dCt)
        a_new = a_ring / self.tiploss(self.mus, phi)

        if return_data:
            W = np.sqrt(vx**2 + vt**2)
            return a_new - a, (
                self.mus,
                a_new,
                a_ring,
                phi,
                W,
                Cn,
                Ctan,
                dCt,
                sigma,
            )
        else:
            return a_new - a

    def induction(self, pitch, tsr, yaw):
        a0 = 1 / 3 * np.ones(self.N_r)
        a = fixedpointiteration(
            self.bem, x0=a0, args=(pitch, tsr, yaw), relax=0.4, maxiter=1000
        )

        return a

    def Ct(self, pitch, tsr, yaw):
        a = self.induction(pitch, tsr, yaw)
        _, (mus, a, a_ring, phi, W, Cn, Ctan, dCt, sigma) = self.bem(
            a, pitch, tsr, yaw, return_data=True
        )

        Ct = 2 * np.trapz(W**2 * sigma * Cn * mus, mus)
        return Ct

    def Cp(self, pitch, tsr, yaw):
        a = self.induction(pitch, tsr, yaw)
        _, (mus, a, a_ring, phi, W, Cn, Ctan, dCt, sigma) = self.bem(
            a, pitch, tsr, yaw, return_data=True
        )

        Cp = 2 * np.trapz(tsr * mus**2 * W**2 * sigma * Ctan, mus)
        return Cp

    def rotor_induction(self, pitch, tsr, yaw):
        a = self.induction(pitch, tsr, yaw)

        a_rot = 2 * np.trapz(self.mus * a, self.mus)

        return a_rot


class Rotor:
    @classmethod
    def from_windio(cls, windio: dict):
        blade = windio["components"]["blade"]

        N_blades = windio["assembly"]["number_of_blades"]
        D = windio["assembly"]["rotor_diameter"]

        data_twist = blade["outer_shape_bem"]["twist"]
        data_chord = blade["outer_shape_bem"]["chord"]
        twist_func = interpolate.interp1d(
            data_twist["grid"], data_twist["values"], fill_value="extrapolate"
        )
        chord_func = interpolate.interp1d(
            data_chord["grid"], data_chord["values"], fill_value="extrapolate"
        )

        solidity_func = (
            lambda mu: N_blades * chord_func(mu) / (2 * np.pi * mu * (D / 2))
        )

        airfoil_func = BladeAirfoils.from_windio(windio)

        return cls(twist_func, solidity_func, airfoil_func, N_blades, D)

    def __init__(self, twist_func, solidity_func, airfoil_func, N_blades, D):
        self.N_blades = N_blades
        self.D = D
        self.R = D / 2

        self.bem = GenericRotor(
            twist_func, solidity_func, airfoil_func, PrandtlTiploss, Ct2a_HAWC2
        )

    def induction(self, pitch, tsr, yaw):
        return self.bem.induction(pitch, tsr, yaw)

    def bem_residual(self, a, pitch, tsr, yaw, return_data=False):
        if return_data:
            _, data = self.bem.bem(a, pitch, tsr, yaw, return_data=True)
            return data
        else:
            return self.bem.bem(a, pitch, tsr, yaw)

    def Ct(self, pitch, tsr, yaw):
        return self.bem.Ct(pitch, tsr, yaw)

    def Cp(self, pitch, tsr, yaw):
        return self.bem.Cp(pitch, tsr, yaw)

    def rotor_induction(self, pitch, tsr, yaw):
        return self.bem.rotor_induction(pitch, tsr, yaw)


def IEA15MW():
    with open(fn_IEA15MW, "r") as f:
        data = yaml.safe_load(f)

    return Rotor.from_windio(data)
