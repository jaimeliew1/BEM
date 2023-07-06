from typing import Callable
from pathlib import Path
import yaml
from rich import print
import numpy as np
from scipy import interpolate

fn_IEA15MW = Path(__file__).parent / "IEA-15-240-RWT.yaml"

RHO = 1.293


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


class Rotor:
    @classmethod
    def from_windio(cls, windio: dict):
        blade = windio["components"]["blade"]

        data_twist = blade["outer_shape_bem"]["twist"]
        data_chord = blade["outer_shape_bem"]["chord"]
        twist_func = interpolate.interp1d(
            data_twist["grid"], data_twist["values"], fill_value="extrapolate"
        )
        chord_func = interpolate.interp1d(
            data_chord["grid"], data_chord["values"], fill_value="extrapolate"
        )

        airfoil_func = BladeAirfoils.from_windio(windio)

        D = windio["assembly"]["rotor_diameter"]
        N_blades = windio["assembly"]["number_of_blades"]

        return cls(twist_func, chord_func, airfoil_func, N_blades, D)

    def __init__(
        self, twist_func, chord_func, airfoil_func, N_blades, D, N_r=30, N_theta=31
    ):
        self.twist = twist_func
        self.chord = chord_func
        self.airfoil = airfoil_func
        self.N_blades = N_blades
        self.D = D
        self.R = D / 2
        self.N_r = N_r
        self.N_theta = N_theta

        thetas = np.linspace(0, 2 * np.pi, N_theta)
        Rs = np.linspace(0.0001, 1, self.N_r)

        self.r_mesh, self.theta_mesh = np.meshgrid(Rs, thetas)

    def bem_residual(self, a, pitch, tsr, _yaw, residual=True):
        r_mesh = np.linspace(0.01, 0.999, self.N_r)
        vx = 1 - a
        vt = tsr * r_mesh
        W = np.sqrt(vx**2 + vt**2)

        # inflow angle
        phi = np.arctan2(vx, vt)
        aoa = phi - self.twist(r_mesh) - pitch

        Cl, Cd = self.airfoil(r_mesh, aoa)
        print(np.rad2deg(aoa.max()), np.rad2deg(aoa.min()))
        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ctan = Cl * np.sin(phi) - Cd * np.cos(phi)

        solidity = self.N_blades * self.chord(r_mesh) / (2 * np.pi * r_mesh * self.R)

        dCt = (1 - a) ** 2 * solidity * Cn / np.sin(phi) ** 2

        a_new = 0.0883 * dCt**3 + 0.0586 * dCt**2 + 0.246 * dCt

        F = (
            2
            / np.pi
            * np.arccos(
                np.exp(-self.N_blades / 2 * (1 - r_mesh) / (r_mesh * np.sin(phi)))
            )
        )
        a_new /= F

        if residual:
            return a_new - a
        else:
            return r_mesh, a_new, phi, W, Cn, Ctan, solidity

    # def bem_residual_mike(self, _a, pitch, tsr, yaw, residual=True):
    #     a = _a * np.ones((self.N_theta, self.N_r))

    #     # local wind direction
    #     gamma = yaw * np.ones((self.N_theta, self.N_r))  # + veer
    #     vx = (
    #         (1 - a)
    #         * np.cos(gamma * np.cos(self.theta_mesh))
    #         * np.cos(gamma * np.sin(self.theta_mesh))
    #     )

    #     vt = tsr * self.r_mesh - (1 - a) * np.sin(
    #         gamma * np.cos(self.theta_mesh)
    #     ) * np.cos(gamma * np.sin(self.theta_mesh))

    #     # inflow angle
    #     phi = np.arctan2(vx, vt)
    #     aoa = phi - self.twist(self.r_mesh) - pitch

    #     W = np.sqrt(vx**2 + vt**2)
    #     Cl, Cd = self.airfoil(self.r_mesh, aoa)

    #     Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
    #     Ctan = Cl * np.sin(phi) - Cd * np.cos(phi)

    #     solidity = (
    #         self.N_blades * self.chord(self.r_mesh) / (2 * np.pi * self.r_mesh * self.R)
    #     )

    #     dCt = W**2 * solidity * Cn * self.r_mesh

    #     Ct_ = np.trapz(
    #         np.trapz(dCt, dx=1 / self.N_r, axis=-1),
    #         dx=2 * np.pi / self.N_theta,
    #         axis=-1,
    #     )
    #     Ct = min(Ct_, 1)
    #     print(Ct_)
    #     a_new = (2 * Ct - 4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)) / (
    #         -4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)
    #     )

    #     if residual:
    #         return a_new - _a
    #     else:
    #         return a_new, phi, Cn, Ctan, solidity, dCt

    def Ct(self, pitch, tsr, yaw):
        a = self.induction(pitch, tsr, yaw)
        r, a, phi, W, Cn, Ctan, solidity = self.bem_residual(
            a, pitch, tsr, yaw, residual=False
        )

        dT = 0.5 * RHO * W**2 * Cn
        Ct = np.trapz(dT, r) / (0.5 * RHO * np.pi)

        return Ct

    def induction(self, pitch, tsr, yaw):
        # a0 = 1 / 3 * np.ones((self.N_theta, self.N_r))
        a0 = 1 / 3 * np.ones(self.N_r)
        a = fixedpointiteration(
            self.bem_residual, x0=a0, args=(pitch, tsr, yaw), relax=0.5, maxiter=1000
        )
        # a0 = 1 / 3
        # a = fixedpointiteration(
        #     self.bem_residual_mike, x0=a0, args=(pitch, omega, yaw), relax=0
        # )

        return a


def IEA15MW():
    with open(fn_IEA15MW, "r") as f:
        data = yaml.safe_load(f)

    return Rotor.from_windio(data)
