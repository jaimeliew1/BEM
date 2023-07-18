from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
import polars as pl

from mit_bem.Turbine import IEA15MW
from mit_bem.BEM import BEM

PARALLEL = True
ROTOR = IEA15MW()
DERATE = 0.9
BOUNDS = [0, 1]

tsr_opt, pitch_opt = 8.678696, np.deg2rad(-3.484844)
Cp_max = 0.5237
print(Cp_max * DERATE)

np.seterr(all="ignore")
figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


def for_each(func, params, parallel=True):
    N = len(params)
    out = []
    if parallel:
        with Pool() as pool:
            for x in tqdm(
                pool.imap(
                    func,
                    params,
                ),
                total=N,
            ):
                out.append(x)
        return out
    else:
        for param in tqdm(params):
            out.append(func(param))
        return out


def generate_unit_hemisphere(nz=40, nazi=40):
    z = np.linspace(0, 1, nz)
    azi = np.linspace(0, 2 * np.pi, nazi)
    z_mesh, azi_mesh = np.meshgrid(z, azi)

    r = np.sqrt(1 - z_mesh**2)

    x = r * np.cos(azi_mesh)
    y = r * np.sin(azi_mesh)

    return x * np.deg2rad(10), y * 5, z_mesh * np.deg2rad(50)


def get_derate_setpoint(Cp_target, x0, dx):
    x0, dx = np.array(x0), np.array(dx)
    bem = BEM(ROTOR)

    def func(coef):
        pitch, tsr, yaw = x0 + coef * dx
        bem.solve(pitch, tsr, yaw)
        if bem.converged:
            return bem.Cp() - Cp_target
        else:
            return np.nan

    # check bounds
    fa = func(BOUNDS[0])
    fb = func(BOUNDS[1])
    if fa * fb > 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    sol = optimize.root_scalar(func, x0=1, bracket=BOUNDS, xtol=0.0001)
    if not sol.converged:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    pitch, tsr, yaw = x0 + sol.root * dx
    bem.solve(pitch, tsr, yaw)

    return np.rad2deg(pitch), tsr, np.rad2deg(yaw), bem.Cp(), bem.Ct(), bem.Ctprime()


def get_derate_setpoint_wrapper(dx):
    return get_derate_setpoint(DERATE * Cp_max, (pitch_opt, tsr_opt, 0), dx)


if __name__ == "__main__":
    # pitch, tsr, yaw
    X, Y, Z = generate_unit_hemisphere()

    params = list(zip(X.ravel(), Y.ravel(), Z.ravel()))
    out = for_each(get_derate_setpoint_wrapper, params, parallel=PARALLEL)

    df = pl.DataFrame(out, schema=["pitch", "tsr", "yaw", "Cp", "Ct", "Ctprime"])

    X = np.reshape(df["pitch"], X.shape)
    Y = np.reshape(df["tsr"], Y.shape)
    Z = np.reshape(df["yaw"], Z.shape)
    Ct = np.reshape(df["Ctprime"], Z.shape)
    color_val = (Ct - np.nanmin(Ct)) / (np.nanmax(Ct - np.nanmin(Ct)))
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(color_val))

    ax.set_xlabel("pitch [deg]")
    ax.set_ylabel("$\lambda$ [-]")
    ax.set_zlabel("yaw [deg]")

    ax.view_init(elev=50, azim=60)

    plt.savefig(figdir / "derate_frontier.png", dpi=300, bbox_inches="tight")
