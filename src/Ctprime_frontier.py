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
CTPRIME = 2.0
BOUNDS = [0, 1]
Nyaw, Nangle = 10, 150

tsr_opt, pitch_opt = 8.678696, np.deg2rad(-3.484844)
Cp_max = 0.5237

pl.Config.set_tbl_rows(100)
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


def get_ctprime_setpoint(ctprime_target, x0, dx):
    x0, dx = np.array(x0), np.array(dx)
    bem = BEM(ROTOR, Cta_method="mike")

    def func(coef):
        pitch, tsr, yaw = x0 + coef * dx
        bem.solve(pitch, tsr, yaw)
        if bem.converged:
            return bem.Ctprime() - ctprime_target
        else:
            return np.nan

    # check bounds
    fa = func(BOUNDS[0])
    fb = func(BOUNDS[1])
    if fa * fb > 0:
        return 6 * [np.nan]
    sol = optimize.root_scalar(func, x0=0, x1=0.01, bracket=BOUNDS, xtol=0.0001)
    if not sol.converged:
        return 6 * [np.nan]
    pitch, tsr, yaw = x0 + sol.root * dx
    # bem.solve(pitch, tsr, yaw)
    if np.abs(bem.Ctprime() - ctprime_target) > 0.001:
        return 6 * [np.nan]

    return np.rad2deg(pitch), tsr, np.rad2deg(yaw), bem.Cp(), bem.Ct(), bem.Ctprime()


def get_ctprime_setpoint_wrapper(x):
    yaw, angle = x
    dpitch = np.deg2rad(20) * np.cos(angle)
    dtsr = 5 * np.sin(angle)

    return get_ctprime_setpoint(CTPRIME, (pitch_opt, tsr_opt, yaw), (dpitch, dtsr, 0))


if __name__ == "__main__":
    yaws = np.deg2rad(np.linspace(0, 50, Nyaw))
    angles = np.linspace(0, 2 * np.pi, Nangle)
    X, Y = np.meshgrid(yaws, angles)

    params = list(zip(X.ravel(), Y.ravel()))
    out = for_each(get_ctprime_setpoint_wrapper, params, parallel=PARALLEL)

    df = pl.DataFrame(out, schema=["pitch", "tsr", "yaw", "Cp", "Ct", "Ctprime"])
    df_opt_by_yaw = (
        df.groupby("yaw", maintain_order=True)
        .agg(pl.all().sort_by("Cp").last())
        .select(["pitch", "tsr", "yaw"])
    )

    X = np.reshape(df["pitch"], X.shape)
    Y = np.reshape(df["tsr"], X.shape)
    Z = np.reshape(df["yaw"], X.shape)
    Cp = np.reshape(df["Cp"], X.shape)
    color_val = (Cp - np.nanmin(Cp)) / (np.nanmax(Cp - np.nanmin(Cp)))
    plt.figure()
    ax = plt.axes(projection="3d", computed_zorder=False)
    ax.plot_surface(
        X, Y, Z, facecolors=plt.cm.viridis(color_val), shade=False, zorder=2
    )
    ax.plot(
        df_opt_by_yaw["pitch"],
        df_opt_by_yaw["tsr"],
        df_opt_by_yaw["yaw"],
        "r",
        zorder=3,
    )

    ax.set_xlabel("pitch [deg]")
    ax.set_ylabel("$\lambda$ [-]")
    ax.set_zlabel("yaw [deg]")
    ax.set_title(f"$C_t'$={CTPRIME}")

    ax.view_init(elev=30, azim=-30)

    plt.savefig(
        figdir / f"ctprime_frontier_Ctp{CTPRIME}.png", dpi=300, bbox_inches="tight"
    )
