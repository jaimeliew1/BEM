from pathlib import Path
import itertools
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from tqdm import tqdm
import polars as pl

from mit_bem.Turbine import IEA15MW
from mit_bem.Windfarm import BEMWindfarm
from Optimizations import JointOptimization

np.seterr(all="ignore")
FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = Path("cache")
CACHEDIR.mkdir(parents=True, exist_ok=True)

gridsearch_fn = CACHEDIR / "wakes_2_turbine_optimal.csv"

X = [0, 8]
Y = [0, 0.5]

pitch2, tsr2, yaw2 = 0, 8, 0

rotor = IEA15MW()


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


def _gridsearch_func(x):
    pitch1, tsr1, yaw1 = x
    pitches, tsrs, yaws = [pitch1, pitch2], [tsr1, tsr2], [yaw1, yaw2]
    farm = BEMWindfarm(X, Y, pitches, tsrs, yaws, rotor)

    Cp1 = farm.turbines[0].bem.Cp()
    Cp2 = farm.turbines[1].bem.Cp()
    Cptot = np.mean([Cp1, Cp2])

    return dict(
        pitch=np.round(np.rad2deg(pitch1), 2),
        tsr=tsr1,
        yaw=np.round(np.rad2deg(yaw1), 2),
        Cp1=Cp1,
        Cp2=Cp2,
        Cptot=Cptot,
    )


def gridsearch():
    pitches = np.deg2rad(np.arange(-10, 10.1, 1))
    tsrs = np.arange(3, 15.1, 1)
    yaws = np.deg2rad(np.arange(-30, 30.1, 5))

    params = list(itertools.product(pitches, tsrs, yaws))

    outs = for_each(_gridsearch_func, params, parallel=True)

    df = pl.from_dicts(outs)

    return df


def plot_surface_ax(
    pitch, tsr, Z, ax, levels=None, cmap="viridis", vmin=None, vmax=None
):
    ax.contourf(pitch, tsr, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    CS = ax.contour(pitch, tsr, Z, levels=levels, colors="k")
    ax.clabel(CS, inline=True, fontsize=10)


def run():
    optimization = JointOptimization(X, Y, wf_kwargs=dict(rotor=rotor))
    Cp, setpoints = optimization.find_optimal()
    pitch_opt, tsr_opt, yaw_opt = np.split(setpoints, 3)
    pitch_opt, yaw_opt = np.rad2deg(pitch_opt), np.rad2deg(yaw_opt)

    if gridsearch_fn.exists():
        df = pl.read_csv(gridsearch_fn)
    else:
        df = gridsearch()
        df.write_csv(gridsearch_fn)

    print(df)

    for i, (yaw, _df) in enumerate(df.sort("yaw").groupby("yaw")):
        df_Cp = (
            _df.pivot(
                index="tsr", columns="pitch", values="Cptot", aggregate_function=None
            )
            .sort("tsr")
            .interpolate()
        )

        tsr = df_Cp["tsr"].to_numpy()
        pitch = np.array(df_Cp.columns[1:], dtype=float)
        Cp = df_Cp.to_numpy()[:, 1:]
        fn_out = FIGDIR / f"wakes_2_turbine_optimal_yaw_{i}_{yaw:05.2f}.png"

        plt.figure()
        ax = plt.gca()
        levels = np.arange(0.0, 0.451, 0.025)
        plot_surface_ax(pitch, tsr, Cp, ax, levels)

        plt.xlabel("Blade pitch [deg]")
        plt.ylabel("Tip Speed Ratio")
        plt.title(f"Yaw = {yaw} degrees")

        plt.savefig(fn_out, dpi=300, bbox_inches="tight")
        plt.close()

    return pitch_opt, tsr_opt, yaw_opt, Cp_opt


if __name__ == "__main__":
    pitch_opt, tsr_opt, yaw_opt, Cp_opt = run()
    print("pitch_opt: ", pitch_opt)
    print("tsr_opt: ", tsr_opt)
    print("yaw_opt: ", yaw_opt)
    print("Cp_opt: ", Cp_opt)
