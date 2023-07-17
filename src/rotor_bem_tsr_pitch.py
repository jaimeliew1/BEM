from itertools import product, repeat
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from mit_bem.Turbine import IEA3_4MW, IEA10MW, IEA15MW
from mit_bem.BEM import BEM
from find_optimal_setpoint import find_optimal
from power_curve_noyaw import setpoint_trajectory

np.seterr(all="raise")
figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

PARALLEL = True
METHOD = "HAWC2"
METHOD = "mike"
# METHOD = "mike_corrected"

cachedir = Path(f"cache/{METHOD}")
cachedir.mkdir(exist_ok=True, parents=True)


rotors = {
    "IEA15MW": IEA15MW(),
    "IEA10MW": IEA10MW(),
    # "IEA3.4MW": IEA3_4MW(),
}


def func(x):
    pitch, tsr, yaw, rotor_name = x
    bem = BEM(rotors[rotor_name], Cta_method=METHOD, tiploss="prandtl")
    # bem = rotors["IEA15MW"].solve(pitch, tsr, yaw, method=METHOD)
    converged = bem.solve(pitch, tsr, yaw)
    if converged:
        Ct = bem.Ct("rotor")
        Cp = bem.Cp("rotor")
        Ctprime = bem.Ctprime("rotor")
        a = bem.a("rotor")
    else:
        Ct, Cp, Ctprime, a = np.nan, np.nan, np.nan, np.nan

    return dict(
        pitch=np.round(np.rad2deg(pitch), 2),
        tsr=tsr,
        yaw=np.round(np.rad2deg(yaw), 2),
        Ct=Ct,
        Cp=Cp,
        Ctprime=Ctprime,
        a=a,
    )


# def func_IEA10MW(x):
#     pitch, tsr, yaw = x
#     bem = rotors["IEA10MW"].solve(pitch, tsr, yaw, method=METHOD)
#     if bem.status == "converged":
#         Ct = bem.Ct("rotor")
#         Cp = bem.Cp("rotor")
#         Ctprime = bem.Ctprime("rotor")
#     else:
#         Ct, Cp, Ctprime = np.nan, np.nan, np.nan

#     return dict(
#         pitch=np.round(np.rad2deg(pitch), 2),
#         tsr=tsr,
#         yaw=np.round(np.rad2deg(yaw), 2),
#         Ct=Ct,
#         Cp=Cp,
#         Ctprime=Ctprime,
#     )


# funcs = {
#     "IEA15MW": func_IEA15MW,
#     "IEA10MW": func_IEA10MW,
# }


pitches = np.deg2rad(np.arange(-15, 30, 1))
tsrs = np.arange(0, 20, 0.5)
yaws = np.deg2rad(np.arange(0, 50, 10))
# yaws = [0]

params = list(product(pitches, tsrs, yaws))


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


def get_pitch_tsr_yaw_surface(rotor_name, cache_fn):
    if cache_fn.exists():
        df = pl.read_csv(cache_fn)

    else:
        df = generate_pitch_tsr_yaw_surface(rotor_name)
        df.write_csv(cache_fn)

    return df


def generate_pitch_tsr_yaw_surface(rotor_name):
    _params = [(*p, rotor_name) for p in params]
    out = for_each(func, _params, parallel=PARALLEL)
    df = pl.from_dicts(out)

    return df


def plot_surface_ax(
    pitch, tsr, Z, ax, levels=None, cmap="viridis", vmin=None, vmax=None
):
    ax.contourf(pitch, tsr, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    CS = ax.contour(pitch, tsr, Z, levels=levels, colors="k")
    ax.clabel(CS, inline=True, fontsize=10)

    # [ax.plot(np.rad2deg(pitch_best), tsr_best, "*") for ax in axes]


def plot_Cp_Ct_surfaces(
    pitch, tsr, CP, CT, CTPrime, title=None, setpoints=None, Cp_norm=None, save=None
):
    fig, axes = plt.subplots(1, 3, sharey=True)
    pitch_mesh, tsr_mesh = np.meshgrid(pitch, tsr)
    idx_tsr, idx_pitch = np.unravel_index(np.nanargmax(CP), CP.shape)
    # Cp
    if Cp_norm is None:
        levels = np.arange(0, 0.6, 0.05)
    else:
        CP /= Cp_norm
        levels = np.arange(0, 1.01, 0.1)
    plot_surface_ax(pitch_mesh, tsr_mesh, CP, axes[0], levels)
    axes[0].plot(pitch[idx_pitch], tsr[idx_tsr], "*")

    # CT
    levels = np.arange(0, 2.1, 0.2)
    plot_surface_ax(
        pitch_mesh,
        tsr_mesh,
        CT,
        axes[1],
        levels=levels,
        cmap="plasma",
    )

    # CTprime
    levels = np.arange(0, 10, 1)
    plot_surface_ax(
        pitch_mesh,
        tsr_mesh,
        CTPrime,
        axes[2],
        levels=levels,
        cmap="plasma",
    )

    # Control setpoint trajectory
    if setpoints:
        sp_pitch, sp_tsr = setpoints
        axes[0].plot(sp_pitch, sp_tsr)
        axes[1].plot(sp_pitch, sp_tsr)
        axes[2].plot(sp_pitch, sp_tsr)

    axes[0].set_title("$C_p$")
    axes[1].set_title("$C_T$")
    axes[2].set_title("$C_T'$")

    axes[0].set_ylabel("$\lambda$ [-]")

    fig.suptitle(title)

    [ax.set_xlabel(r"$\theta_p$ [deg]") for ax in axes]

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for rotor_name in rotors.keys():
        cache_fn = cachedir / f"tsr_pitch_yaw_{rotor_name}.csv"

        # tsr_opt, pitch_opt, Cp_opt = find_optimal(rotors[rotor_name])

        df_all = get_pitch_tsr_yaw_surface(rotor_name, cache_fn)
        # sp_pitch, sp_tsr = setpoint_trajectory(rotors[rotor_name])

        df_all = df_all.filter(pl.col("tsr") != 0)
        for yaw, df in df_all.groupby("yaw"):
            df_Cp = df.pivot(
                index="tsr", columns="pitch", values="Cp", aggregate_function=None
            )
            df_Ct = df.pivot(
                index="tsr", columns="pitch", values="Ct", aggregate_function=None
            )
            df_Ctprime = df.pivot(
                index="tsr", columns="pitch", values="Ctprime", aggregate_function=None
            )

            tsr = df_Cp["tsr"].to_numpy()
            pitch = np.array(df_Cp.columns[1:], dtype=float)
            Cp = df_Cp.to_numpy()[:, 1:]
            Ct = df_Ct.to_numpy()[:, 1:]
            Ctprime = df_Ctprime.to_numpy()[:, 1:]
            fn_out = figdir / f"tsr_pitch_{rotor_name}_{METHOD}_{yaw}.png"

            plot_Cp_Ct_surfaces(
                pitch,
                tsr,
                Cp,
                Ct,
                Ctprime,
                title=f"$\gamma$={yaw}°",
                # setpoints=(sp_pitch, sp_tsr),
                # Cp_norm=Cp_opt,
                save=fn_out,
            )
