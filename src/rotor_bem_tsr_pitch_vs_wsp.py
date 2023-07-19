from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from mit_bem.Turbine import IEA10MW, IEA15MW

pl.Config.set_tbl_rows(40)
np.seterr(all="raise")
figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


cache_fn = Path("cache/mike_corrected/tsr_pitch_yaw_IEA15MW.csv")


rotors = {
    "IEA15MW": IEA15MW(),
    # "IEA3.4MW": IEA3_4MW(),
}
R = 120.96991314691175
rho = 1.293
P_rated = 15000000
omega_max = 0.7916813487046278


def get_pitch_tsr_yaw_surface(cache_fn):
    if cache_fn.exists():
        df = pl.read_csv(cache_fn)

    else:
        print(f"{cache_fn} not found!")

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
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True)

    axes[0].set_xlim(-15, 35)
    axes[0].set_ylim(0, 19)

    pitch_mesh, tsr_mesh = np.meshgrid(pitch, tsr)
    # idx_tsr, idx_pitch = np.unravel_index(np.nanargmax(CP), CP.shape)
    # Cp
    if Cp_norm is None:
        levels = np.arange(0.0, 0.61, 0.1)
    else:
        CP /= Cp_norm
        levels = np.arange(0, 1.01, 0.1)
    plot_surface_ax(pitch_mesh, tsr_mesh, CP, axes[0], levels)
    # axes[0].plot(pitch[idx_pitch], tsr[idx_tsr], "*")

    # CT
    levels = np.arange(0, 3.1, 0.5)
    plot_surface_ax(
        pitch_mesh,
        tsr_mesh,
        CT,
        axes[1],
        levels=levels,
        cmap="plasma",
    )

    # CTprime
    levels = np.arange(0, 4.1, 0.5)
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


def plot_surfaces_by_wsp(df_all):
    for U, df in tqdm(df_all.filter(pl.col("yaw") == 0).groupby("U")):
        df_Cp = (
            df.pivot(index="tsr", columns="pitch", values="Cp", aggregate_function=None)
            .sort("tsr")
            .interpolate()
        )
        df_Ct = (
            df.pivot(index="tsr", columns="pitch", values="Ct", aggregate_function=None)
            .sort("tsr")
            .interpolate()
        )
        df_Ctprime = (
            df.pivot(
                index="tsr", columns="pitch", values="Ctprime", aggregate_function=None
            )
            .sort("tsr")
            .interpolate()
        )
        df_a = (
            df.pivot(
                index="tsr", columns="pitch", values="a", aggregate_function=None
            )
            .sort("tsr")
            .interpolate()
        )

        tsr = df_Cp["tsr"].to_numpy()
        pitch = np.array(df_Cp.columns[1:], dtype=float)
        Cp = df_Cp.to_numpy()[:, 1:]
        Ct = df_Ct.to_numpy()[:, 1:]
        Ctprime = df_Ctprime.to_numpy()[:, 1:]
        a = df_a.to_numpy()[:, 1:]
        fn_out = figdir / f"tsr_pitch_vs_wsp{U:2.2f}.png"

        plot_Cp_Ct_surfaces(
            pitch,
            tsr,
            Cp,
            Ct,
            # Ctprime,
            a,
            title=f"$U$={U}m/s",
            save=fn_out,
        )


def plot_constraints_by_wsp_and_yaw(df_all):
    fig, axes = plt.subplots(4, 1, sharex=True)

    for yaw, _df in df_all.sort("yaw").groupby("yaw"):
        df = (
            _df.groupby("U")
            .agg(
                [
                    (pl.col("Cp").max()).suffix("_max"),
                    (pl.col("Ctprime").max()).suffix("_max"),
                    (pl.col("tsr").max()).suffix("_max"),
                    (pl.col("a").max()).suffix("_max"),
                ]
            )
            .sort("U")
        )
        axes[0].plot(
            df["U"], df["Ctprime_max"], c=plt.cm.viridis(yaw / 50), label=f"{yaw}"
        )
        axes[1].plot(df["U"], df["Cp_max"], c=plt.cm.viridis(yaw / 50), label=f"{yaw}")
        axes[2].plot(df["U"], df["tsr_max"], c=plt.cm.viridis(yaw / 50), label=f"{yaw}")
        axes[3].plot(df["U"], df["a_max"], c=plt.cm.viridis(yaw / 50), label=f"{yaw}")

    axes[0].set_ylabel("$C_T'$ max")
    axes[1].set_ylabel("$C_p$ max")
    axes[2].set_ylabel("$\lambda$ max")
    axes[3].set_ylabel("$a$ max")

    axes[0].set_ylim(0, 4)

    axes[-1].set_xlabel("Wind speed [m/s]")

    axes[0].legend(
        bbox_to_anchor=(0.5, 1.05),
        loc="lower center",
        ncol=df_all["yaw"].n_unique(),
        title="yaw [deg]",
    )

    plt.savefig(figdir / "constraints_vs_wsp.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df_all = get_pitch_tsr_yaw_surface(cache_fn)

    df_all = df_all.with_columns(
        [
            np.maximum(pl.col("Cp"), 0.001),
            np.maximum(pl.col("Ct"), 0.001),
            np.maximum(pl.col("Ctprime"), 0.001),
        ]
    )
    df_list = []
    for U in np.arange(4, 25, 0.5):
        df = (
            df_all.with_columns(
                [
                    (pl.col("tsr") * U / R).alias("omega"),
                    (0.5 * rho * U**3 * np.pi * R**2 * pl.col("Cp")).alias("P"),
                ]
            )
            .filter((pl.col("omega") < omega_max) & (pl.col("P") < P_rated))
            .drop_nulls()
        ).with_columns(U=U)
        df_list.append(df)

    df_all = pl.concat(df_list)

    plot_surfaces_by_wsp(df_all)
    plot_constraints_by_wsp_and_yaw(df_all)
