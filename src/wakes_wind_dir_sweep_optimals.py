from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


data_fn = "cache/wind_dir_sweep.csv"


FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    df = pl.read_csv(data_fn)
    breakpoint()
    fig, axes = plt.subplots(4, 1, sharex=True)
    df_new = df.select(
        [
            "wdir",
            "Cp",
            pl.col("pitch1").alias("pitch"),
            pl.col("tsr1").alias("tsr"),
            pl.col("yaw1").alias("yaw"),
        ]
    )

    axes[0].plot(df_new["wdir"], df_new["Cp"])
    axes[1].plot(df_new["wdir"], df_new["pitch"])
    axes[2].plot(df_new["wdir"], df_new["tsr"])
    axes[3].plot(df_new["wdir"], df_new["yaw"])

    axes[0].set_ylabel("Farm $C_p$")
    axes[1].set_ylabel("$\\theta_p$")
    axes[2].set_ylabel("$\lambda$")
    axes[3].set_ylabel("$\gamma$ [deg]")

    axes[-1].set_xlabel("Wind direction [deg]")

    [ax.grid() for ax in axes]

    # axes[0].legend()

    axes[0].set_xlim(-10, 10)

    plt.savefig(FIGDIR / "wdir_sweep_optimal.png", dpi=300, bbox_inches="tight")
    plt.close()
