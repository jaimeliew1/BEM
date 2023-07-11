from pathlib import Path
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW, IEA10MW, IEA3_4MW
from tqdm import tqdm
import polars as pl
from a_ct_yawed_relation import load_LES

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


def CtA_mike(a, yaw):
    Ct = 4 * a * (1 - a) / (1 + 0.25 * (1 - a) ** 2 * np.sin(yaw) ** 2)
    return Ct


tsrs = np.arange(4, 15)
PITCH = np.deg2rad(0)
yaws = np.deg2rad(np.arange(0, 21, 5))

rotors = {
    "IEA15MW": IEA15MW(),
    "IEA10MW": IEA10MW(),
}

to_plot = ["a", "Ct", "Cp"]
if __name__ == "__main__":
    params = list(product(rotors, tsrs, yaws))
    out = []
    N = len(params)
    for rotor_name, tsr, yaw in tqdm(params):
        data = rotors[rotor_name].solve(PITCH, tsr, yaw)
        if data.status == "converged":
            out.append(
                (
                    rotor_name,
                    tsr,
                    np.rad2deg(yaw),
                    data.a("rotor"),
                    data.Ct("rotor"),
                    data.Cp("rotor"),
                )
            )
        # else:
        #     print(rotor_name, tsr, np.rad2deg(yaw))
    df = pl.DataFrame(out, schema=["rotor", "tsr", "yaw", "a", "Ct", "Cp"])
    df = df.with_columns(
        [
            # pl.col("a") / np.cos(np.deg2rad(pl.col("yaw"))),
            (
                pl.col("Ct")
                / ((1 - pl.col("a")) ** 2 * np.cos(np.deg2rad(pl.col("yaw"))) ** 2)
            ).alias("Ct_p")
        ]
    )
    print(df)

    for rotor, _df in df.groupby("rotor"):
        fig, axes = plt.subplots(len(to_plot), 1)
        axes[-1].set_xlabel("$\lambda$")

        for ax, key in zip(axes, to_plot):
            for yaw, __df in _df.groupby("yaw"):
                ax.plot(__df["tsr"], __df[key], c=plt.cm.viridis(yaw / 20))
            ax.set_ylabel(key)

        plt.savefig(figdir / f"vs_yaw_{rotor}.png", dpi=300, bbox_inches="tight")
        plt.close()

    for rotor, _df in df.groupby("rotor"):
        fig, axes = plt.subplots(1, 1)
        plt.xlabel("a")
        plt.ylabel("$C_T'$")

        for yaw, __df in _df.groupby("yaw"):
            plt.plot(__df["a"], __df["Ct_p"], c=plt.cm.viridis(yaw / 20))

            Ct_mike = CtA_mike(__df["a"].to_numpy(), np.deg2rad(yaw))
            plt.plot(__df["a"], Ct_mike, "--", c=plt.cm.viridis(yaw / 20))
        ax.set_ylabel(key)

        # plot LES data
        df_les = load_LES()
        for yaw in df_les["yaw"].unique():
            df_ = df.filter(pl.col("yaw") == yaw)
            plt.plot(df_["a"], df_["Ct_p"], ":", c=plt.cm.viridis(yaw / 20))
        plt.savefig(figdir / f"vs_yaw_a_Ct_{rotor}.png", dpi=300, bbox_inches="tight")
        plt.close()
