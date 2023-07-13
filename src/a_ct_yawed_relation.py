from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import polars as pl

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


def CtA_mike(a, yaw):
    Ct = 4 * a * (1 - a) / (1 + 0.25 * (1 - a) ** 2 * np.sin(yaw) ** 2)
    return Ct


def CtA_glauert(a, yaw):
    Ct = 4 * a * np.sqrt(1 - a * (2 * np.cos(yaw) - a))
    return Ct


def CtA_vortex(a, yaw):
    X = (0.6 * a + 1) * yaw
    Ct = 4 * a * (np.cos(yaw) + np.tan(X / 2) * np.sin(yaw) - a / np.cos(X / 2) ** 2)
    return Ct


def aCt_HAWC2(Ct, yaw):
    Ct = np.minimum(Ct, 0.9)
    k3 = -0.6481 * yaw**3 + 2.1667 * yaw**2 - 2.0705 * yaw
    k2 = 0.8646 * yaw**3 - 2.6145 * yaw**2 + 2.1735 * yaw
    k1 = -0.1640 * yaw**3 + 0.4438 * yaw**2 - 0.5136 * yaw
    Ka = k3 * Ct**3 + k2 * Ct**2 + k1 * Ct + 1

    a = Ka * (0.0883 * Ct**3 + 0.0586 * Ct**2 + 0.246 * Ct)

    return a


def CtA_HAWC2(ass, yaw):
    Cts = np.zeros_like(ass)

    for i, a in enumerate(ass):

        def residual(Ct, yaw):
            return aCt_HAWC2(Ct, yaw) - a

        res = root_scalar(residual, (yaw), x0=0, x1=3)
        if res.converged:
            Cts[i] = res.root

    return Cts


methods = {
    "mike": CtA_mike,
    "glauert": CtA_glauert,
    "vortex": CtA_vortex,
    "HAWC2": aCt_HAWC2,
}

fn_mike = Path(__file__).parent.parent / "yaw_turbU.csv"


def load_LES():
    df = (
        pl.read_csv(fn_mike)
        .rename({"": "yaw"})
        .melt(id_vars="yaw", variable_name="CT_p", value_name="U_d")
        .select([pl.col("yaw"), pl.col("CT_p").cast(float), pl.col("U_d")])
    ).with_columns(
        [
            (pl.col("CT_p") * pl.col("U_d") ** 2).alias("CT"),
            ((1 - pl.col("U_d") / np.cos(np.deg2rad(pl.col("yaw"))))).alias("a"),
        ]
    )
    return df


def color(angle):
    return plt.cm.viridis(angle / 60)


if __name__ == "__main__":
    yaws = np.deg2rad(np.arange(0, 60, 10))

    # fig, axes = plt.subplots(1, len(methods), sharey=True, sharex=True, figsize=(10, 4))
    fig, axes = plt.subplots(1, 1)
    axes = [axes]
    for ax, (method, func) in zip(axes, methods.items()):
        for i, yaw in enumerate(yaws):
            if method == "HAWC2":
                Ct = np.linspace(0, 3, 100)
                a = func(Ct, yaw)
            else:
                a = np.linspace(0, 1, 100)
                Ct = func(a, yaw)
            ax.plot(
                a,
                Ct,
                c=color(np.rad2deg(yaw)),
                label=f"{np.rad2deg(yaw):2.0f}° (AD)",
            )

        # ax.set_title(method)
        ax.set_xlabel("a")
    axes[0].set_ylabel("$C_T$")

    plt.ylim(0, 1.5)
    plt.xlim(0, 0.5)

    # Plot LES data
    df = load_LES()
    print(df)

    for yaw in np.rad2deg(yaws):
        df_ = df.filter(pl.col("yaw") == yaw)
        axes[0].plot(
            df_["a"], df_["CT"], "--", c=color(yaw), label=f"{yaw:2.0f}° (LES)"
        )

    axes[-1].legend(title="$\gamma$", ncol=6, fontsize="xx-small")
    plt.savefig(figdir / f"CtA.png", dpi=300, bbox_inches="tight")
