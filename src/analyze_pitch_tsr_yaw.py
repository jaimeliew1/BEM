from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm


figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

data_fn = Path(f"cache/mike/tsr_pitch_yaw_IEA15MW.csv")


if __name__ == "__main__":
    df_all = pl.read_csv(data_fn)
    print(df_all)

    # Ct-a relationship
    df = df_all.select(["Ct", "a", "yaw"]).sort(["yaw", "a"])

    plt.figure()
    for yaw, _df in df.groupby("yaw"):
        plt.plot(_df["a"], _df["Ct"], color=plt.cm.viridis(yaw / 60), label=f"{yaw}")

    plt.xlabel("a")
    plt.ylabel("$C_T$")

    plt.legend(title="yaw")

    plt.xlim(0, 1)
    plt.ylim(0, 2)

    plt.savefig(figdir / "analyze_ct_a.png", dpi=300, bbox_inches="tight")

    # Ctprime-a relationship
    df = df_all.select(["Ctprime", "a", "yaw"]).sort(["yaw", "a"])

    plt.figure()
    for yaw, _df in df.groupby("yaw"):
        plt.plot(
            _df["a"], _df["Ctprime"], color=plt.cm.viridis(yaw / 60), label=f"{yaw}"
        )

    plt.xlabel("a")
    plt.ylabel("$C_T'$")

    plt.legend(title="yaw")

    plt.xlim(0, 1)
    plt.ylim(0, 4)

    plt.savefig(figdir / "analyze_ctprime_a.png", dpi=300, bbox_inches="tight")

    # Ctprime-pitch relationship
    df = df_all.select(["Ctprime", "pitch", "yaw"]).sort(["yaw", "pitch"])

    plt.figure()
    for yaw, _df in df.groupby("yaw"):
        plt.plot(
            _df["pitch"],
            _df["Ctprime"],
            ".",
            color=plt.cm.viridis(yaw / 60),
            label=f"{yaw}",
        )

    plt.xlabel("pitch")
    plt.ylabel("$C_T'$")

    plt.legend(title="yaw")

    plt.savefig(figdir / "analyze_ctprime_pitch.png", dpi=300, bbox_inches="tight")
