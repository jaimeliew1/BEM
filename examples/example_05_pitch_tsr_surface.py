from pathlib import Path
import itertools
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

ROTOR = IEA15MW()


def func(x):
    pitch, tsr = x
    bem = BEM(ROTOR)
    bem.solve(pitch=np.deg2rad(pitch), tsr=tsr, yaw=0)

    return dict(pitch=pitch, tsr=tsr, Cp=bem.Cp(), Ct=bem.Ct())


if __name__ == "__main__":
    # Loop over grid points in parallel to create Cp grid.
    pitches = np.linspace(-20, 40)
    tsrs = np.linspace(0, 25)

    data = []
    params = list(itertools.product(pitches, tsrs))
    with Pool() as pool:
        for out in tqdm(pool.imap(func, params), total=len(params)):
            data.append(out)

    df = pl.from_dicts(data)

    # Format data for plotting
    df_piv = df.pivot(
        index="tsr", columns="pitch", values="Cp", aggregate_function=None
    )
    tsr = df_piv["tsr"].to_numpy()
    pitch = np.array(df_piv.columns[1:], dtype=float)
    Cp = df_piv.to_numpy()[:, 1:]

    # Plotting
    fig, ax = plt.subplots(1, 1)
    levels = np.arange(0.0, 0.61, 0.1)
    ax.contourf(pitch, tsr, Cp, levels=levels, cmap="viridis", vmin=0, vmax=0.6)
    CS = ax.contour(pitch, tsr, Cp, levels=levels, colors="k")
    ax.clabel(CS, inline=True, fontsize=10)

    ax.set_xlabel("Pitch [deg]")
    ax.set_ylabel("Tip Speed Ratio [-]")
    ax.set_title("Power Coefficient")
    plt.savefig(
        FIGDIR / "example_05_pitch_tsr_surface.png", dpi=300, bbox_inches="tight"
    )
