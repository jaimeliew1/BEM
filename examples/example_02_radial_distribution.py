from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    bem = BEM(IEA15MW())

    converged = bem.solve(pitch=0, tsr=8, yaw=0)

    to_plot = {
        "a": bem.a(agg="azim"),
        "phi": bem.phi(agg="azim"),
        "Vax": bem.Vax(agg="azim"),
        "Vtan": bem.Vtan(agg="azim"),
        "W": bem.W(agg="azim"),
        "Cax": bem.Cax(agg="azim"),
        "Ctan": bem.Ctan(agg="azim"),
        "Cl": bem.Cl(agg="azim"),
        "Cd": bem.Cd(agg="azim"),
        "Ct": bem.Ct(agg="azim"),
        "Ctprime": bem.Ctprime(agg="azim"),
        "Cp": bem.Cp(agg="azim"),
    }

    fig, axes = plt.subplots(len(to_plot), 1, sharex=True, figsize=(4, 10))

    for ax, (key, vals) in zip(axes, to_plot.items()):
        ax.plot(bem.mu, vals)
        ax.set_ylabel(key)

    plt.xlim(0, 1)

    axes[-1].set_xlabel("radius [-]")

    plt.savefig(
        FIGDIR / f"example_02_radial_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
