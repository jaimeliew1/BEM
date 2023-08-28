from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MITBEM.BEM import BEMSolver
from MITBEM.ReferenceTurbines import IEA15MW

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    bem = BEMSolver(IEA15MW())

    sol = bem.solve(pitch=0, tsr=8, yaw=0)

    to_plot = {
        "a": sol.a(agg="azim"),
        "phi": sol.phi(agg="azim"),
        "Vax": sol.Vax(agg="azim"),
        "Vtan": sol.Vtan(agg="azim"),
        "W": sol.W(agg="azim"),
        "Cax": sol.Cax(agg="azim"),
        "Ctan": sol.Ctan(agg="azim"),
        "Cl": sol.Cl(agg="azim"),
        "Cd": sol.Cd(agg="azim"),
        "Ct": sol.Ct(agg="azim"),
        "Ctprime": sol.Ctprime(agg="azim"),
        "Cp": sol.Cp(agg="azim"),
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
