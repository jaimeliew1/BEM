from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW


figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    rotor = IEA15MW()

    x = np.linspace(0, 1, 100)

    twist = rotor.bem.twist(x)
    solidity = rotor.bem.solidity(x)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(x, np.rad2deg(twist))
    axes[1].plot(x, solidity)

    axes[0].set_ylabel("twist [deg]")
    axes[1].set_ylabel("solidity [m]")

    [ax.grid() for ax in axes]

    axes[0].set_xlim(0, 1)

    plt.savefig(figdir / "soliditytwist.png", dpi=300, bbox_inches="tight")
