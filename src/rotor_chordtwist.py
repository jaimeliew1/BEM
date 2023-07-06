from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW


figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    rotor = IEA15MW()

    x = np.linspace(0, 1, 100)

    twist = rotor.twist(x)
    chord = rotor.chord(x)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(x, np.rad2deg(twist))
    axes[1].plot(x, chord)

    axes[0].set_ylabel("twist [deg]")
    axes[1].set_ylabel("chord [m]")

    [ax.grid() for ax in axes]

    axes[0].set_xlim(0, 1)

    plt.savefig(figdir / "chordtwist.png", dpi=300, bbox_inches="tight")
