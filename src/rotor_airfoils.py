from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW


figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

N = 10
if __name__ == "__main__":
    rotor = IEA15MW()

    x = np.linspace(0, 1, N)
    inflow = np.linspace(-np.pi, np.pi, 100)

    x_mesh, inflow_mesh = np.meshgrid(x, inflow)

    cl, cd = rotor.airfoil(x_mesh, inflow_mesh)
    fig, axes = plt.subplots(2, 1, sharex=True)

    for i, (_cl, _cd) in enumerate(zip(cl.T, cd.T)):
        axes[0].plot(np.rad2deg(inflow), _cl, c=plt.cm.viridis(i / N))
        axes[1].plot(np.rad2deg(inflow), _cd, c=plt.cm.viridis(i / N))
    [ax.grid() for ax in axes]
    axes[0].set_ylabel("Cl")
    axes[1].set_ylabel("Cd")

    axes[0].set_xlim(-180, 180)
    axes[0].set_xticks(np.arange(-180, 181, 90))

    plt.savefig(figdir / "airfoils.png", dpi=300, bbox_inches="tight")
