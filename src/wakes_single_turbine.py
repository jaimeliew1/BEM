from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from mit_bem.Turbine import IEA15MW
from mit_bem.Windfarm import BEMTurbine
from mit_bem.Windfield import Custom

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)
YAW = np.deg2rad(30)

if __name__ == "__main__":
    rotor = IEA15MW()
    turbine = BEMTurbine(0, 7, YAW, rotor, 0, 0)

    mu_mesh, theta_mesh = turbine.gridpoints()

    base_U = np.ones_like(mu_mesh)
    windfield = Custom(base_U)

    # wake summation stuff goes here

    turbine.post_init(windfield)

    xs = np.linspace(-1, 8, 300)
    ys = np.linspace(-1.5, 1.5, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    deficit = turbine.deficit(xmesh, ymesh)

    plt.figure()
    ax = plt.gca()
    ax.imshow(
        deficit,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap="YlGnBu_r",
    )

    # Draw turbine
    turb_x, turb_y = 0, 0
    R = 0.5
    p = np.array([[turb_x, turb_x], [turb_y + R, turb_y - R]])
    rotmat = np.array([[np.cos(YAW), -np.sin(YAW)], [np.sin(YAW), np.cos(YAW)]])

    p = rotmat @ p

    ax.plot(p[0, :], p[1, :], "k", lw=5)
    plt.savefig(figdir / "test.png", dpi=300, bbox_inches="tight")
