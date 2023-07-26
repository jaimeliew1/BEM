from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from mit_bem.Turbine import IEA15MW
from mit_bem.Windfarm import BEMWindfarm


figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)
YAW = np.deg2rad(30)


X = [0, 4, 8]
# X = [0, 8, 4]
Y = [0, -0.5, 0.5]
pitches = [0, 0, 0]
tsrs = [8, 8, 8]
yaws = np.deg2rad([24, 10, 0])


def plot_field(
    X, Y, field, turb_xs, turb_ys, yaws, ax, cmap=None, vmin=None, vmax=None, title=None
):
    ax.imshow(
        field,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    for turb_x, turb_y, yaw in zip(turb_xs, turb_ys, yaws):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        rotmat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        p = rotmat @ p + np.array([[turb_x], [turb_y]])

        ax.plot(p[0, :], p[1, :], "k", lw=5)

    ax.text(
        0.5,
        0.98,
        title,
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )


if __name__ == "__main__":
    rotor = IEA15MW()

    windfarm = BEMWindfarm(X, Y, pitches, tsrs, yaws, rotor)

    print("REWS: ", [x.bem.REWS() for x in windfarm.turbines])
    print("Cp: ", [x.bem.Cp() for x in windfarm.turbines])
    print("Ctprime: ", [x.bem.Ctprime() for x in windfarm.turbines])

    xs = np.linspace(-1, 12, 300)
    ys = np.linspace(-1.5, 1.5, 300)
    xmesh, ymesh = np.meshgrid(xs, ys)

    U = windfarm.wsp(xmesh, ymesh)

    plt.figure()
    ax = plt.gca()
    plot_field(xs, ys, U, X, Y, yaws, ax)
    plt.savefig(figdir / "wake_simple_windfarm.png", dpi=300, bbox_inches="tight")
    plt.close()
