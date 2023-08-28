from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW


FIGDIR = Path("fig/yawed_distributions")
FIGDIR.mkdir(exist_ok=True, parents=True)

labels = {
    "a": "$a$",
    "phi": "$\phi$",
    "Vax": "$V_{ax}$",
    "Vtan": "$V_{tan}$",
    "W": "$W$",
    "Cax": "$C_{ax}$",
    "Ctan": "$C_{tan}$",
    "Cl": "$C_L$",
    "Cd": "$C_D$",
    "Ct": "$C_T$",
    "Ctprime": "$C_T'$",
    "Cp": "$C_p$",
}


def plot_3d_distributions(bem, save=None):
    to_plot = {
        "Cax": bem.Cax(agg="segment"),
        "Ctan": bem.Ctan(agg="segment"),
        "Ct": bem.Ct(agg="segment"),
        "Cp": bem.Cp(agg="segment"),
    }
    r_mesh, theta_mesh = bem.mu_mesh, bem.theta_mesh

    # repeat first azimuthal data points.
    r_mesh = np.hstack([r_mesh, r_mesh[:, 0].reshape([-1, 1])])
    theta_mesh = np.hstack([theta_mesh, theta_mesh[:, 0].reshape([-1, 1])])
    to_plot = {
        key: np.hstack([val, val[:, 0].reshape([-1, 1])])
        for key, val in to_plot.items()
    }

    x_mesh, y_mesh = r_mesh * np.cos(theta_mesh), r_mesh * np.sin(theta_mesh)

    # Set up subplots and spacing
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection="3d"), figsize=(8, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.6)

    # Plot each surface. Remove axes
    for ax, (key, vals) in zip(axes.ravel(), to_plot.items()):
        vals[r_mesh > 0.98] = np.nan
        ax.plot_surface(x_mesh, y_mesh, vals, cmap="viridis", lw=0)
        ax.set_title(labels[key])
        ax.set_axis_off()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    bem = BEM(IEA15MW(), Nr=100, Ntheta=100)

    yaws = np.arange(-50, 51, 1)

    for i, yaw in enumerate(tqdm(yaws)):
        converged = bem.solve(pitch=0, tsr=8, yaw=np.deg2rad(yaw))

        fn = FIGDIR / f"yawed_distributions_{i:03}.png"

        plot_3d_distributions(bem, save=fn)
