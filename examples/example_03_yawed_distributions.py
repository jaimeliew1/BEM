from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW

FIGDIR = Path("fig")
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
        "a": bem.a(agg="segment"),
        "phi": bem.phi(agg="segment"),
        "Vax": bem.Vax(agg="segment"),
        "Vtan": bem.Vtan(agg="segment"),
        "W": bem.W(agg="segment"),
        "Cax": bem.Cax(agg="segment"),
        "Ctan": bem.Ctan(agg="segment"),
        "Cl": bem.Cl(agg="segment"),
        "Cd": bem.Cd(agg="segment"),
        "Ct": bem.Ct(agg="segment"),
        "Ctprime": bem.Ctprime(agg="segment"),
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
    fig, axes = plt.subplots(4, 3, subplot_kw=dict(projection="3d"), figsize=(8, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.6)

    # Plot each surface. Remove axes
    for ax, (key, vals) in zip(axes.ravel(), to_plot.items()):
        vals[r_mesh > 0.98] = np.nan
        ax.plot_surface(x_mesh, y_mesh, vals, cmap="viridis", linewidth=0)
        ax.set_title(labels[key])
        ax.set_axis_off()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    bem = BEM(IEA15MW(), Nr=100, Ntheta=100)

    converged = bem.solve(pitch=0, tsr=8, yaw=np.deg2rad(45))

    fn = FIGDIR / f"example_03_yawed_distributions.png"

    plot_3d_distributions(bem, save=fn)
