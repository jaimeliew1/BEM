from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW, IEA10MW, IEA3_4MW
from tqdm import tqdm

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


TSR = 8
PITCH = np.deg2rad(0)
YAW = np.deg2rad(40)
rotors = {
    "IEA15MW": IEA15MW(),
    "IEA10MW": IEA10MW(),
    # "IEA3.4MW": IEA3_4MW(),
}

to_plot = {
    "a": [],
    "aprime": [],
    "phi": [],
    "Vax": [],
    "Vtan": [],
    "W": [],
    "Cax": [],
    "Ctan": [],
    "Ct": [],
    "Cp": [],
    # "Fax": [10, 120],
    # "Ftan": [10, 120],
    # "thrust": [10, 120],
    # "power": [10, 120],
}
if __name__ == "__main__":
    for name, rotor in rotors.items():
        R = rotor.R
        data = rotor.solve(PITCH, TSR, YAW)

        fig, axes = plt.subplots(
            3, 4, figsize=2 * np.array([4, 3]), subplot_kw=dict(polar=True)
        )
        fig.tight_layout()

        # X, Y = data.mu_grid * np.cos(data.theta_grid), data.mu_grid * np.sin(
        #     data.theta_grid
        # )
        for ax, (key, args) in zip(axes.ravel(), to_plot.items()):
            # ax.plot_surface(X, Y, getattr(data, key)(*args), cmap="viridis")
            ax.pcolormesh(
                data.theta_grid,
                data.mu_grid,
                getattr(data, key)(*args),
                edgecolors="face",
            )
            ax.pcolormesh(
                data.theta_grid,
                data.mu_grid,
                getattr(data, key)(*args),
                edgecolors="face",
            )
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(figdir / f"3d_dist_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()
