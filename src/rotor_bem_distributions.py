from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW, IEA10MW, IEA3_4MW
from tqdm import tqdm

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


tsr_min, tsr_max = 4, 15
PITCH = np.deg2rad(0)
YAW = np.deg2rad(0)

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
    # "Vtan": [],
    "W": [],
    "Cax": [],
    "Ctan": [],
    "Ct": [],
    "Cp": [],
    "Fax": [10, 120],
    "Ftan": [10, 120],
    "thrust": [10, 120],
    "power": [10, 120],
}
if __name__ == "__main__":
    for name, rotor in rotors.items():
        R = rotor.R
        tsrs = np.arange(tsr_min, tsr_max, 1)
        data = []
        for tsr in tqdm(tsrs):
            out = rotor.solve(PITCH, tsr, YAW)
            print(out.status)
            data.append(out)

        print(np.max([x.Cp("rotor") for x in data]))

        fig, axes = plt.subplots(
            len(to_plot), 1, sharex=True, figsize=np.array([4, 10])
        )

        for i, (tsr, dat) in enumerate(zip(tsrs, data)):
            for ax, (key, args) in zip(axes, to_plot.items()):
                ax.plot(
                    dat.mu,
                    getattr(dat, key)(*args, "azim"),
                    color=plt.cm.viridis(i / len(tsrs)),
                )
                ax.set_ylabel(key)

        plt.xlim(0, 1)

        plt.savefig(figdir / f"radius_vs {name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(
            3, 4, subplot_kw=dict(projection="3d"), figsize=2 * np.array([4, 3])
        )
        fig.tight_layout()

        dat = [x for x in data if x.tsr == 8][0]
        X, Y = dat.mu_grid * np.cos(dat.theta_grid), dat.mu_grid * np.sin(
            dat.theta_grid
        )
        for ax, (key, args) in zip(axes.ravel(), to_plot.items()):
            ax.plot_surface(X, Y, getattr(dat, key)(*args), cmap="viridis")
            ax.set_title(key)
        plt.savefig(figdir / f"3d_dist_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()
