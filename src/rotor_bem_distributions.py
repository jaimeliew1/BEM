from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW, IEA10MW, IEA3_4MW
from mit_bem.BEM import BEM
from tqdm import tqdm

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

METHOD = "HAWC2"
# METHOD = "mike_corrected"

tsr_min, tsr_max = 4, 12
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
    "Ctprime": [],
    "Cp": [],
    "Fax": [10],
    "Ftan": [10],
    "thrust": [10],
    "power": [10],
}
if __name__ == "__main__":
    for name, rotor in rotors.items():
        R = rotor.R
        tsrs = np.arange(tsr_min, tsr_max, 1)
        data = []
        for tsr in tqdm(tsrs):
            bem = BEM(rotor, Cta_method=METHOD)
            converged = bem.solve(PITCH, tsr, YAW)
            data.append(bem)

        print(np.max([x.Cp() for x in data]))

        fig, axes = plt.subplots(
            len(to_plot), 1, sharex=True, figsize=np.array([4, 10])
        )
        axes[0].set_title(METHOD)

        for i, (tsr, dat) in enumerate(zip(tsrs, data)):
            for ax, (key, args) in zip(axes, to_plot.items()):
                ax.plot(
                    dat.mu,
                    getattr(dat, key)(*args, "azim"),
                    color=plt.cm.viridis(i / len(tsrs)),
                    label=f"{tsr}",
                )
                ax.set_ylabel(key)

        plt.xlim(0, 1)

        axes[-1].set_xlabel("radius [-]")

        axes[0].legend(title="TSR", loc="upper left", bbox_to_anchor=(1.05, 0))

        plt.savefig(
            figdir / f"radius_vs_{name}_{METHOD}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        fig, axes = plt.subplots(
            3, 4, subplot_kw=dict(projection="3d"), figsize=2 * np.array([4, 3])
        )
        fig.tight_layout()

        dat = [x for x in data if x.tsr == 8][0]
        X, Y = dat.mu_mesh * np.cos(dat.theta_mesh), dat.mu_mesh * np.sin(
            dat.theta_mesh
        )
        for ax, (key, args) in zip(axes.ravel(), to_plot.items()):
            ax.plot_surface(X, Y, getattr(dat, key)(*args, "segment"), cmap="viridis")
            ax.set_title(key)
        plt.savefig(
            figdir / f"3d_dist_{name}_{METHOD}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
