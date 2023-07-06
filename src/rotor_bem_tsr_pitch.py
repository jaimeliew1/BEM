from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine2 import IEA15MW
from tqdm import tqdm

np.seterr(all="raise")
figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

tsr_min, tsr_max = 3, 15
pitch_min, pitch_max = np.deg2rad(0), np.deg2rad(10)
if __name__ == "__main__":
    rotor = IEA15MW()

    tsrs = np.arange(tsr_min, tsr_max, 0.5)
    pitches = np.linspace(pitch_min, pitch_max, 20)

    tsr_mesh, pitch_mesh = np.meshgrid(tsrs, pitches)
    params = list(zip(tsr_mesh.ravel(), pitch_mesh.ravel()))
    Cts = []
    Cps = []
    for tsr, pitch in tqdm(params):
        Cts.append(rotor.Ct(pitch, tsr, 0))
        Cps.append(rotor.Cp(pitch, tsr, 0))

    Cts = np.reshape(Cts, tsr_mesh.shape)
    Cps = np.reshape(Cps, tsr_mesh.shape)

    fig, axes = plt.subplots(1, 2, sharey=True)

    axes[0].contourf(np.rad2deg(pitch_mesh), tsr_mesh, Cps)
    axes[1].contourf(np.rad2deg(pitch_mesh), tsr_mesh, Cts)

    axes[0].set_title("$C_p$")
    axes[1].set_title("$C_T$")

    axes[0].set_ylabel("$\lambda$ [-]")
    [ax.set_xlabel(r"$\theta_p$ [deg]") for ax in axes]

    CS = axes[0].contour(np.rad2deg(pitch_mesh), tsr_mesh, Cps, colors="k")
    axes[0].clabel(CS, inline=True, fontsize=10)

    CS = axes[1].contour(np.rad2deg(pitch_mesh), tsr_mesh, Cts, colors="k")
    axes[1].clabel(CS, inline=True, fontsize=10)
    plt.savefig(figdir / "tsr_pitch.png", dpi=300, bbox_inches="tight")
    plt.close()
