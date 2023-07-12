from pathlib import Path
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

from mit_bem.Turbine import IEA15MW

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

# optimal set point for IEA15MW
tsr_opt = 9.4604
pitch_opt = np.deg2rad(-2.7197)

METHOD = "standard"
METHOD = "mike"


def setpoint(rotor, U, tsr_target=tsr_opt, pitch_target=pitch_opt):
    tsr_max = rotor.rotorspeed_max * rotor.R / U
    tsr = min(tsr_target, tsr_max)

    bem = rotor.solve(pitch_target, tsr, 0, method=METHOD)
    power = bem.power(U, rotor.R, agg="rotor")

    if power > rotor.P_rated:
        # find pitch angle which produces rated power
        def func(pitch):
            bem = rotor.solve(pitch, tsr, 0)
            power = bem.power(U, rotor.R, agg="rotor")
            return power - rotor.P_rated

        sol = root_scalar(
            func,
            x0=pitch_target,
            x1=np.deg2rad(30),
            bracket=(pitch_target, np.deg2rad(50)),
        )
        pitch = sol.root
        bem = rotor.solve(pitch, tsr, 0)

    out = {
        "U": U,
        "a": bem.a(agg="rotor"),
        "power": bem.power(U, rotor.R, agg="rotor"),
        "thrust": bem.thrust(U, rotor.R, agg="rotor"),
        "Cp": bem.Cp(agg="rotor"),
        "Ct": bem.Ct(agg="rotor"),
        "Ctprime": bem.Ctprime(agg="rotor"),
        "tsr": bem.tsr,
        "rotorspeed": bem.tsr * U / rotor.R,
        "pitch": np.rad2deg(bem.pitch),
    }
    return out


def setpoint_trajectory(rotor):
    Us = np.linspace(3, 25)
    out = []
    for U in tqdm(Us):
        out.append(setpoint(rotor, U))

    df = pl.from_dicts(out)
    pitch = df["pitch"].to_numpy()
    tsr = df["tsr"].to_numpy()
    return pitch, tsr


if __name__ == "__main__":
    rotor = IEA15MW()
    Us = np.linspace(3, 25)

    out = []
    for U in tqdm(Us):
        out.append(setpoint(rotor, U))

    df = pl.from_dicts(out)

    fig, axes = plt.subplots(
        len(df.columns), 1, sharex=True, figsize=1.5 * np.array([2, 5])
    )
    axes[-1].set_xlabel("U [m/s]")
    axes[0].set_xlim(Us.min(), Us.max())
    for ax, key in zip(axes, df.columns):
        ax.plot(Us, df[key])
        ax.set_ylabel(key)
        ax.grid()

    plt.savefig(figdir / f"powercurve_noyaw_{METHOD}.png", dpi=300, bbox_inches="tight")
    plt.close()
