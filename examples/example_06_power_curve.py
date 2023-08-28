from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import root_scalar
from tqdm import tqdm

from MITBEM.BEM import BEMSolver
from MITBEM.ReferenceTurbines import IEA15MW
from example_04_optimal_setpoint import calc_optimal_setpoint

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

ROTOR = IEA15MW()
bem = BEMSolver(ROTOR)


def setpoint(rotor, U, pitch_target, tsr_target):
    tsr_max = rotor.rotorspeed_max * rotor.R / U
    tsr = min(tsr_target, tsr_max)

    sol = bem.solve(pitch_target, tsr, 0)
    power = sol.power(U)

    if power > rotor.P_rated:
        # find pitch angle which produces rated power

        def func(pitch):
            power = bem.solve(pitch, tsr).power(U)
            return power - rotor.P_rated

        sol = root_scalar(
            func,
            x0=pitch_target,
            x1=np.deg2rad(30),
            bracket=(pitch_target, np.deg2rad(50)),
        )
        pitch = sol.root
        sol = bem.solve(pitch, tsr)

    out = {
        "U": U,
        "pitch\n[deg]": np.rad2deg(sol.pitch),
        "Torque\n[kNm]": sol.torque(U) / 10e3,
        "Rotor Speed\n[rad/s]": sol.tsr * U / rotor.R,
        "Power\n[MW]": sol.power(U) / 10e6,
        "Thrust\n[kN]": sol.thrust(U) / 10e3,
        "TSR": sol.tsr,
        "$C_p$": sol.Cp(agg="rotor"),
        "$C_T$": sol.Ct(agg="rotor"),
    }
    return out


if __name__ == "__main__":
    # Determine optimal set point for below-rated operation
    pitch_opt, tsr_opt = calc_optimal_setpoint(ROTOR)

    # Perform wind speed sweep.
    Us = np.linspace(3, 25)
    out = [setpoint(ROTOR, U, pitch_opt, tsr_opt) for U in tqdm(Us)]

    df = pl.from_dicts(out)

    # Plotting
    fig, axes = plt.subplots(len(df.columns) - 1, 1, sharex=True, figsize=[3, 8])

    axes[-1].set_xlabel("U [m/s]")
    axes[0].set_xlim(Us.min(), Us.max())
    for ax, key in zip(axes, [col for col in df.columns if col != "U"]):
        ax.plot(Us, df[key])
        ax.set_ylabel(key)
        ax.grid()

    plt.savefig(FIGDIR / f"example_06_power_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
