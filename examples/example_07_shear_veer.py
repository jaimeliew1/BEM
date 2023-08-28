from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import tqdm

from MITBEM.ReferenceTurbines import IEA15MW
from MITBEM.BEM import BEMSolver
from MITBEM.Windfield import ShearVeer
from MITBEM.Utilities import fixedpointiteration

from example_04_optimal_setpoint import calc_optimal_setpoint

figdir = Path("fig")
figdir.mkdir(parents=True, exist_ok=True)


rotor = IEA15MW()


YAW = np.deg2rad(0)


to_plot = {
    "Cp": ("$C_p/C_{p,ref}$", True),
    "Ct": ("$C_T/C_{T,ref}$", True),
    "Ctprime": ("$C_T'/C_{T,ref}'$", True),
    "a": ("$a$", False),
    "tsr": ("$\lambda$", False),
    "tsr": ("$\lambda/\lambda_{ref}$", True),
}

pitch_opt, tsr_opt = calc_optimal_setpoint(rotor)
Cp_opt = BEMSolver(rotor).solve(pitch_opt, tsr_opt).Cp()


def calc(x):
    shear_exp, dveerdz = x
    windfield = ShearVeer(rotor.hub_height / rotor.R, shear_exp, rotor.R * dveerdz)
    bem = BEMSolver(rotor)

    def torque_control_residual(tsr):
        sol = bem.solve(pitch_opt, tsr, YAW, windfield)

        if sol.converged:
            return np.cbrt(sol.Cp() / Cp_opt * tsr_opt**3) - tsr

    _, tsr = fixedpointiteration(torque_control_residual, 7, relax=0.5)
    sol = bem.solve(pitch_opt, tsr, YAW, windfield)

    return dict(
        exp=shear_exp,
        veer=dveerdz,
        Cp=sol.Cp(),
        Ct=sol.Ct(),
        Ctprime=sol.Ctprime(),
        tsr=sol.tsr,
        a=sol.a(),
    )


def plot_heatmap(df, key, save=None, label="", normalize=True):
    if normalize:
        val_ref = df.filter(pl.col("exp") == 0).filter(pl.col("veer") == 0)[key][0]
        df = df.with_columns(pl.col(key) / val_ref)

    df_piv = df.pivot(
        index="veer", columns="exp", values=key, aggregate_function=None
    ).sort("veer", descending=True)

    plt.figure(figsize=1.2 * np.array([7, 4]))

    sns.heatmap(
        df_piv.select(pl.exclude("veer")).to_numpy(),
        xticklabels=df_piv.columns[1:],
        yticklabels=df_piv["veer"].to_numpy(),
        ax=plt.gca(),
        annot=True,
        fmt=".2f",
        linewidth=0.5,
        cmap="vlag_r",
        center=1,
        cbar_kws=dict(label=label),
    )

    plt.xlabel("Shear exponent")
    plt.ylabel("Veer [deg/m]")
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    exps = np.round(np.arange(-0.3, 0.81, 0.1), 2)
    veers = np.round(np.arange(-0.1, 0.41, 0.1), 2)
    params = list(product(exps, veers))

    outs = [calc(args) for args in tqdm(params)]

    df = pl.from_dicts(outs)

    for key, (label, normalize) in to_plot.items():
        fn = figdir / f"example_07_shear_veer_{key}.png"
        plot_heatmap(df, key, save=fn, label=label, normalize=normalize)
