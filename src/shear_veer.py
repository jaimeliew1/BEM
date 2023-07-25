from pathlib import Path
from itertools import product
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import tqdm

from mit_bem.Turbine import IEA15MW
from mit_bem.BEM import BEM
from mit_bem.Windfield import ShearVeer
from mit_bem.Utilities import fixedpointiteration

figdir = Path("fig")
figdir.mkdir(parents=True, exist_ok=True)


rotor = IEA15MW()
rho = 1.293
Cp_target = 1.5  # This is a guess
Uinf = 10
YAW = np.deg2rad(0)


def calc_K(R, Cp, tsr, rho=1.293):
    K = 0.5 * rho * R**5 * Cp / tsr**3
    return K


K = calc_K(rotor.R, Cp_target, rotor.tsr_target)


def calc(x):
    shear_exp, dveerdz = x
    windfield = ShearVeer(rotor.hub_height, shear_exp, dveerdz, rotor.hub_height)
    bem = BEM(rotor, windfield)

    def torque_control_residual(tsr, K, Uinf=Uinf):
        bem.solve(0, tsr, YAW)
        torque = bem.torque(Uinf)
        new_rotor_speed = np.sqrt(torque / K)
        return new_rotor_speed * rotor.R / Uinf - tsr

    _, tsr = fixedpointiteration(torque_control_residual, 7, [K], relax=0.5)
    bem.solve(0, tsr, YAW)

    return dict(
        exp=shear_exp,
        veer=dveerdz,
        Cp=bem.Cp(),
        Ct=bem.Ct(),
        tsr=bem.tsr,
        a=bem.a(),
    )


def for_each(func, params, parallel=True):
    N = len(params)
    out = []
    if parallel:
        with Pool() as pool:
            for x in tqdm(
                pool.imap(
                    func,
                    params,
                ),
                total=N,
            ):
                out.append(x)
        return out
    else:
        for param in tqdm(params):
            out.append(func(param))
        return out


if __name__ == "__main__":
    exps = np.round(np.arange(-0.3, 0.81, 0.1), 2)
    veers = np.round(np.arange(-0.1, 0.41, 0.1), 2)
    params = list(product(exps, veers))

    outs = for_each(calc, params, parallel=True)

    df = pl.from_dicts(outs)

    Cp_ref = df.filter(pl.col("exp") == 0).filter(pl.col("veer") == 0)["Cp"][0]
    df = df.with_columns(pl.col("Cp") / Cp_ref)

    df_Cp = df.pivot(
        index="veer", columns="exp", values="Cp", aggregate_function=None
    ).sort("veer", descending=True)

    print(f"{Cp_ref=:2.4f}")
    print("Cp", df_Cp)

    plt.figure(figsize=1.2 * np.array([7, 4]))

    sns.heatmap(
        df_Cp.select(pl.exclude("veer")).to_numpy(),
        xticklabels=df_Cp.columns[1:],
        yticklabels=df_Cp["veer"].to_numpy(),
        ax=plt.gca(),
        annot=True,
        fmt=".2f",
        linewidth=0.5,
        cmap="vlag_r",
        center=1,
        cbar_kws=dict(label="$C_p$"),
    )

    plt.xlabel("Shear exponent")
    plt.ylabel("Veer [deg/m]")
    plt.savefig(figdir / "shear_veer.png", dpi=300, bbox_inches="tight")
    plt.close()
