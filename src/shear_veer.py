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

from find_optimal_setpoint import find_optimal

figdir = Path("fig")
figdir.mkdir(parents=True, exist_ok=True)


rotor = IEA15MW()
rho = 1.293


Uinf = 10
YAW = np.deg2rad(0)
Cta_method = "mike_corrected"
tiploss = "tiproot"

to_plot = {
    "Cp": ("$C_p/C_{p,ref}$", True),
    "Ct": ("$C_T/C_{T,ref}$", True),
    "Ctprime": ("$C_T'/C_{T,ref}'$", True),
    "a": ("$a$", False),
    "tsr": ("$\lambda$", False),
    "tsr": ("$\lambda/\lambda_{ref}$", True),
}

tsr_opt, pitch_opt, Cp_opt = find_optimal(rotor, Cta_method=Cta_method, tiploss=tiploss)
print(f"{tsr_opt=:2.3f}, {np.rad2deg(pitch_opt)=:2.3f}, {Cp_opt=:2.3f}")


def calc_K(R, Cp, tsr, rho=1.293):
    K = 0.5 * rho * np.pi * R**5 * Cp / tsr**3
    return K


K = calc_K(rotor.R, Cp_opt, tsr_opt)


def calc(x):
    shear_exp, dveerdz = x
    windfield = ShearVeer(rotor.hub_height, shear_exp, dveerdz, rotor.hub_height)
    bem = BEM(rotor, windfield, Cta_method=Cta_method, tiploss=tiploss)

    def torque_control_residual(tsr, K, Uinf=Uinf):
        bem.solve(pitch_opt, tsr, YAW)
        # torque = bem.torque(Uinf)
        # new_rotor_speed = np.sqrt(torque / K)
        # return new_rotor_speed * rotor.R / Uinf - tsr
        return np.sqrt(bem.Cq() / Cp_opt * tsr_opt**3) - tsr

    _, tsr = fixedpointiteration(torque_control_residual, 7, [K], relax=0.5)
    bem.solve(pitch_opt, tsr, YAW)

    return dict(
        exp=shear_exp,
        veer=dveerdz,
        Cp=bem.Cp(),
        Ct=bem.Ct(),
        Ctprime=bem.Ctprime(),
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


def plot_heatmap(df, key, save=None, label="", normalize=True):
    if normalize:
        val_ref = df.filter(pl.col("exp") == 0).filter(pl.col("veer") == 0)[key][0]
        df = df.with_columns(pl.col(key) / val_ref)
        print(key, val_ref)

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

    outs = for_each(calc, params, parallel=True)

    df = pl.from_dicts(outs)

    for key, (label, normalize) in to_plot.items():
        fn = figdir / f"shear_veer_{key}.png"
        plot_heatmap(df, key, save=fn, label=label, normalize=normalize)
