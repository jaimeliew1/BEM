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
from MITWake.Rotor import yawthrust

figdir = Path("fig")
figdir.mkdir(parents=True, exist_ok=True)


rotor = IEA15MW()
rho = 1.293
Uinf = 10
YAW = np.deg2rad(0)
Cta_method = "mike_corrected"
tiploss = "tiproot"


def calc(x):
    pitch, tsr, yaw = x
    bem = BEM(rotor, Cta_method=Cta_method, tiploss=tiploss)
    converged = bem.solve(np.deg2rad(pitch), tsr, np.deg2rad(yaw))
    if converged:
        return dict(
            tsr=bem.tsr,
            pitch=np.rad2deg(bem.pitch),
            yaw=np.rad2deg(bem.yaw),
            Cp=bem.Cp(),
            Ct=bem.Ct(),
            Ctprime=bem.Ctprime(),
            a=bem.a(),
        )
    else:
        return None


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
    # Define parameter space
    pitches = np.linspace(-4, 4, 10)
    tsrs = np.linspace(5, 10, 10)
    yaws = np.linspace(0, 40, 10)
    params = list(product(pitches, tsrs, yaws))

    # Calculate BEM quantities, particularly Ct, Ctprime, u4, v4
    outs = for_each(calc, params, parallel=True)

    Ctprime = pl.col("Ctprime")
    Ct = pl.col("Ct")
    a = pl.col("a")
    yaw = pl.col("yaw")

    df_bem = pl.from_dicts(outs)
    df_bem = df_bem.with_columns(
        [
            (1 - 0.5 * Ct / (1 - a)).alias("u4"),
            (0.25 * Ct * np.sin(np.deg2rad(yaw))).alias("v4"),
        ]
    )

    print(df_bem)

    # Calculate disk quantities from Ctprime and yaw
    outs = []
    for Ctprime, yaw in df_bem.select(["Ctprime", "yaw"]).iter_rows():
        a, u4, v4 = yawthrust(Ctprime, np.deg2rad(yaw))
        outs.append(dict(a=a, v4=v4, u4=u4))

    df_disk = pl.from_dicts(outs)

    print(df_disk)

    for key in ["a", "u4", "v4"]:
        diff = (df_disk[key] - df_bem[key]).abs().mean()
        print(f"{key} diff: {diff:2.5f}")
