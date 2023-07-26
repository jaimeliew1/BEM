import itertools
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from mit_bem.Turbine import IEA15MW
from mit_bem.Windfarm import BEMWindfarm
from Optimizations import JointOptimization

np.seterr(all="ignore")
FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = Path("cache")
CACHEDIR.mkdir(parents=True, exist_ok=True)

wind_dir_sweep_fn = CACHEDIR / "wind_dir_sweep.csv"

coords_base = np.array([[0, 7], [0, 0]])

rotor = IEA15MW()


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


def get_layout(wdir):
    wdir = np.deg2rad(wdir)
    rotmat = np.array([[np.cos(wdir), -np.sin(wdir)], [np.sin(wdir), np.cos(wdir)]])

    p = rotmat @ coords_base

    xs, ys = p[0, :], p[1, :]

    return xs, ys


def run(args):
    wdir = args
    xs, ys = get_layout(wdir)

    Cp, setpoints = JointOptimization(
        xs, ys, wf_kwargs=dict(rotor=rotor)
    ).find_optimal()

    pitches, tsrs, yaws = np.split(setpoints, 3)
    return dict(
        wdir=wdir,
        Cp=Cp,
        x1=xs[0],
        x2=xs[1],
        y1=ys[0],
        y2=ys[1],
        pitch1=np.rad2deg(pitches[0]),
        pitch2=np.rad2deg(pitches[1]),
        tsr1=tsrs[0],
        tsr2=tsrs[1],
        yaw1=np.rad2deg(yaws[0]),
        yaw2=np.rad2deg(yaws[1]),
    )


if __name__ == "__main__":
    wdirs = np.linspace(-10, 10, 51)

    params = list(wdirs)
    out = for_each(run, params)

    df = pl.from_dicts(out)
    df.write_csv(wind_dir_sweep_fn)

    print(df)
