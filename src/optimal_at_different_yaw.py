from pathlib import Path
import itertools
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mit_bem.Turbine import IEA15MW, IEA10MW
from mit_bem.BEM import BEM
from tqdm import tqdm
import polars as pl

np.seterr(all="ignore")
figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

PARALLEL = True

rotors = {
    "IEA15MW": IEA15MW(),
    "IEA10MW": IEA10MW(),
}

# rotor_name: (TSR, pitch [rad])
target_setpoints = {
    "IEA15MW": (8.678696, np.deg2rad(-3.484844)),
}


# TSR and pitch bounds (plus or minus from optimal) for various control methods
strategies = {
    "tsr": (3, np.deg2rad(0.001)),
    "pitch": (0.01, np.deg2rad(5)),
    "both": (3, np.deg2rad(5)),
}

Variables = {
    "strategy": ["both", "pitch", "tsr"],
    "rotor": ["IEA15MW"],
    "method": ["HAWC2", "mike", "mike_corrected"],
    "yaw": [0, 10, 20, 30, 40, 50],
}

tsr_bounds = 3, 15
pitch_bounds = np.deg2rad(-10), np.deg2rad(5)


def find_optimal_both(rotor_name, Cta_method, yaw):
    bem = BEM(rotors[rotor_name], Cta_method=Cta_method)
    strategy = "both"

    tsr_target, pitch_target = target_setpoints[rotor_name]
    tsr_bounds = (
        tsr_target - strategies[strategy][0],
        tsr_target + strategies[strategy][0],
    )
    pitch_bounds = (
        pitch_target - strategies[strategy][1],
        pitch_target + strategies[strategy][1],
    )

    def func(x0):
        tsr, pitch = x0
        bem.solve(pitch, tsr, np.deg2rad(yaw))
        if not bem.converged:
            cost = 1000
        else:
            cost = -bem.Cp()
        return cost

    res = minimize(func, (tsr_target, pitch_target), bounds=[tsr_bounds, pitch_bounds])
    tsr_opt, pitch_opt = res.x

    bem.solve(pitch_opt, tsr_opt, np.deg2rad(yaw))

    return dict(
        rotor=rotor_name,
        Cta_method=Cta_method,
        strategy=strategy,
        yaw=np.rad2deg(bem.yaw),
        tsr=bem.tsr,
        pitch=np.rad2deg(bem.pitch),
        Cp=bem.Cp(),
        a=bem.a(),
        Ct=bem.Ct(),
        Ctprime=bem.Ctprime(),
    )


def find_optimal_pitch(rotor_name, Cta_method, yaw):
    bem = BEM(rotors[rotor_name], Cta_method=Cta_method)
    strategy = "pitch"

    tsr_target, pitch_target = target_setpoints[rotor_name]
    pitch_bounds = (
        pitch_target - np.deg2rad(5),
        pitch_target + np.deg2rad(5),
    )

    def func(x0):
        pitch = x0
        bem.solve(pitch, tsr_target, np.deg2rad(yaw))
        if not bem.converged:
            cost = 1000
        else:
            cost = -bem.Cp()
        return cost

    res = minimize(func, pitch_target, bounds=[pitch_bounds])
    pitch_opt = res.x

    bem.solve(pitch_opt, tsr_target, np.deg2rad(yaw))

    return dict(
        rotor=rotor_name,
        Cta_method=Cta_method,
        strategy=strategy,
        yaw=np.rad2deg(bem.yaw),
        tsr=bem.tsr,
        pitch=np.rad2deg(bem.pitch),
        Cp=bem.Cp(),
        a=bem.a(),
        Ct=bem.Ct(),
        Ctprime=bem.Ctprime(),
    )


def find_optimal_tsr(rotor_name, Cta_method, yaw):
    bem = BEM(rotors[rotor_name], Cta_method=Cta_method)
    strategy = "tsr"

    tsr_target, pitch_target = target_setpoints[rotor_name]
    tsr_bounds = (
        tsr_target - 4,
        tsr_target + 4,
    )

    def func(x0):
        tsr = x0
        bem.solve(pitch_target, tsr, np.deg2rad(yaw))
        if not bem.converged:
            cost = 1000
        else:
            cost = -bem.Cp()
        return cost

    res = minimize(func, tsr_target, bounds=[tsr_bounds])
    tsr_opt = res.x

    bem.solve(pitch_target, tsr_opt, np.deg2rad(yaw))

    return dict(
        rotor=rotor_name,
        Cta_method=Cta_method,
        strategy=strategy,
        yaw=np.rad2deg(bem.yaw),
        tsr=bem.tsr,
        pitch=np.rad2deg(bem.pitch),
        Cp=bem.Cp(),
        a=bem.a(),
        Ct=bem.Ct(),
        Ctprime=bem.Ctprime(),
    )


def find_optimal_wrapper(x):
    strategy, *args = x

    return OptimalFinder[strategy](*args)


OptimalFinder = {
    "tsr": find_optimal_tsr,
    "pitch": find_optimal_pitch,
    "both": find_optimal_both,
}


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
    params = pl.DataFrame(
        itertools.product(*list(Variables.values())), schema=Variables.keys()
    )
    print("input parameters:", params)

    outs = for_each(find_optimal_wrapper, list(params.iter_rows()), parallel=PARALLEL)

    df = pl.from_dicts(outs)
    print(df)

    fig, axes = plt.subplots(5, 1, sharex=True)

    for ax, value_key in zip(axes, ["Cp", "tsr", "pitch", "Ct", "Ctprime"]):
        df_piv = df.filter(pl.col("Cta_method") == "mike_corrected").pivot(
            index="yaw", columns="strategy", values=value_key
        )

        strategies = df["strategy"].unique()
        for strategy in strategies:
            ax.plot(df_piv["yaw"], df_piv[strategy], label=strategy)

        ax.set_ylabel(value_key)

    axes[-1].set_xlabel("yaw [deg]")

    axes[0].legend()

    plt.savefig(figdir / "optimal_at_different_yaw.png", dpi=300, bbox_inches="tight")
