from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mit_bem.Turbine import IEA15MW, IEA10MW, IEA3_4MW
from tqdm import tqdm

np.seterr(all="raise")
figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


rotors = {
    "IEA15MW": IEA15MW(),
    "IEA10MW": IEA10MW(),
    # "IEA3.4MW": IEA3_4MW(),
}


tsr_bounds = 3, 15
pitch_bounds = np.deg2rad(-5), np.deg2rad(5)


def find_optimal(
    rotor, x0=[7, 0], tsr_bounds=(3, 9), pitch_bounds=(np.deg2rad(-3), np.deg2rad(5))
):
    def func(x0):
        tsr, pitch = x0
        bem = rotor.solve(pitch, tsr, 0)
        return -bem.Cp(agg="rotor")

    res = minimize(func, x0, bounds=[tsr_bounds, pitch_bounds])
    tsr_opt, pitch_opt = res.x
    Cp_opt = -res.fun

    return tsr_opt, pitch_opt, Cp_opt


if __name__ == "__main__":
    for name, rotor in rotors.items():
        tsr_opt, pitch_opt, Cp_opt = find_optimal(rotor)
        print(
            f"{name}: tsr={tsr_opt:2.4f}, pitch={np.rad2deg(pitch_opt):2.4f}, Cp_max = {Cp_opt:2.4f}"
        )
