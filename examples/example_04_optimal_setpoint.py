from pathlib import Path

import numpy as np
from scipy.optimize import minimize


from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

ROTOR = IEA15MW()


def calc_optimal_setpoint(rotor, yaw=0):
    def objective(x):
        pitch, tsr = x
        bem = BEM(rotor)
        bem.solve(pitch=pitch, tsr=tsr, yaw=yaw)

        return -bem.Cp()

    pitch_opt, tsr_opt = minimize(objective, (0, 5)).x
    return pitch_opt, tsr_opt


if __name__ == "__main__":
    pitch_opt, tsr_opt = calc_optimal_setpoint(ROTOR)
    bem = BEM(ROTOR)
    bem.solve(pitch=pitch_opt, tsr=tsr_opt, yaw=0)

    print(f"Maximum power coefficient:  {bem.Cp():2.4f}")
    print(f"Optimal pitch: {np.rad2deg(pitch_opt):2.4f} deg")
    print(f"Optimal TSR: {tsr_opt:2.4f}")
