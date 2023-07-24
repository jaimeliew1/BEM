import numpy as np
from mit_bem.Turbine import IEA15MW
from mit_bem.BEM import BEM
from mit_bem.Windfield import ShearVeer
from mit_bem.Utilities import fixedpointiteration

rotor = IEA15MW()
rho = 1.293
Cp_target = 5  # This is a guess


def calc_K(R, Cp, tsr, rho=1.293):
    K = 0.5 * rho * R**5 * Cp / tsr**3
    return K


if __name__ == "__main__":
    K = calc_K(rotor.R, 5, rotor.tsr_target)

    windfield = ShearVeer(rotor.hub_height, 0.14, 0, rotor.hub_height)
    bem = BEM(rotor, windfield)

    def torque_control_residual(tsr, K, Uinf=10):
        bem.solve(0, tsr, 0)
        torque = bem.torque(Uinf)
        new_rotor_speed = np.sqrt(torque / K)
        return new_rotor_speed * rotor.R / Uinf - tsr

    _, tsr = fixedpointiteration(torque_control_residual, 7, [K], relax=0.5)

    print(f"{K=:2.4f}")
    bem.solve(0, tsr, 0)
    print(f"{tsr=:2.4f}")
    print(f"{bem.Cp()=:2.4f}")
