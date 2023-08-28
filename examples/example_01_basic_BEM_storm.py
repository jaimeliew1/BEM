import numpy as np

from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW
from MITBEM.Windfield import ShearVeer

if __name__ == "__main__":
    bem = BEM(IEA15MW())
    windfield = ShearVeer(bem.rotor.hub_height / bem.rotor.R, 0.8, bem.rotor.R * 0.4)

    converged = bem.solve(pitch=np.deg2rad(-2.34), tsr=8, yaw=0, windfield=windfield)

    print(f"Converged: {converged}")

    print(f"{bem.Cp()=:2.4f}")
    print(f"{bem.tsr=:2.4f}")
    print(f"{bem.a()=:2.4f}")
