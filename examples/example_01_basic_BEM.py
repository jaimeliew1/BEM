from MITBEM.BEM import BEM
from MITBEM.ReferenceTurbines import IEA15MW


if __name__ == "__main__":
    bem = BEM(IEA15MW())

    converged = bem.solve(pitch=0, tsr=8, yaw=0)

    print(f"Converged: {converged}")
    print(f"Power coefficient: {bem.Cp():2.4f}")
    print(f"Rotor-averaged axial induction: {bem.a():2.4f}")
