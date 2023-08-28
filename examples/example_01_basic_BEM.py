from MITBEM.BEM import BEMSolver
from MITBEM.ReferenceTurbines import IEA15MW


if __name__ == "__main__":
    bem = BEMSolver(IEA15MW())

    sol = bem.solve(pitch=0, tsr=8, yaw=0)

    print(f"Converged: {sol.converged}")
    print(f"Power coefficient: {sol.Cp():2.4f}")
    print(f"Rotor-averaged axial induction: {sol.a():2.4f}")
