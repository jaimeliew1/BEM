from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mit_bem.Turbine import IEA15MW, IEA10MW, IEA3_4MW
from tqdm import tqdm

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


tsr_min, tsr_max = 3, 15
PITCH = np.deg2rad(0)

rotors = {
    "IEA15MW": IEA15MW(),
    "IEA10MW": IEA10MW(),
    # "IEA3.4MW": IEA3_4MW(),
}
if __name__ == "__main__":
    for name, rotor in rotors.items():
        tsrs = np.arange(tsr_min, tsr_max, 1)
        # tsrs = np.linspace(0.1, 2, 100)
        data = []
        Cts = []
        Cps = []
        a_rots = []
        for tsr in tqdm(tsrs):
            ind = rotor.induction(PITCH, tsr, 0)
            data.append(rotor.bem_residual(ind, PITCH, tsr, 0, return_data=True))
            Cts.append(rotor.Ct(PITCH, tsr, 0, agg="rotor"))
            Cps.append(rotor.Cp(PITCH, tsr, 0, agg="rotor"))
            a_rots.append(rotor.rotor_induction(PITCH, tsr, 0, agg="rotor"))
            # Rs, a_new, phi, W, Cn, Ctan, sigma

            # Ct = rotor.Ct(0, tsr, 0)
            # print("CT ", Ct)

        print(np.max(Cps))

        fig, axes = plt.subplots(6, 1, sharex=True, figsize=np.array([4, 10]))

        for i, (tsr, dat) in enumerate(zip(tsrs, data)):
            Rs, a, a_ring, phi, W, Cn, Ctan, dCt, chord = dat
            axes[0].plot(Rs, a, color=plt.cm.viridis(i / len(tsrs)))
            axes[0].plot(Rs, a_ring, "--", color=plt.cm.viridis(i / len(tsrs)))
            axes[0].set_ylabel("a")
            axes[1].plot(Rs, phi, color=plt.cm.viridis(i / len(tsrs)))
            axes[1].set_ylabel("phi")
            axes[2].plot(Rs, W, color=plt.cm.viridis(i / len(tsrs)))
            axes[2].set_ylabel("W")
            axes[3].plot(Rs, Cn, color=plt.cm.viridis(i / len(tsrs)))
            axes[3].set_ylabel("Cn")
            axes[4].plot(Rs, Ctan, color=plt.cm.viridis(i / len(tsrs)))
            axes[4].set_ylabel("Ctan")
            axes[5].plot(Rs, dCt, color=plt.cm.viridis(i / len(tsrs)))
            axes[5].set_ylabel("$\delta C_T$")

        plt.xlim(0, 1)
        # plt.plot(inductions.T)

        plt.savefig(figdir / "radius_vs.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].plot(tsrs, Cts)
        axes[0].set_ylabel("$C_T$")
        axes[1].plot(tsrs, Cps)
        axes[1].set_ylabel("$C_P$")
        axes[2].plot(tsrs, a_rots)
        axes[2].set_ylabel("$a_{rot}$")
        plt.savefig(figdir / f"tsr_vs_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()
