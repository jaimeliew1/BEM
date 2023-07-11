import numpy as np
from .Utilities import fixedpointiteration
from .BEMOut import BEMOut


class GenericRotor:
    def __init__(self, twist, solidity, clcd, tiploss, Ct2a, N_r=30, N_theta=21):
        self.twist = twist
        self.solidity = solidity
        self.clcd = clcd
        self.tiploss = tiploss
        self.Ct2a = Ct2a

        self.N_r = N_r
        self.N_theta = N_theta

        self.mus = np.linspace(0.04, 0.98, N_r)
        self.thetas = np.linspace(0.0, 2 * np.pi, N_theta)

        self.theta_mesh, self.mu_mesh = np.meshgrid(self.thetas, self.mus)

    def bem(self, a_init, pitch, tsr, yaw, return_data=False):
        a, aprime = a_init
        vx = (
            (1 - a)
            * np.cos(yaw * np.cos(self.theta_mesh))
            * np.cos(yaw * np.sin(self.theta_mesh))
        )
        vt = (1 + aprime) * tsr * self.mu_mesh - (1 - a) * np.cos(
            yaw * np.sin(self.theta_mesh)
        ) * np.sin(yaw * np.cos(self.theta_mesh))

        # inflow angle
        phi = np.arctan2(vx, vt)
        aoa = phi - self.twist(self.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = self.clcd(self.mu_mesh, aoa)

        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ctan = Cl * np.sin(phi) - Cd * np.cos(phi)

        sigma = self.solidity(self.mu_mesh)
        dCt = np.minimum((1 - a) ** 2 * sigma * Cn / np.sin(phi) ** 2, 4)

        aprime_new = np.zeros_like(aprime)
        # aprime_new = 1 / (4 * np.sin(phi) * np.cos(phi) / (sigma * Ctan) - 1)
        a_new = self.Ct2a(dCt / self.tiploss(self.mu_mesh, phi))
        a_ring = self.Ct2a(dCt)

        residual = np.stack([a_new - a, aprime_new - aprime])
        if return_data:
            W = np.sqrt(vx**2 + vt**2)
            return (
                pitch,
                tsr,
                yaw,
                self.mus,
                self.thetas,
                self.mu_mesh,
                self.theta_mesh,
                a_new,
                a_ring,
                aprime_new,
                phi,
                vx,
                vt,
                W,
                Cn,
                Ctan,
                sigma,
            )
        else:
            return residual

    def solve(self, pitch, tsr, yaw):
        a0 = 1 / 3 * np.ones((self.N_r, self.N_theta))
        aprime0 = np.zeros((self.N_r, self.N_theta))

        a_init = np.stack([a0, aprime0])

        try:
            a = fixedpointiteration(
                self.bem, x0=a_init, args=(pitch, tsr, yaw), relax=0.4, maxiter=100
            )
            status = "converged"
            data = self.bem(a, pitch, tsr, yaw, return_data=True)
            out = BEMOut(status, *data)
        except ValueError:
            status = "unconverged"
            out = BEMOut(status)

        return out
