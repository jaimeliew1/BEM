import numpy as np
from .Utilities import fixedpointiteration, aggregate
from . import ThrustInduction, TipLoss


class BEM:
    def __init__(self, rotor, Cta_method="HAWC2", tiploss=None, Nr=20, Ntheta=21):
        self.rotor = rotor

        self.Nr, self.Ntheta = Nr, Ntheta
        self.mu = np.linspace(0.04, 0.98, Nr)
        self.theta = np.linspace(0.0, 2 * np.pi, Ntheta)

        self.theta_mesh, self.mu_mesh = np.meshgrid(self.theta, self.mu)

        self.pitch = None
        self.tsr = None
        self.yaw = None

        if Cta_method == "HAWC2":
            self.Cta_func = ThrustInduction.HAWC2
        elif Cta_method == "mike":
            self.Cta_func = ThrustInduction.mike
        else:
            raise ValueError

        if tiploss is None:
            self.tiploss_func = TipLoss.NoTiploss
        elif tiploss == "prandtl":
            self.tiploss_func = TipLoss.PrandtlTiploss
        elif tiploss == "tiproot":
            raise NotImplementedError
            # self.tiploss_func = TipLoss.PrandtlTipAndRootLossGenerator
        else:
            raise ValueError

        self.reset()

    def reset(self):
        self._a = 1 / 3 * np.ones((self.Nr, self.Ntheta))
        self._aprime = np.zeros((self.Nr, self.Ntheta))
        self._phi = np.zeros((self.Nr, self.Ntheta))
        self._aoa = np.zeros((self.Nr, self.Ntheta))
        self._Vax = np.zeros((self.Nr, self.Ntheta))
        self._Vtan = np.zeros((self.Nr, self.Ntheta))
        self._W = np.zeros((self.Nr, self.Ntheta))
        self._Cax = np.zeros((self.Nr, self.Ntheta))
        self._Ctan = np.zeros((self.Nr, self.Ntheta))
        self._tiploss = np.zeros((self.Nr, self.Ntheta))
        self.solidity = np.zeros((self.Nr, self.Ntheta))
        self.converged = False

    def solve(self, pitch, tsr, yaw, reset=True):
        if reset:
            self.reset()

        self.pitch = pitch
        self.tsr = tsr
        self.yaw = yaw

        a0 = 1 / 3 * np.ones((self.Nr, self.Ntheta))
        aprime0 = np.zeros((self.Nr, self.Ntheta))

        a_init = np.stack([a0, aprime0])

        try:
            a = fixedpointiteration(self.bem_iterate, x0=a_init, maxiter=100)
            self.converged = True
        except ValueError:
            self.converged = False
        except FloatingPointError:
            self.converged = False

        return self.converged

    def bem_iterate(self, a_input):
        self._a, self._aprime = a_input
        self._Vax = (
            (1 - self._a)
            * np.cos(self.yaw * np.cos(self.theta_mesh))
            * np.cos(self.yaw * np.sin(self.theta_mesh))
        )
        self._Vtan = (1 + self._aprime) * self.tsr * self.mu_mesh - (
            1 - self._a
        ) * np.cos(self.yaw * np.sin(self.theta_mesh)) * np.sin(
            self.yaw * np.cos(self.theta_mesh)
        )
        self._W = np.sqrt(self._Vax**2 + self._Vtan**2)

        # inflow angle
        self._phi = np.arctan2(self._Vax, self._Vtan)
        self._aoa = self._phi - self.rotor.bem.twist(self.mu_mesh) - self.pitch
        self._aoa = np.clip(self._aoa, -np.pi / 2, np.pi / 2)

        # Lift and drag coefficients
        Cl, Cd = self.rotor.bem.clcd(self.mu_mesh, self._aoa)

        # axial and tangential force coefficients
        self._Cax = Cl * np.cos(self._phi) + Cd * np.sin(self._phi)
        self._Ctan = Cl * np.sin(self._phi) - Cd * np.cos(self._phi)

        self.solidity = self.rotor.bem.solidity(self.mu_mesh)

        # Tip-loss correction
        self._tiploss = self.tiploss_func(self.mu_mesh, self._phi)
        Ct, a_new = self.Cta_func(self)

        # aprime_new = np.zeros_like(aprime)
        aprime_new = 1 / (
            4 * np.sin(self._phi) * np.cos(self._phi) / (self.solidity * self._Ctan) - 1
        )

        residual = np.stack([a_new - self._a, aprime_new - self._aprime])

        return residual

    def Ct(self, agg=None):
        ddCt = self._W**2 * self.solidity * self._Cax
        return aggregate(self.mu, self.theta_mesh, ddCt, agg)

    def Ctprime(self, agg=None):
        Ct = self.Ct(agg=agg)
        a = self.a(agg=agg)
        Ctprime = Ct / ((1 - a) ** 2 * np.cos(self.yaw) ** 2)
        return Ctprime

    def Cp(self, agg=None):
        ddCp = self.tsr * self._W**2 * self.solidity * self._Ctan * self.mu_mesh
        return aggregate(self.mu, self.theta_mesh, ddCp, agg)

    def a(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._a, agg)

    def aprime(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._aprime, agg)

    def phi(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._phi, agg)

    def aoa(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._aoa, agg)

    def Vax(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._Vax, agg)

    def Vtan(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._Vtan, agg)

    def W(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._W, agg)

    def Cax(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._Cax, agg)

    def tiploss(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._tiploss, agg)

    def Ctan(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._Ctan, agg)

    def Fax(self, U_inf, agg=None, rho=1.293):
        R = self.rotor.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**2 * self.Cax() * self.mu_mesh * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**2 * self.Cax("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**2 * self.Cax("rotor") * np.pi * R**2

        raise ValueError

    def Ftan(self, U_inf, agg=None, rho=1.293):
        R = self.rotor.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**2 * self.Ctan() * self.mu_mesh * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**2 * self.Ctan("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**2 * self.Ctan("rotor") * np.pi * R**2

        raise ValueError

    def thrust(self, U_inf, agg=None, rho=1.293):
        R = self.rotor.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**2 * self.Ct() * self.mu_mesh * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**2 * self.Ct("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**2 * self.Ct("rotor") * np.pi * R**2

        raise ValueError

    def power(self, U_inf, agg=None, rho=1.293):
        R = self.rotor.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**3 * self.Cp() * self.mu_mesh * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**3 * self.Cp("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**3 * self.Cp("rotor") * np.pi * R**2

        raise ValueError
