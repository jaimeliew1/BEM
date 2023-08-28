import numpy as np
from .Utilities import aggregate


class BEMSolution:
    def __init__(self, mu, theta, mu_mesh, theta_mesh, pitch, tsr, yaw, U, wdir, R):
        self.mu, self.theta = mu, theta
        self.mu_mesh, self.theta_mesh = mu_mesh, theta_mesh
        self.Nr, self.Ntheta = len(mu), len(theta)

        self.pitch, self.tsr, self.yaw = pitch, tsr, yaw
        self.U, self.wdir = U, wdir

        self.R = R

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

    def Cq(self, agg=None):
        ddCq = self._W**2 * self.solidity * self._Ctan * self.mu_mesh
        return aggregate(self.mu, self.theta_mesh, ddCq, agg)

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

    def Cl(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._Cl, agg)

    def Cd(self, agg=None):
        return aggregate(self.mu, self.theta_mesh, self._Cd, agg)

    def Fax(self, U_inf, agg=None, rho=1.293):
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        if agg is None or (agg == "rotor"):
            A = np.pi * R**2
        elif agg == "azim":
            A = self.mu * R * dR
        elif agg == "segment":
            A = self.mu_mesh * R * dR * dtheta
        else:
            ValueError

        return 0.5 * rho * U_inf**2 * self.Cax(agg) * A

    def Ftan(self, U_inf, agg=None, rho=1.293):
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        if agg is None or (agg == "rotor"):
            A = np.pi * R**2
        elif agg == "azim":
            A = self.mu * R * dR
        elif agg == "segment":
            A = self.mu_mesh * R * dR * dtheta
        else:
            ValueError

        return 0.5 * rho * U_inf**2 * self.Ctan(agg) * A

    def thrust(self, U_inf, agg=None, rho=1.293):
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        if agg is None or (agg == "rotor"):
            A = np.pi * R**2
        elif agg == "azim":
            A = self.mu * R * dR
        elif agg == "segment":
            A = self.mu_mesh * R * dR * dtheta
        else:
            ValueError

        return 0.5 * rho * U_inf**2 * self.Ct(agg) * A

    def power(self, U_inf, agg=None, rho=1.293):
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        if agg is None or (agg == "rotor"):
            A = np.pi * R**2
        elif agg == "azim":
            A = self.mu * R * dR
        elif agg == "segment":
            A = self.mu_mesh * R * dR * dtheta
        else:
            ValueError

        return 0.5 * rho * U_inf**3 * self.Cp(agg) * A

    def torque(self, U_inf, rho=1.293):
        R = self.R
        rotor_speed = self.tsr * U_inf / R

        return self.power(U_inf, rho=rho) / rotor_speed

    def u4(self):
        return 1 - 0.5 * self.Ct() / (1 - self.a())

    def v4(self):
        # POSSIBLE SIGN ERROR
        return -0.25 * self.Ct() * np.sin(self.yaw)

    def REWS(self):
        return aggregate(self.mu, self.theta, self.U)
