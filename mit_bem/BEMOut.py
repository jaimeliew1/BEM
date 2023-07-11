import numpy as np
from dataclasses import dataclass


def aggregate(mu, theta_grid, X, agg):
    if agg == None:
        return X

    # Integrate over azimuth
    X_azim = 1 / (2 * np.pi) * np.trapz(X, theta_grid, axis=-1)
    if agg == "azim":
        return X_azim

    # Integrate over rotor
    X_rotor = 2 * np.trapz(X_azim * mu, mu)
    if agg == "rotor":
        return X_rotor

    raise ValueError


@dataclass
class BEMOut:
    status: str
    pitch: float = None
    tsr: float = None
    yaw: float = None
    mu: np.ndarray = None
    theta: np.ndarray = None
    mu_grid: np.ndarray = None
    theta_grid: np.ndarray = None
    _a: np.ndarray = None
    _a_ring: np.ndarray = None
    _aprime: np.ndarray = None
    _phi: np.ndarray = None
    _Vax: np.ndarray = None
    _Vtan: np.ndarray = None
    _W: np.ndarray = None
    _Cax: np.ndarray = None
    _Ctan: np.ndarray = None
    solidity: np.ndarray = None

    def Ct(self, agg=None):
        ddCt = self._W**2 * self.solidity * self._Cax
        return aggregate(self.mu, self.theta_grid, ddCt, agg)

    def Cp(self, agg=None):
        ddCp = self.tsr * self._W**2 * self.solidity * self._Ctan * self.mu_grid
        return aggregate(self.mu, self.theta_grid, ddCp, agg)

    def a(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._a_ring, agg)

    def aprime(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._aprime, agg)

    def phi(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._phi, agg)

    def Vax(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._Vax, agg)

    def Vtan(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._Vtan, agg)

    def W(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._W, agg)

    def Cax(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._Cax, agg)

    def Ctan(self, agg=None):
        return aggregate(self.mu, self.theta_grid, self._aprime, agg)

    def Fax(self, U_inf, R, agg=None, rho=1.293):
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**2 * self.Cax() * self.mu_grid * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**2 * self.Cax("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**2 * self.Cax("rotor") * np.pi * R**2

        raise ValueError

    def Ftan(self, U_inf, R, agg=None, rho=1.293):
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**2 * self.Ctan() * self.mu_grid * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**2 * self.Ctan("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**2 * self.Ctan("rotor") * np.pi * R**2

        raise ValueError

    def thrust(self, U_inf, R, agg=None, rho=1.293):
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**2 * self.Ct() * self.mu_grid * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**2 * self.Ct("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**2 * self.Ct("rotor") * np.pi * R**2

        raise ValueError

    def power(self, U_inf, R, agg=None, rho=1.293):
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        if agg == None:
            return 0.5 * rho * U_inf**3 * self.Cp() * self.mu_grid * R * dR * dtheta

        # Integrate over azimuth
        if agg == "azim":
            return 0.5 * rho * U_inf**3 * self.Cp("azim") * self.mu * R * dR

        # Integrate over rotor
        if agg == "rotor":
            return 0.5 * rho * U_inf**3 * self.Cp("rotor") * np.pi * R**2

        raise ValueError
