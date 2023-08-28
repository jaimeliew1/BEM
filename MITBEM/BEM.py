import numpy as np
from .Utilities import adaptivefixedpointiteration
from . import ThrustInduction, TipLoss
from .BEMSolution import BEMSolution


class _BEMSolverBase:
    def __init__(
        self,
        rotor,
        Cta_method="mike_corrected",
        tiploss="PrandtlRootTip",
        Nr=20,
        Ntheta=21,
    ):
        self.rotor = rotor

        self.Nr, self.Ntheta = Nr, Ntheta

        self.mu, self.theta, self.mu_mesh, self.theta_mesh = self.calc_gridpoints(
            Nr, Ntheta
        )

        self.Cta_func = ThrustInduction.build_cta_model(Cta_method)
        self.tiploss_func = TipLoss.build_tiploss_model(tiploss, rotor)

        self._solidity = self.rotor.solidity(self.mu_mesh)

    def solve(self):
        ...

    @classmethod
    def calc_gridpoints(cls, Nr, Ntheta):
        mu = np.linspace(0.0, 1.0, Nr)
        theta = np.linspace(0.0, 2 * np.pi, Ntheta)

        theta_mesh, mu_mesh = np.meshgrid(theta, mu)

        return mu, theta, mu_mesh, theta_mesh

    def gridpoints_cart(self, yaw):
        """
        Returns the grid point locations in cartesian coordinates
        nondimensionialized by rotor radius. Origin is located at hub center.

        Note: effect of yaw angle on grid points is not yet implemented.
        """
        # Probable sign error here.
        X = np.zeros_like(self.mu_mesh)
        Y = self.mu_mesh * np.sin(self.theta_mesh)  # lateral
        Z = self.mu_mesh * np.cos(self.theta_mesh)  # vertical

        return X, Y, Z

    def _sample_windfield(self, windfield):
        yaw = 0  # To do: change grid points based on yaw angle.
        _X, _Y, _Z = self.gridpoints_cart(yaw)
        Y = _Y
        Z = self.rotor.hub_height / self.rotor.R + _Z

        U = windfield.wsp(Y, Z)
        wdir = windfield.wdir(Y, Z)

        return U, wdir


class BEMSolver(_BEMSolverBase):
    def solve(
        self, pitch: float, tsr: float, yaw: float = 0.0, windfield=None
    ) -> BEMSolution:
        if callable(windfield):
            self.U, self.wdir = self._sample_windfield(windfield)
        elif windfield:
            self.U, self.wdir = windfield
        else:
            self.U, self.wdir = np.ones_like(self.mu_mesh), np.zeros_like(self.mu_mesh)

        self.sol = BEMSolution(
            self.mu,
            self.theta,
            self.mu_mesh,
            self.theta_mesh,
            pitch,
            tsr,
            yaw,
            self.U,
            self.wdir,
            self.rotor.R,
        )
        self.sol.solidity = self._solidity

        a0 = 1 / 3 * np.ones((self.Nr, self.Ntheta))
        aprime0 = np.zeros((self.Nr, self.Ntheta))
        a_init = np.stack([a0, aprime0])

        try:
            converged, a = adaptivefixedpointiteration(
                self.bem_iterate, x0=a_init, maxiter=100
            )
        except FloatingPointError:
            converged = False
        except ValueError:
            converged = False
        self.sol.converged = converged

        return self.sol

    def bem_iterate(self, a_input):
        sol = self.sol

        sol._a, sol._aprime = a_input

        local_yaw = sol.wdir - sol.yaw
        sol._Vax = sol.U * (
            (1 - sol._a)
            * np.cos(local_yaw * np.cos(sol.theta_mesh))
            * np.cos(local_yaw * np.sin(sol.theta_mesh))
        )
        sol._Vtan = (1 + sol._aprime) * sol.tsr * sol.mu_mesh - sol.U * (
            1 - sol._a
        ) * np.cos(local_yaw * np.sin(sol.theta_mesh)) * np.sin(
            local_yaw * np.cos(sol.theta_mesh)
        )
        sol._W = np.sqrt(sol._Vax**2 + sol._Vtan**2)

        # inflow angle
        sol._phi = np.arctan2(sol._Vax, sol._Vtan)
        sol._aoa = sol._phi - self.rotor.twist(sol.mu_mesh) - sol.pitch
        sol._aoa = np.clip(sol._aoa, -np.pi / 2, np.pi / 2)

        # Lift and drag coefficients
        sol._Cl, sol._Cd = self.rotor.clcd(sol.mu_mesh, sol._aoa)

        # axial and tangential force coefficients
        sol._Cax = sol._Cl * np.cos(sol._phi) + sol._Cd * np.sin(sol._phi)
        sol._Ctan = sol._Cl * np.sin(sol._phi) - sol._Cd * np.cos(sol._phi)

        # Tip-loss correction
        sol._tiploss = self.tiploss_func(sol.mu_mesh, sol._phi)

        a_new = self.Cta_func(sol)

        aprime_new = np.zeros_like(sol._a)

        residual = np.stack([a_new - sol._a, aprime_new - sol._aprime])

        return residual
