# typing.Literal was introduced in Python3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import List, Optional, Tuple

import numpy as np
from MITWake import Superposition, Wake
from MITWake.Turbine import BasicTurbine
from MITWake.Windfarm import Windfarm

from .BEM import BEM
from .Windfield import Custom


class BEMTurbine(BasicTurbine):
    def __init__(
        self,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor,
        x=0.0,
        y=0.0,
        sigma=0.25,
        kw=0.07,
        Nr=20,
        Ntheta=21,
        **kwargs,
    ) -> None:
        """

        Args:
            Ct (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).
            x (float): Longitudinal turbine position. Defaults to 0.0.
            y (float): Lateral turbine position. Defaults to 0.0.
            sigma (float): Gaussian wake proportionality constant. Defaults to 0.25.
            kw (float): Wake spreading parameter. Defaults to 0.07.
            induction_eps (float): Convergence tolerance. Defaults to 0.000001.
        """
        self.x, self.y = x, y
        self.pitch, self.tsr, self.yaw = pitch, tsr, yaw

        self.sigma, self.kw = sigma, kw

        self.rotor = rotor
        self.Nr, self.Ntheta = Nr, Ntheta
        self.BEM_kwargs = kwargs

    def gridpoints(self):
        mu, theta, mu_mesh, theta_mesh = BEM.gridpoints(self.Nr, self.Ntheta)

        return mu_mesh, theta_mesh

    def post_init(self, windfield):
        self.bem = BEM(
            self.rotor, windfield, Nr=self.Nr, Ntheta=self.Ntheta, **self.BEM_kwargs
        )
        self.bem.solve(self.pitch, self.tsr, self.yaw)

        self.wake = Wake.Gaussian(self.bem.u4(), self.bem.v4(), self.sigma, self.kw)


class BEMWindfarm(Windfarm):
    def __init__(
        self,
        xs: List[float],
        ys: List[float],
        pitches: List[float],
        tsrs: List[float],
        yaws: List[float],
        rotor,
        summation: Literal[
            "linear", "quadratic", "linearniayifar", "quadraticniayifar", "zong"
        ] = "linear",
        sigmas: Optional[List[float]] = 0.25,
        kwts: Optional[List[float]] = 0.07,
    ) -> None:
        N = len(xs)
        assert all(N == len(x) for x in [pitches, tsrs, yaws, ys])

        # Convert kwts ans sigmas to list if single value is given
        if kwts is None or type(kwts) in [int, float]:
            kwts = N * [kwts]

        if sigmas is None or type(sigmas) in [int, float]:
            sigmas = N * [sigmas]

        # Iteratively instantiate turbines
        self.turbines = []
        for pitch, tsr, yaw, x, y, kw in zip(pitches, tsrs, yaws, xs, ys, kwts):
            self.turbines.append(BEMTurbine(pitch, tsr, yaw, rotor, x, y, kw=kw))

        if summation == "linear":
            self.summation_method = Superposition.Linear()
        elif summation == "quadratic":
            raise NotImplementedError
        elif summation == "linearniayifar":
            self.summation_method = Superposition.LinearNiayifar()
        elif summation == "quadraticniayifar":
            raise NotImplementedError
        elif summation == "zong":
            raise NotImplementedError
        else:
            raise ValueError(f"Wake summation method {summation} not found.")

        # self.REWS = self._REWS_at_rotors()

        idx_upstream = []
        for idx in np.argsort(xs):
            mu_mesh, theta_mesh = self.turbines[idx].gridpoints()

            X = self.turbines[idx].x * np.ones_like(mu_mesh)
            Y = (
                0.5 * mu_mesh * np.sin(theta_mesh) + self.turbines[idx].y
            )  # POSSIBLE SIGN ERROR
            Z = 0.5 * mu_mesh * np.cos(theta_mesh)

            base_U = np.ones_like(mu_mesh)

            deficits = []
            for iidx in idx_upstream:
                deficits.append(self.turbines[iidx].deficit(X, Y, Z))

            # Linear superposition
            U = base_U
            for deficit in deficits:
                U -= deficit

            windfield = Custom(U)

            self.turbines[idx].post_init(windfield)
            idx_upstream.append(idx)

    def _REWS_at_rotors(self) -> np.ndarray:
        # Get turbine locations
        N = len(self.turbines)
        X_t = np.array([turbine.x for turbine in self.turbines])
        Y_t = np.array([turbine.y for turbine in self.turbines])

        # Define gridpoints to sample based on REWS method.
        Xs, Ys, Zs = self.REWS_method.grid_points(X_t, Y_t)

        deficits = np.zeros((N, *Xs.shape))
        for i, turbine in enumerate(self.turbines):
            deficits[i, :] = turbine.deficit(Xs, Ys, Zs)

        # ignore effect of own wake on self
        for i in range(N):
            deficits[i, i, :] = 0

        # Perform summation
        REWS = self.summation_method.calculate_REWS(deficits, self.REWS_method, self)

        return REWS
