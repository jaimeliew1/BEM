import numpy as np
from scipy import optimize
from tqdm import tqdm

from mit_bem.Windfarm import BEMWindfarm
from mit_bem.Turbine import IEA15MW


class Optimization:
    def __init__(self, X, Y, wf_kwargs={}):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.wf_kwargs = wf_kwargs

        self.minima_list = []
        self.f_list = []

    def _x0(self) -> np.ndarray:
        raise NotImplementedError

    def _bounds(self) -> np.ndarray:
        raise NotImplementedError

    def objective(self, x):
        raise NotImplementedError

    def find_optimal(self):
        x0 = self._x0()
        bounds = self._bounds()

        res = optimize.minimize(
            self.objective,
            x0=x0,
            bounds=bounds,
            jac=False,
        )

        setpoints = res.x
        Cp = -res.fun

        return Cp, setpoints

    def basinhop(self, total=100, progress=False):
        if progress:
            self.progressbar = tqdm(total=total + 1)
        else:
            self.progressbar = None

        res = optimize.basinhopping(
            self.objective,
            self._x0(),
            niter=total,
            # T=1,
            stepsize=3,
            callback=self._basin_callback,
            minimizer_kwargs=dict(
                bounds=self._bounds(),
                jac=False,
            ),
        )
        return self.f_list, self.minima_list

    def _basin_callback(self, x, f, accept):
        # iterate progress bar if it exists.
        if self.progressbar:
            self.progressbar.update()

        at_lower_bound = np.any(np.isclose(x, self._bounds()[:, 0], atol=1e-3))
        at_upper_bound = np.any(np.isclose(x, self._bounds()[:, 1], atol=1e-3))

        # Ignore optima at boundary
        if at_lower_bound or at_upper_bound:
            return

        # Ignore optima which have already been recorded
        if any(np.allclose(from_list, x, atol=1e-3) for from_list in self.minima_list):
            return

        self.minima_list.append(np.round(x, 3))
        self.f_list.append(-f)


# class YawOptimization(Optimization):
#     def _x0(self):
#         return self.N * [0]

#     def _bounds(self):
#         return np.array(self.N * [(-np.deg2rad(89), np.deg2rad(89))])

#     def objective(self, x):
#         Cts, yaws = 2 * np.ones(self.N), x
#         farm = GradWindfarm(self.X, self.Y, Cts, yaws, **self.wf_kwargs)
#         Cp, dCpdCt, dCpdyaw = farm.total_Cp()
#         return -Cp, -dCpdyaw

#     def find_optimal(cls, *args, **kwargs):
#         Cp, setpoints = super().find_optimal(*args, **kwargs)

#         return Cp, np.concatenate([2 * np.ones_like(setpoints), setpoints])


# class InductionOptimization(Optimization):
#     def _x0(self):
#         return self.N * [2]

#     def _bounds(self):
#         return np.array(self.N * [(0.00001, 4)])

#     def objective(self, x):
#         Cts, yaws = x, np.zeros(self.N)
#         farm = GradWindfarm(self.X, self.Y, Cts, yaws, **self.wf_kwargs)
#         Cp, dCpdCt, dCpdyaw = farm.total_Cp()
#         return -Cp, -dCpdCt

#     def find_optimal(cls, *args, **kwargs):
#         Cp, setpoints = super().find_optimal(*args, **kwargs)

#         return Cp, np.concatenate([setpoints, np.zeros_like(setpoints)])


class JointOptimization(Optimization):
    def _x0(self):
        return self.N * [0] + self.N * [8] + self.N * [0]

    def _bounds(self):
        return np.array(
            +self.N * [np.deg2rad((-10, 40))]
            + self.N * [(1, 20)]
            + self.N * [np.deg2rad((-89, 89))]
        )

    def objective(self, x):
        pitches, tsrs, yaws = x[: self.N], x[self.N : 2 * self.N], x[2 * self.N :]
        farm = BEMWindfarm(self.X, self.Y, pitches, tsrs, yaws, **self.wf_kwargs)
        Cp = np.mean([x.bem.Cp() for x in farm.turbines])
        return -Cp

    def find_optimal(cls, *args, **kwargs):
        Cp, setpoints = super().find_optimal(*args, **kwargs)
        return Cp, setpoints
