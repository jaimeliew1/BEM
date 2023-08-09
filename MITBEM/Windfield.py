import numpy as np


class Uniform:
    def __init__(self):
        pass

    def wsp(self, Y, Z):
        # Y is lateral, Z is vertical.
        return np.ones_like(Z)

    def wdir(self, Y, Z):
        # Y is lateral, Z is vertical.
        return np.zeros_like(Z)

    def __call__(self, Y, Z):
        return self.wsp(Y, Z), self.wdir(Y, Z)


class ShearVeer:
    def __init__(self, z0, exp, dveerdz):
        """
        Preliminary parameterisations for wind shear and veer.
        Assumes power law shear and linear veer.

        args:
        zr (float): reference height (-) nondimensionalized by rotor radius
        exp (float): shear exponent
        dveerdz (float): change in veer angle (rad) per rotor radius.
        """
        self.z0 = z0
        self.exp = exp
        self.dveerdz = dveerdz

    def wsp(self, Y, Z):
        return (Z / self.z0) ** self.exp

    def wdir(self, Y, Z):
        return np.deg2rad(self.dveerdz * (Z - self.z0))

    def __call__(self, Y, Z):
        return self.wsp(Y, Z), self.wdir(Y, Z)


class Custom:
    def __init__(self, U, wdir=None):
        self._U = U
        if wdir is None:
            self._wdir = np.zeros_like(U)
        else:
            assert wdir.shape == U.shape
            self._wdir = wdir

    def wsp(self, Y, Z):
        assert Y.shape == self._U.shape
        assert Z.shape == self._U.shape

        return self._U

    def wdir(self, Y, Z):
        assert Y.shape == self._U.shape
        assert Z.shape == self._U.shape

        return self._wdir

    def __call__(self, Y, Z):
        return self.wsp(Y, Z), self.wdir(Y, Z)
