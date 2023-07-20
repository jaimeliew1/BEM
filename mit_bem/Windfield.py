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


class ShearVeer:
    def __init__(self, zr, exp, dveerdz, z0veer):
        """
        Preliminary parameterisations for wind shear and veer.
        Assumes power law shear and linear veer.

        args:
        zr (float): reference height (m) for wind shear
        exp (float): shear exponent
        dveerdz (float): rate of change of veer angle (rad) per vertical meters.
        z0veer (float): height (m) where veer is 0.
        """
        self.zr = zr
        self.exp = exp
        self.dveerdz = dveerdz
        self.z0veer = z0veer

    def wsp(self, Y, Z):
        return (Z / self.zr) ** self.exp

    def wdir(self, Y, Z):
        return self.dveerdz * (Z - self.z0veer)
