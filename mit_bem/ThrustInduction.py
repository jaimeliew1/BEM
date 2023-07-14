import numpy as np
from .Utilities import aggregate


def Ct2a_HAWC2(Ct, tiploss):
    Ct = Ct / tiploss
    return 0.0883 * Ct**3 + 0.0586 * Ct**2 + 0.246 * Ct


def HAWC2(bem_obj):
    Ct = np.minimum(
        (1 - bem_obj._a) ** 2
        * bem_obj.solidity
        * bem_obj._Cax
        / np.sin(bem_obj._phi) ** 2,
        4,
    )
    Ct = Ct / bem_obj._tiploss
    a = 0.0883 * Ct**3 + 0.0586 * Ct**2 + 0.246 * Ct
    return Ct, a


def mike(bem_obj):
    Ct = bem_obj._W**2 * bem_obj.solidity * bem_obj._Cax

    Ct_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, Ct, agg="rotor")

    a_target = (
        2 * Ct_rotor
        - 4
        + np.sqrt(-(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16)
    ) / (-4 + np.sqrt(-(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16))

    a_new = bem_obj._tiploss
    a_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, a_new, agg="rotor")
    a_new *= a_target / a_rotor

    return Ct, a_new


def Ct2a_Mike(mu, theta_mesh, Ct, tiploss, yaw):
    Ct_rotor = aggregate(mu, theta_mesh, Ct, agg="rotor")

    a_target = (
        2 * Ct_rotor
        - 4
        + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)
    ) / (-4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16))

    a_new = tiploss
    a_rotor = aggregate(mu, theta_mesh, a_new, agg="rotor")
    a_new *= a_target / a_rotor

    return a_new
