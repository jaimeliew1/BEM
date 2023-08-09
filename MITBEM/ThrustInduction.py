import numpy as np
from .Utilities import aggregate


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
    return a


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

    return a_new


def mike_corrected(bem_obj, ac=1 / 3):
    Ct = bem_obj._W**2 * bem_obj.solidity * bem_obj._Cax

    Ct_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, Ct, agg="rotor")

    Ctc = 4 * ac * (1 - ac) / (1 + 0.25 * (1 - ac) ** 2 * np.sin(bem_obj.yaw) ** 2)
    slope = (16 * (1 - ac) ** 2 * np.sin(bem_obj.yaw) ** 2 - 128 * ac + 64) / (
        (1 - ac) ** 2 * np.sin(bem_obj.yaw) ** 2 + 4
    ) ** 2

    if Ct_rotor > Ctc:
        a_target = (Ct_rotor - Ctc) / slope + ac
    else:
        a_target = (
            2 * Ct_rotor
            - 4
            + np.sqrt(-(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16)
        ) / (
            -4
            + np.sqrt(-(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16)
        )

    a_new = bem_obj._tiploss
    a_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, a_new, agg="rotor")
    a_new *= a_target / a_rotor

    return a_new


def fixed_induction(bem_obj):
    return 1 / 3 * np.ones_like(bem_obj._a)
