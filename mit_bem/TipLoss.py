import numpy as np


def PrandtlTiploss(mu, phi, B=3):
    f = B / 2 * (1 - mu) / (mu * np.abs(np.sin(phi)))
    F = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f, -100, 100)), -1, 1))
    return np.maximum(F, 0.01)


def PrandtlTipAndRootLossGenerator(R_hub, B=3):
    """
    Returns a function (of mu and phi) for the tip loss including root correction.
    R_hub should be nondimensional.
    """

    def func(mu, phi):
        f_tip = B / 2 * (1 - mu) / (mu * np.abs(np.sin(phi)))
        F_tip = (
            2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -1, 1))
        )
        f_hub = B / 2 * (mu - R_hub) / (mu * np.abs(np.sin(phi)))
        F_hub = (
            2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -1, 1))
        )

        return np.maximum(F_hub * F_tip, 0.00001)

    return func


def NoTiploss(mu, phi):
    return np.ones_like(mu)
