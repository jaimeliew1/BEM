import numpy as np


def PrandtlTiploss(mu, phi, B=3):
    f = B / 2 * (1 - mu) / (mu * np.sin(phi))
    F = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f, -100, 100)), -1, 1))
    return F


def NoTiploss(mu, phi):
    return np.ones_like(mu)
