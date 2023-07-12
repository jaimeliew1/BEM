from typing import Callable
import numpy as np


def fixedpointiteration(
    f: Callable[[np.ndarray, any], np.ndarray],
    x0: np.ndarray,
    args=(),
    eps=0.000001,
    maxiter=100,
    relax=0,
) -> np.ndarray:
    """
    Performs fixed-point iteration on function f until residuals converge or max
    iterations is reached.

    Args:
        f (Callable): residual function of form f(x, *args) -> np.ndarray
        x0 (np.ndarray): Initial guess
        args (tuple): arguments to pass to residual function. Defaults to ().
        eps (float): Convergence tolerance. Defaults to 0.000001.
        maxiter (int): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: Max iterations reached.

    Returns:
        np.ndarray: Solution to residual function.
    """
    for c in range(maxiter):
        residuals = f(x0, *args)

        x0 += (1 - relax) * residuals
        if np.abs(residuals).max() < eps:
            break
    else:
        raise ValueError("max iterations reached.")

    # print(f"niter: {c}")
    return x0


def adaptivefixedpointiteration(
    f: Callable[[np.ndarray, any], np.ndarray],
    x0: np.ndarray,
    args=(),
    eps=0.000001,
    maxiter=100,
):
    for relax in [0.0, 0.3, 0.7, 0.9]:
        try:
            ans = fixedpointiteration(f, x0, args, eps, maxiter, relax)
            return ans
        except ValueError:
            pass
    raise ValueError("max iterations reached.")
