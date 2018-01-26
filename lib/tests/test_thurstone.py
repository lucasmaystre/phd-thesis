import math
import numpy as np

from thesis.thurstone import _Functions
from scipy.optimize import check_grad, approx_fprime


RND = np.random.RandomState(42)
EPS = math.sqrt(np.finfo(float).eps)

# `8-random` case.
MAT = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 2, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 2, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 2, 0, 0],
    [1, 0, 0, 2, 0, 0, 1, 0]])


def test_gradient():
    """Gradient of pairwise-data objective should be correct."""
    fcts = _Functions(MAT, 0.2)
    for sigma in np.linspace(1, 5, num=5):
        xs = sigma * RND.randn(len(MAT))
        val = approx_fprime(xs, fcts.objective, EPS)
        err = check_grad(fcts.objective, fcts.gradient, xs, epsilon=EPS)
        assert abs(err / np.linalg.norm(val)) < 1e-5
