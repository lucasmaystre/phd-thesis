"""(Penalized) maximum-likelihood inference for Thurstone's model."""
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm


class _Functions:

    """Optimization-related methods for Thurstone's pairwise comparison model.

    This class provides methods to compute the negative log-likelihood (the
    "objective") and its gradient, given model parameters and
    pairwise-comparison data.
    """

    def __init__(self, mat, penalty):
        self._mat = mat
        self._penalty = penalty

    def objective(self, params):
        """Compute the negative penalized log-likelihood."""
        reg = self._penalty * np.sum(params**2)
        diffs = np.tile(params, (len(params), 1))
        diffs = diffs.T - diffs  # diffs[i,j] = x[i] - x[j]
        return reg - np.sum(self._mat * norm.logcdf(diffs))

    def gradient(self, params):
        grad = 2 * self._penalty * params
        diffs = np.tile(params, (len(params), 1))
        diffs = diffs.T - diffs  # diffs[i,j] = x[i] - x[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = norm.pdf(diffs) / norm.cdf(diffs)
        for i in range(len(params)):
            grad[i] += np.dot(self._mat[:,i], ratios[:,i])
            grad[i] -= np.dot(self._mat[i,:], ratios[i,:])
        return grad


def thurstone_mle(mat, penalty=1e-6, max_iter=None, tol=1e-5):
    # mat[i,j] should contain the number of wins of i against j.
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], (
        "data should be a squared matrix with win counts")
    x0 = np.zeros(mat.shape[0])
    fcts = _Functions(mat, penalty)
    # `gtol`: Gradient norm must be less than gtol before successful
    # termination [scipy doc].
    res = minimize(
            fcts.objective, x0, method="BFGS", jac=fcts.gradient,
            options={"gtol": tol, "maxiter": max_iter})
    return res.x
