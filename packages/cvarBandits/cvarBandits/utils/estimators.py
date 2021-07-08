import numpy as np
import itertools
from scipy.stats import chi2
from pprint import pprint


def empirical_cvar(samples, alpha):
    """
    Simple empirical estimation of the CVaR from quantile 0 to alpha
    @param samples: i.i.d. samples
    @type samples: list, 1D
    @param alpha: CVaR's alpha level
    @type alpha: float
    @return: empirical CVaR @ alpha
    @rtype: float
    """
    n = len(samples)
    sorted_samples = sorted(samples)
    n_alpha = int(np.ceil(n * alpha))
    to_n_alpha_values = sorted_samples[:n_alpha]
    emp_cvar = np.mean(to_n_alpha_values)
    return emp_cvar


def moment_empirical_cvar(samples, alpha):
    """
    Quantile 0 to alpha CVaR moment based estimator as described in https://arxiv.org/abs/1401.1123
    @param samples: i.i.d. samples
    @type samples: list, 1D
    @param alpha: CVaR's alpha level
    @type alpha: float
    @return: empirical CVaR @ alpha
    @rtype: float
    """
    sorted_samples = np.sort(samples)
    n = int(alpha * len(samples))
    return samples[n] + 1 / alpha / len(samples) * (sorted_samples[:n] - samples[n]).sum()


def cvar_ci(samples, b_t, alpha, support, upper=True):
    """
    Confidence intervals for quantile 0 to quantile alpha CVaR of a bounded real valued random variable.
    Adapted from https://arxiv.org/pdf/1901.00997.pdf
    @param samples: i.i.d. samples
    @type samples: list, 1D
    @param b_t: term to be added/discounted to the CDF for CVaR confidence bounds
    @type b_t: 0 < float
    @param alpha: CVaR's alpha level
    @type alpha: float
    @param supp: bounded real valued random variable support
    @type supp: list [lower_support_bound, upper_support_bound]
    @param upper: if True, computation of an upper bound, else of the lower
    @type upper: bool
    @return: CVaR @ alpha confidence bounds
    @rtype: float
    """
    sorted_samples = sorted(samples)
    n = len(samples)
    if upper:
        term1 = np.diff(sorted_samples, append=support[1])
        term2 = np.maximum(np.minimum([i / n - b_t for i in range(1, n + 1)], alpha), 0)
        term = term1 * term2
        bound = support[1] - 1 / alpha * term.sum()
    else:
        term1 = np.diff(sorted_samples, prepend=support[0])
        term2 = np.minimum([i / n + b_t for i in range(n)], alpha)
        term = term1 * term2
        bound = sorted_samples[-1] - 1 / alpha * term.sum()
    return bound


def paired_variance(samples):
    """
    performs 1 / (n * (n-1)) sum i > j sum j (xi - xj) ** 2
    empirical estimate of the variance
    """
    n = len(samples)
    assert n > 1
    combinations = list(itertools.combinations(samples, 2))
    paired_var = (np.diff(combinations) ** 2).sum() / n / (n - 1)
    return paired_var
