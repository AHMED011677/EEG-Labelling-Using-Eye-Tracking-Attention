# src/stats_analysis.py

import numpy as np
from scipy.stats import shapiro, normaltest, anderson, chi2


def cov_and_corr(x, y):
    """
    Compute 2x2 covariance matrix and correlation between x and y.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    Z = np.vstack([x, y])  # shape (2, n)
    cov = np.cov(Z, bias=False)

    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return cov, float(corr)


def gaussianity_univariate(x, max_n=5000):
    """
    Run common univariate normality tests.
    """
    x = np.asarray(x).ravel()
    x_test = x[:max_n] if x.size > max_n else x

    sh = shapiro(x_test)          # Shapiro-Wilk
    k2 = normaltest(x_test)       # D’Agostino K^2
    ad = anderson(x_test, dist="norm")

    return {
        "n_used": int(x_test.size),
        "shapiro_stat": float(sh.statistic),
        "shapiro_p": float(sh.pvalue),
        "dagostino_stat": float(k2.statistic),
        "dagostino_p": float(k2.pvalue),
        "anderson_stat": float(ad.statistic),
        "anderson_critical_values": ad.critical_values.tolist(),
        "anderson_significance_levels": ad.significance_level.tolist(),
    }


def multivariate_gaussian_check(x, y):
    """
    Multivariate (2D) Gaussian check via Mahalanobis d^2 quantiles
    compared to Chi-square(df=2).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    Z = np.column_stack([x, y])   # shape (n, 2)
    mu = Z.mean(axis=0)
    S = np.cov(Z, rowvar=False)

    # small ridge for numerical stability
    Sinv = np.linalg.inv(S + 1e-10 * np.eye(2))
    diffs = Z - mu

    d2 = np.einsum("ij,jk,ik->i", diffs, Sinv, diffs)

    q = [0.5, 0.9, 0.95, 0.99]
    empirical = np.quantile(d2, q)
    theoretical = chi2.ppf(q, df=2)

    return {
        "empirical_d2": empirical.tolist(),
        "chi2_theoretical": theoretical.tolist(),
    }