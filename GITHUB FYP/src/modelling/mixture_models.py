import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gmms_bic(F, k_list=(1, 2, 3, 4, 5), seed=0):
    """
    Fit GMMs for different K and select best using BIC.
    """
    rows = []
    best_bic = None
    best_gmm = None

    for k in k_list:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            reg_covar=1e-6
        )
        gmm.fit(F)

        bic = gmm.bic(F)
        aic = gmm.aic(F)

        rows.append((k, bic, aic))

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    return rows, best_gmm