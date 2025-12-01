from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------
# KMeans sweeps over k
# ---------------------------------------------------------------------

def kmeans_sweep(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Fit KMeans for k in [k_min, k_max], compute inertia and silhouette.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled or PCA-reduced).
    k_min, k_max : int
        Range of k to try.
    random_state : int
        Random seed.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: ['k', 'inertia'].
    scores_silhouette : dict
        Mapping k -> silhouette score.
    """
    results = []
    scores_silhouette: Dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        km_model = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = km_model.fit_predict(X)

        inertia = km_model.inertia_
        sil = silhouette_score(X, labels) if k > 1 else np.nan

        results.append({"k": k, "inertia": inertia})
        scores_silhouette[k] = sil

    return pd.DataFrame(results), scores_silhouette


# ---------------------------------------------------------------------
# GMM sweeps over k (with BIC/AIC + silhouette)
# ---------------------------------------------------------------------

def gmm_sweep_labeled(
    X: np.ndarray,
    space: str,
    k_min: int = 2,
    k_max: int = 8,
    covariance_type: str = "full",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit GaussianMixture for k in [k_min, k_max], compute BIC, AIC and silhouette.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled or PCA-reduced).
    space : str
        Label describing the feature space ("RAW" or "PCA"),
        used only for identification in the results.
    k_min, k_max : int
        Range of k to try.
    covariance_type : str
        Covariance type for GMM.
    random_state : int
        Random seed.

    Returns
    -------
    df : pd.DataFrame
        Columns: ['space', 'k', 'bic', 'aic', 'silhouette'].
    """
    rows = []

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm.fit(X)

        bic = gmm.bic(X)
        aic = gmm.aic(X)
        labels = gmm.predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan

        rows.append(
            {
                "space": space,
                "k": k,
                "bic": bic,
                "aic": aic,
                "silhouette": sil,
            }
        )

    return pd.DataFrame(rows)
