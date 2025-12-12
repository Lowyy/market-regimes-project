from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Default feature set
# ---------------------------------------------------------------------

DEFAULT_FEATURE_COLS: List[str] = [
    "log_daily_return",
    "20d_return",
    "5d_volatility",
    "20d_volatility",
    "60d_volatility",
    "vol_of_vol_20d",
    "realized_quarticity_20d",
    "realized_skew_20d",
    "realized_kurtosis_20d",
    "return_Zscore_20d",
    "slope_MA_20",
    "slope_MA_50",
    "slope_MA_200",
    "current_drawdown",
    "max_drawdown_252d",
    "downside_dev_20d",
    "autocorr_20d",
    "dVIX_1d",
    "dVIX_5d",
    "VIX_Zscore_20d",
    "VIX_vol_20d",
    "realized_vol_20d_ann",
    "VRP_Proxy",
    "corr_ret_dVIX_20d",
    "ret_x_VIX",
    "vol_x_VIX",
    "vol_ratio",
    "VIX_Zscore_X_DD",
    "RSI_20",
    "hurst_100d",
    "ADX_14",
]


# ---------------------------------------------------------------------
# 1. Build feature matrix and scale
# ---------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Tuple[np.ndarray, pd.Index, List[str]]:
    """
    Extract the feature matrix X from a feature-rich dataframe,
    dropping rows with NaNs in the selected feature columns.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    index : pd.Index
        Index of the rows kept (for aligning labels back to df).
    feature_cols : list[str]
        List of feature column names actually used.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    X_df = df[feature_cols].dropna()
    X = X_df.values
    idx = X_df.index

    return X, idx, feature_cols


def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to zero mean and unit variance.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix.
    scaler : StandardScaler
        Fitted scaler object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------
# 2. PCA analysis
# ---------------------------------------------------------------------

def pca_analysis(
    X_scaled: np.ndarray,
    variance_target: float = 0.9,
) -> Tuple[np.ndarray, int, PCA]:
    """
    Perform PCA on scaled features and reduce to the number of
    components that explain at least `variance_target` of variance.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    variance_target : float
        Desired fraction of explained variance (e.g. 0.9 for 90%).

    Returns
    -------
    X_pca_reduced : np.ndarray
        PCA-transformed data with n_components columns.
    n_components : int
        Number of components selected to reach variance_target.
    pca : PCA
        Fitted PCA object.
    """
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    n_components = np.argmax(np.cumsum(explained_var) >= variance_target) + 1
    X_pca_reduced = X_pca[:, :n_components]

    return X_pca_reduced, n_components, pca


# ---------------------------------------------------------------------
# 3. Fit final models (KMeans & GMM)
# ---------------------------------------------------------------------

def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Fit final KMeans with chosen n_clusters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled or PCA-reduced).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each observation.
    km : KMeans
        Fitted KMeans model.
    """
    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def fit_gmm(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = "full",
    random_state: int = 42,
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Fit final GaussianMixture with chosen n_components.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled or PCA-reduced).
    n_components : int
        Number of mixture components.
    covariance_type : str
        GMM covariance type ('full', 'tied', 'diag', 'spherical').
    random_state : int
        Random seed.

    Returns
    -------
    labels : np.ndarray
        Hard cluster assignments.
    gmm : GaussianMixture
        Fitted GMM model.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = gmm.fit_predict(X)
    return labels, gmm


# ---------------------------------------------------------------------
# 4. Attach cluster labels back to the dataframe
# ---------------------------------------------------------------------

def assign_cluster_labels(
    df: pd.DataFrame,
    idx: pd.Index,
    labels: np.ndarray,
    col_name: str,
) -> pd.DataFrame:
    """
    Attach cluster labels back to a copy of the original dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe (with all dates).
    idx : pd.Index
        Index of rows used in clustering (from build_feature_matrix).
    labels : np.ndarray
        Cluster labels for those rows.
    col_name : str
        Name of the column to store labels in.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with a new Int64 column containing cluster labels
        (NaN for rows that were not included in X).
    """
    df_out = df.copy()
    df_out[col_name] = np.nan
    df_out.loc[idx, col_name] = labels
    df_out[col_name] = df_out[col_name].astype("Int64")
    return df_out
