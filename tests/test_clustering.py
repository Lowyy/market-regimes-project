import numpy as np
import pandas as pd

from src import data_loader, processing, feature_engineering
from src.clustering import core

# Time between the start date and end date shoud be long enough for this test
# since a lot of rows are flushed out due to features requiring long timeframes before yielding a non-NaN value

def test_clustering_pipeline():
    """
    End-to-end test of the clustering pipeline:

    1. Download raw SPX + VIX data
    2. Prepare market data
    3. Add features
    4. Build feature matrix + scale
    5. Run PCA
    6. Fit KMeans and GMM
    7. Assign KMeans labels back to the dataframe
    """

    # 1) Download a small sample of data
    df_raw = data_loader.download_raw_data(
        start_date="2010-01-01",
        end_date="2020-01-01",
    )
    assert isinstance(df_raw, pd.DataFrame)
    assert not df_raw.empty


    # 2) Prepare market data (rename / flatten)
    df_proc = processing.prepare_market_data(df_raw)
    assert "SP500_Close" in df_proc.columns
    assert "VIX_Close" in df_proc.columns

    # 3) Add features
    df_feat = feature_engineering.add_spx_vix_features(df_proc)
    for col in core.DEFAULT_FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature: {col}"

    # 4) Build feature matrix & scale
    X, idx, feat_cols = core.build_feature_matrix(df_feat, core.DEFAULT_FEATURE_COLS)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(idx)
    assert X.shape[1] == len(feat_cols)
    assert len(X) > 50  # should have enough rows

    X_scaled, scaler = core.scale_features(X)
    assert X_scaled.shape == X.shape

    # Check standardization
    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))


    # 5) PCA
    X_pca, n_components, pca = core.pca_analysis(X_scaled, variance_target=0.9)

    assert 1 <= n_components <= X.shape[1]
    assert X_pca.shape == (X.shape[0], n_components)
    # cumulative explained variance should be >= 0.9 by construction
    assert pca.explained_variance_ratio_.cumsum()[n_components - 1] >= 0.9 - 1e-6

    # 6) KMeans & GMM
    km_labels, km_model = core.fit_kmeans(X_pca, n_clusters=2, random_state=42)
    assert len(km_labels) == X.shape[0]
    assert set(np.unique(km_labels)) <= {0, 1}

    gmm_labels, gmm_model = core.fit_gmm(X_pca, n_components=2, random_state=42)
    assert len(gmm_labels) == X.shape[0]
    assert set(np.unique(gmm_labels)) <= {0, 1}

    # 7) Assign KMeans labels back to dataframe
    df_labeled = core.assign_cluster_labels(
        df=df_feat,
        idx=idx,
        labels=km_labels,
        col_name="KMeans_Regime",
    )

    assert "KMeans_Regime" in df_labeled.columns
    # all rows used in X should have a non-null label
    assert df_labeled.loc[idx, "KMeans_Regime"].isna().sum() == 0

    print("\nclustering pipeline test passed")


if __name__ == "__main__":
    test_clustering_pipeline()
