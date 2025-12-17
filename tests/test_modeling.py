import numpy as np
import pandas as pd

from src import data_loader, processing, feature_engineering, modeling
from src.clustering import core


def test_modeling_pipeline_end_to_end():
    """
    End-to-end test of the modeling pipeline:

    1. Download raw SP500 + VIX data
    2. Prepare market data
    3. Add features
    4. Build feature matrix + scale + PCA
    5. Cluster with KMeans and assign regimes
    6. Build supervised dataset (X_t, y_{t+1})
    7. Run ML splits (Logistic, RF, GB)
    8. Build a majority-vote regime prediction
    """

    # 1) Download raw data
    df_raw = data_loader.download_raw_data(
        start_date="2000-01-01",
        end_date="2024-12-31",
    )
    assert isinstance(df_raw, pd.DataFrame)
    assert not df_raw.empty

    # 2) Prepare market data
    df_proc = processing.prepare_market_data(df_raw)
    assert "SP500_Close" in df_proc.columns
    assert "VIX_Close" in df_proc.columns

    # 3) Add features
    df_feat = feature_engineering.add_spx_vix_features(df_proc)

    # Make sure core features are present
    for col in core.DEFAULT_FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature {col}"

    # 4) Build feature matrix + scale + PCA (for clustering)
    X, idx, feat_cols = core.build_feature_matrix(df_feat, core.DEFAULT_FEATURE_COLS)

    X_scaled, scaler = core.scale_features(X)
    assert X_scaled.shape == X.shape

    X_pca, n_components, pca = core.pca_analysis(X_scaled, variance_target=0.9)
    assert 1 <= n_components <= X.shape[1]
    assert X_pca.shape == (X.shape[0], n_components)

    # 5) Cluster with KMeans and assign regimes
    labels, km_model = core.fit_kmeans(X_pca, n_clusters=2, random_state=42)
    assert len(labels) == X.shape[0]
    assert set(np.unique(labels)) <= {0, 1}

    df_assigned = core.assign_cluster_labels(
        df=df_feat,
        idx=idx,
        labels=labels,
        col_name="KMeans_Regime",
    )

    assert "KMeans_Regime" in df_assigned.columns
    assert df_assigned.loc[idx, "KMeans_Regime"].isna().sum() == 0

    # 6) Build supervised dataset (X_t, y_{t+1})
    df_sup, X_all, y_all = modeling.build_supervised_dataset(
        df_assigned=df_assigned,
        feature_cols=feat_cols,
        regime_col="KMeans_Regime",
    )

    assert isinstance(df_sup, pd.DataFrame)
    assert not df_sup.empty
    assert X_all.shape[0] == len(df_sup)
    assert X_all.shape[1] == len(feat_cols)
    assert y_all.shape[0] == len(df_sup)
    assert set(y_all.unique()) <= {0, 1}

    # 7) Run ML splits
    # Use one reasonably large, non-overlapping split to keep the test fast
    splits = {
        "2010-18 to 2019-23": (
            "2010-01-01", "2018-12-31",
            "2019-01-01", "2023-12-31",
        ),
    }

    results_df, preds_df = modeling.run_splits(
        df_sup=df_sup,
        X_all=X_all,
        y_all=y_all,
        splits=splits,
    )

    # Basic checks on results
    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty

    required_metric_cols = {
        "accuracy", "balanced_accuracy", "f1", "roc_auc", "split", "model"
    }
    assert required_metric_cols.issubset(results_df.columns)

    # Metrics should be within [0, 1]
    for col in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        assert results_df[col].between(0, 1).all()

    # Predictions dataframe checks
    assert isinstance(preds_df, pd.DataFrame)
    assert not preds_df.empty
    for col in ["split", "model", "date", "y_true", "y_pred", "y_prob"]:
        assert col in preds_df.columns

    # 8) Majority vote prediction for that split
    vote_series = modeling.build_majority_vote(
        preds_df=preds_df,
        split_name="2010-18 to 2019-23",
    )

    assert isinstance(vote_series, pd.Series)
    assert not vote_series.empty
    assert set(vote_series.unique()).issubset({0, 1})

    print("\nmodeling pipeline test passed")


if __name__ == "__main__":
    test_modeling_pipeline_end_to_end()
