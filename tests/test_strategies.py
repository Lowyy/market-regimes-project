import numpy as np
import pandas as pd

from src import (
    data_loader,
    processing,
    feature_engineering,
    modeling,
    strategies,
)
from src.clustering import core


def test_strategies_end_to_end():
    """
    End-to-end test for the strategies module:

    1. Download raw data
    2. Prepare market data
    3. Add features
    4. Build feature matrix + scale + PCA
    5. KMeans clustering and assign regimes
    6. Build supervised dataset
    7. Run ML splits and build majority-vote prediction
    8. Run all strategies on the majority-vote regimes
    """

    # 1) Download raw SP500 + VIX data
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

    # 3) Add engineered features
    df_feat = feature_engineering.add_spx_vix_features(df_proc)

    # Ensure all default features are present
    for col in core.DEFAULT_FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature {col}"

    # 4) Build feature matrix + scale + PCA (for clustering)
    X, idx, feat_cols = core.build_feature_matrix(df_feat, core.DEFAULT_FEATURE_COLS)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(idx)
    assert X.shape[0] > 200  # sanity check

    X_scaled, scaler = core.scale_features(X)
    assert X_scaled.shape == X.shape

    X_pca, n_components, pca = core.pca_analysis(X_scaled, variance_target=0.9)
    assert X_pca.shape[0] == X.shape[0]
    assert 1 <= n_components <= X.shape[1]

    # 5) KMeans clustering and assign regimes
    labels, km = core.fit_kmeans(X_pca, n_clusters=2, random_state=42)
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
    assert not df_sup.empty
    assert X_all.shape[0] == len(df_sup)
    assert y_all.shape[0] == len(df_sup)
    assert set(y_all.unique()) <= {0, 1}

    # 7) Run ML split and compute majority-vote regimes
    splits = {
        "2010-20 to 2021-24": (
            "2010-01-01", "2020-12-31",
            "2021-01-01", "2024-12-31",
        ),
    }

    results_df, preds_df = modeling.run_splits(
        df_sup=df_sup,
        X_all=X_all,
        y_all=y_all,
        splits=splits,
    )

    assert not preds_df.empty

    vote_series = modeling.build_majority_vote(
        preds_df=preds_df,
        split_name="2010-20 to 2021-24",
    )
    assert isinstance(vote_series, pd.Series)
    assert not vote_series.empty
    assert set(vote_series.unique()).issubset({0, 1})

    # today's regime signal is yesterday's prediction
    pred_regime_today = vote_series.shift(1).dropna()
    assert not pred_regime_today.empty

    # 8) Run all strategies and sanity-check outputs
    results = strategies.run_all_strategies(pred_regime_today, df_assigned)

    # We expect these strategy keys from strategies.py
    expected_strats = {"strat1", "strat2", "strat3A", "strat3B", "strat3"}
    assert expected_strats.issubset(results.keys())

    for name, out in results.items():
        positions = out["positions"]
        rets = out["returns"]
        perf = out["performance"]

        # Index alignment
        assert isinstance(positions, pd.Series)
        assert isinstance(rets, pd.Series)
        assert positions.index.equals(pred_regime_today.index)
        assert rets.index.equals(pred_regime_today.index)

        # No NaNs in positions / returns
        assert not positions.isna().any()
        assert not rets.isna().any()

        # Positions only in {-1, 0, 1}
        assert set(positions.unique()) <= {-1.0, 0.0, 1.0}

        # Performance dict has all expected keys
        for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"]:
            assert key in perf

    print([print(f"\n{k}: {results[k]['performance']}") for k, v in results.items()])

    print("\nstrategies pipeline test passed if test results are close to what's awaited")


if __name__ == "__main__":
    test_strategies_end_to_end()
