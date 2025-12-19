from pathlib import Path

import pandas as pd

from src import (
    data_loader,
    processing,
    feature_engineering,
    modeling,
    strategies,
    reporting,
)
from src.clustering import core


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# Split used for diagnostics / strategies
MAIN_SPLIT_NAME = "2010-20 to 2021-24"

SPLITS = {
    "2010-20 to 2021-24": ("2010-01-01", "2020-12-31", "2021-01-01", "2024-12-31"),
    "2003-15 to 2016-20": ("2003-01-01", "2015-12-31", "2016-01-01", "2020-12-31"),
}


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    print("\n[Step 1] Downloading raw data ...")
    df_raw = data_loader.download_raw_data(start_date=START_DATE, end_date=END_DATE)
    print(f"  Raw data shape: {df_raw.shape}")

    data_loader.save_raw_data(df_raw, filename="raw_data.parquet")

    print("\n[Step 2] Preparing market data (SPX + VIX) ...")
    df_proc = processing.prepare_market_data(df_raw)
    print(f"  Processed data shape: {df_proc.shape}")

    print("\n[Step 3] Adding SPX/VIX features ...")
    df_with_features = feature_engineering.add_spx_vix_features(df_proc)
    print(f"  Data with features shape: {df_with_features.shape}")

    data_loader.save_processed_data(df_with_features, filename="processed_data.parquet")

    # Unsupervised regimes: PCA + KMeans
    print("\n[Step 4] Building feature matrix and running PCA/KMeans ...")
    features = core.DEFAULT_FEATURE_COLS

    X, idx_features, _ = core.build_feature_matrix(df_with_features, features)
    print(f"  Feature matrix shape: {X.shape}")

    X_scaled, scaler = core.scale_features(X)
    X_pca_reduced, n_components, pca = core.pca_analysis(
        X_scaled,
        variance_target=0.9,
    )
    print(f"  PCA components used: {n_components}")

    labels, kmeans_model = core.fit_kmeans(
        X_pca_reduced,
        n_clusters=2,
        random_state=42,
    )
    df_assigned = core.assign_cluster_labels(
        df=df_with_features,
        idx=idx_features,
        labels=labels,
        col_name="KMeans_Regime",
    )

    print("  Regime counts (KMeans):")
    print(df_assigned["KMeans_Regime"].value_counts(dropna=True))

    # Supervised regime prediction
    print("\n[Step 5] Building supervised dataset and running models ...")
    df_sup, X_all, y_all = modeling.build_supervised_dataset(
        df_assigned=df_assigned,
        feature_cols=features,
        regime_col="KMeans_Regime",
    )
    print(f"  Supervised dataset shape: {df_sup.shape}")
    print("  Target distribution:")
    print(y_all.value_counts())

    results_df, preds_df = modeling.run_splits(
        df_sup=df_sup,
        X_all=X_all,
        y_all=y_all,
        splits=SPLITS,
    )

    print("\n  ML results by split/model:")
    print(results_df)

    # Majority-votes prediction for MAIN_SPLIT_NAME
    print(f"\n[Step 6] Building majority-vote predictions for split: {MAIN_SPLIT_NAME}")
    vote_pred = modeling.build_majority_vote(
        preds_df=preds_df,
        split_name=MAIN_SPLIT_NAME,
    )

    # For trading: today's position uses yesterday's predicted regime
    pred_regime_today = vote_pred.shift(1).dropna()
    print(f"  Predicted regimes available from {pred_regime_today.index.min().date()} "
          f"to {pred_regime_today.index.max().date()} "
          f"({len(pred_regime_today)} observations)")

    # Strategies
    print("\n[Step 7] Running strategies ...")
    strategy_results = strategies.run_all_strategies(
        pred_regime_today=pred_regime_today,
        df=df_assigned,
    )

    # Show a small performance summary in the console
    print("\n  Strategy performance (ann_return, ann_vol, sharpe, sortino, max_drawdown):")
    for name, res in strategy_results.items():
        perf = res["performance"]
        print(f"    {name:12s} → "
              f"ret={perf['ann_return']:.3f}, "
              f"vol={perf['ann_vol']:.3f}, "
              f"sharpe={perf['sharpe']:.3f}, "
              f"sortino={perf['sortino']:.3f}, "
              f"maxDD={perf['max_drawdown']:.3f}")

    # Reporting
    print("\n[Step 8] Generating reporting (plots + tables) ...")
    reporting.run_full_reporting(
        df_with_features=df_with_features,
        df_assigned=df_assigned,
        idx_features=idx_features,
        pca=pca,
        results_df=results_df,
        preds_df=preds_df,
        strategy_results=strategy_results,
        split_for_diagnostics=MAIN_SPLIT_NAME,
        model_for_diagnostics="LogisticRegression",
        # results_dir optional; default is PROJECT_ROOT / "results"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
