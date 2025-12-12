from pathlib import Path
from typing import Dict, Optional, Iterable
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.data_loader import PROJECT_ROOT
from src.clustering import core  # for DEFAULT_FEATURE_COLS

# ****************************************************************************
# 0. Paths & helpers
# ****************************************************************************

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ****************************************************************************
# 1. PCA reporting
# ****************************************************************************

def plot_pca_variance(pca, results_dir: Path = RESULTS_DIR) -> None:
    """
    Save cumulative explained variance and scree plot for a fitted PCA object.
    """
    _ensure_dir(results_dir)
    explained_var = pca.explained_variance_ratio_

    # Cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(explained_var), marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Cumulative Variance Explained")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "pca_cumulative_variance.png")
    plt.close()

    # Scree plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA - Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "pca_scree_plot.png")
    plt.close()


def compute_pca_loadings(
    pca,
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of PCA loadings (features x PCs).
    """
    if feature_cols is None:
        feature_cols = core.DEFAULT_FEATURE_COLS

    loadings = pd.DataFrame(
        pca.components_.T,
        index=list(feature_cols),
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    )
    return loadings


def top_loadings(load_df: pd.DataFrame, pc: str = "PC1", n: int = 8) -> pd.DataFrame:
    """
    Return n most negative and n most positive loadings for a given PC.
    """
    s = load_df[pc].sort_values()
    return pd.concat([s.head(n), s.tail(n)])


def save_pca_loadings(
    pca,
    feature_cols: Optional[Iterable[str]] = None,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Save:
      - full loadings table
      - top loadings for PC1 and PC2
    Returns a dict of DataFrames for further use in notebooks if needed.
    """
    _ensure_dir(results_dir)

    load_df = compute_pca_loadings(pca, feature_cols)
    load_df.to_csv(results_dir / "pca_loadings_full.csv")

    top_pc1 = top_loadings(load_df, "PC1", n=8)
    top_pc2 = top_loadings(load_df, "PC2", n=8)

    top_pc1.to_csv(results_dir / "pca_top_loadings_PC1.csv", header=["loading"])
    top_pc2.to_csv(results_dir / "pca_top_loadings_PC2.csv", header=["loading"])

    return {
        "full": load_df,
        "top_pc1": top_pc1.to_frame("loading"),
        "top_pc2": top_pc2.to_frame("loading"),
    }


# ****************************************************************************
# 2. Regime / clustering reporting
# ****************************************************************************

def plot_sp500_with_regimes(
    df_prices: pd.DataFrame,
    idx_features: pd.Index,
    regime_col: str = "KMeans_Regime",
    results_dir: Path = RESULTS_DIR,
    close_col: str = "SP500_Close",
) -> None:
    """
    Plot S&P 500 close with KMeans regimes as coloured points
    (only KMeans, no GMM here to avoid noise).
    """
    _ensure_dir(results_dir)

    df_used = df_prices.loc[idx_features, [close_col, regime_col]].dropna().copy()

    plt.figure(figsize=(12, 5))
    plt.plot(df_used.index, df_used[close_col], color="black", linewidth=1, alpha=0.6)
    plt.scatter(
        df_used.index,
        df_used[close_col],
        c=df_used[regime_col].astype(int),
        cmap=ListedColormap(["green", "red"]),
        s=4,
        alpha=0.9,
    )
    plt.title("S&P 500 with KMeans Regimes")
    plt.xlabel("Date")
    plt.ylabel(close_col)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "sp500_with_kmeans_regimes.png")
    plt.close()


def summarize_regimes(
    df_assigned: pd.DataFrame,
    regime_col: str = "KMeans_Regime",
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Compute a compact summary of each regime:
    volatility, VIX, drawdown, VRP.
    """
    _ensure_dir(results_dir)

    summary = df_assigned.groupby(regime_col)[
        ["20d_volatility", "VIX_Close", "current_drawdown", "VRP_Proxy"]
    ].agg(["mean", "median", "std", "count"])

    summary.to_csv(results_dir / "regime_summary_stats.csv")
    return summary


# ****************************************************************************
# 3. Machine learning reporting (supervised regime prediction)
# ****************************************************************************

def summarize_ml_results(
    results_df: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Save two CSVs:
      - raw per split x model metrics
      - average metrics per model across splits
    """
    _ensure_dir(results_dir)

    # Raw results
    results_df.to_csv(results_dir / "ml_results_by_split.csv", index=False)

    # Aggregated by model
    metric_cols = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    by_model = (
        results_df
        .groupby("model")[metric_cols]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    by_model.to_csv(results_dir / "ml_results_by_model_mean.csv")

    return {
        "by_split": results_df,
        "by_model": by_model,
    }


def plot_confusion_matrix_and_roc(
    preds_df: pd.DataFrame,
    split_name: str,
    model_name: str,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, object]:
    """
    For a given split/model (e.g. '2010-20 → 2021-24', 'LogisticRegression'),
    plot and save:
      - confusion matrix (as image)
      - ROC curve (as image)
    Returns underlying objects (cm, fpr, tpr, roc_auc) for further use.
    """
    _ensure_dir(results_dir)

    mask = (preds_df["split"] == split_name) & (preds_df["model"] == model_name)
    sub = preds_df[mask].copy().sort_values("date")

    y_true = sub["y_true"].values
    y_pred = sub["y_pred"].values
    y_prob = sub["y_prob"].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix – {model_name} ({split_name})")

    for (i, j), val in np.ndenumerate(cm_norm):
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(results_dir / f"cm_{model_name}_{split_name.replace(' ', '_')}.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {model_name} ({split_name})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"roc_{model_name}_{split_name.replace(' ', '_')}.png")
    plt.close()

    return {
        "cm": cm,
        "cm_norm": cm_norm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }


def compute_transition_matrices(
    df_assigned: pd.DataFrame,
    preds_df: pd.DataFrame,
    split_name: str,
    model_name: str,
    regime_col: str = "KMeans_Regime",
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Compute and save:
      - true transition matrix P(Regime_{t+1} | Regime_t)
      - predicted transition matrix P(pred_{t+1} | pred_t) for a given split/model
    """
    _ensure_dir(results_dir)

    # True transitions
    df_trans = df_assigned[[regime_col]].copy()
    df_trans["Regime_next"] = df_trans[regime_col].shift(-1)
    df_trans = df_trans.dropna(subset=[regime_col, "Regime_next"])

    true_trans = pd.crosstab(
        df_trans[regime_col],
        df_trans["Regime_next"],
        normalize="index",
    )

    true_trans.to_csv(results_dir / "transition_matrix_true.csv")

    # Predicted transitions
    mask = (preds_df["split"] == split_name) & (preds_df["model"] == model_name)
    sub = preds_df[mask].copy().sort_values("date")

    sub["Regime_pred_next"] = sub["y_pred"].shift(-1)
    sub = sub.dropna(subset=["y_pred", "Regime_pred_next"])

    pred_trans = pd.crosstab(
        sub["y_pred"],
        sub["Regime_pred_next"],
        normalize="index",
    )

    pred_trans.to_csv(
        results_dir / f"transition_matrix_pred_{model_name}_{split_name.replace(' ', '_')}.csv"
    )

    return {
        "true": true_trans,
        "pred": pred_trans,
    }


# ****************************************************************************
# 4. Strategy reporting
# ****************************************************************************

def summarize_strategy_performance(
    strategy_results: Dict[str, Dict[str, object]],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    strategy_results is the output of strategies.run_all_strategies:
      {
        "strat1": {"positions": Series, "returns": Series, "performance": dict},
        ...
        "buy_and_hold": {...}
      }

    This function:
      - builds a performance table for all strategies
      - saves it to CSV
    """
    _ensure_dir(results_dir)

    perf_rows = {}
    for name, res in strategy_results.items():
        perf_rows[name] = res["performance"]

    perf_df = pd.DataFrame.from_dict(perf_rows, orient="index")
    perf_df.to_csv(results_dir / "strategy_performance.csv")

    return perf_df


def plot_equity_curves(
    strategy_results: Dict[str, Dict[str, object]],
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Plot:
      - main equity curves: buy & hold + strat1 + strat2 + strat3
      - zoom on strat3 vs its sub-components 3A / 3B + buy & hold
    """
    _ensure_dir(results_dir)

    # Build equity curves
    eq_curves = {}
    for name, res in strategy_results.items():
        ret = res["returns"]
        eq_curves[name] = (1 + ret).cumprod()

    # Identify keys (defensive if some are missing)
    bh_key = "buy_and_hold"
    strat1 = "strat1"
    strat2 = "strat2"
    strat3 = "strat3"
    strat3A = "strat3A"
    strat3B = "strat3B"

    # 1) Main comparison
    plt.figure(figsize=(12, 6))
    if bh_key in eq_curves:
        plt.plot(eq_curves[bh_key], label="Buy & Hold", alpha=0.8)
    if strat1 in eq_curves:
        plt.plot(eq_curves[strat1], label="Strategy 1", alpha=0.9)
    if strat2 in eq_curves:
        plt.plot(eq_curves[strat2], label="Strategy 2", alpha=0.9)
    if strat3 in eq_curves:
        plt.plot(eq_curves[strat3], label="Strategy 3 (Combined)", linewidth=2)

    plt.title("Equity Curves – Strategies vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "equity_curves_main.png")
    plt.close()

    # 2) Strategy 3 decomposition
    plt.figure(figsize=(12, 5))
    if bh_key in eq_curves:
        plt.plot(eq_curves[bh_key], label="Buy & Hold", alpha=0.8)
    if strat3A in eq_curves:
        plt.plot(eq_curves[strat3A], label="Strategy 3A", alpha=0.9)
    if strat3B in eq_curves:
        plt.plot(eq_curves[strat3B], label="Strategy 3B", alpha=0.9)
    if strat3 in eq_curves:
        plt.plot(eq_curves[strat3], label="Strategy 3 (Combined)", linewidth=2)

    plt.title("Equity Curves – Strategy 3 Components")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "equity_curves_strat3_components.png")
    plt.close()


# ****************************************************************************
# 5. High-level orchestrator for main.py
# ****************************************************************************

def run_full_reporting(
    df_with_features: pd.DataFrame,
    df_assigned: pd.DataFrame,
    idx_features: pd.Index,
    pca,
    results_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    strategy_results: Dict[str, Dict[str, object]],
    split_for_diagnostics: str = "2010-20 to 2021-24",
    model_for_diagnostics: str = "LogisticRegression",
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    High-level function to be called from main.py.

    It assumes:
      - df_with_features: dataframe after feature_engineering.add_spx_vix_features
      - df_assigned     : df_with_features + KMeans_Regime column
      - idx_features    : index of rows used in clustering (from build_feature_matrix)
      - pca             : fitted PCA object (from core.pca_analysis)
      - results_df      : metrics output from modeling.run_splits
      - preds_df        : detailed predictions from modeling.run_splits
      - strategy_results: output from strategies.run_all_strategies

    It will create and save all the “useful” plots/tables.
    """
    _ensure_dir(results_dir)

    # Creation of a logs to follow the advancements of the reporting
    def log(msg: str):
        print(msg)
        sys.stdout.flush()

    # 1. PCA diagnostics
    log("\n[1/4] Summerising PCA results ...")
    plot_pca_variance(pca, results_dir=results_dir)
    save_pca_loadings(pca, feature_cols=core.DEFAULT_FEATURE_COLS, results_dir=results_dir)

    # 2. Regimes / clustering
    log("\n[2/4] Describing regimes and clustering (KMeans)...")
    plot_sp500_with_regimes(
        df_prices=df_assigned,
        idx_features=idx_features,
        regime_col="KMeans_Regime",
        results_dir=results_dir,
    )
    summarize_regimes(df_assigned, regime_col="KMeans_Regime", results_dir=results_dir)

    # 3. ML results
    log("\n[3/4] Summarising ML results & diagnostics...")
    summarize_ml_results(results_df, results_dir=results_dir)
    log(f"    → Confusion matrix & ROC for {model_for_diagnostics} – {split_for_diagnostics}")
    plot_confusion_matrix_and_roc(
        preds_df,
        split_name=split_for_diagnostics,
        model_name=model_for_diagnostics,
        results_dir=results_dir,
    )
    log("    → Transition matrices (true vs predicted regimes)")
    compute_transition_matrices(
        df_assigned,
        preds_df,
        split_name=split_for_diagnostics,
        model_name=model_for_diagnostics,
        regime_col="KMeans_Regime",
        results_dir=results_dir,
    )

    # 4. Strategies
    log("\n[4/4] Evaluating strategies and plotting equity curves...")
    summarize_strategy_performance(strategy_results, results_dir=results_dir)
    plot_equity_curves(strategy_results, results_dir=results_dir)

    log(f"\nAll reporting complete. Results written to: {results_dir.resolve()}")
