from copy import deepcopy
from typing import Dict, Tuple, List

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# 1. Base model definitions
# ============================================================

BASE_MODELS: Dict[str, object] = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            penalty="l2",
        )),
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),

    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            random_state=42,
        )),
    ]),
}


def get_models() -> Dict[str, object]:
    """Return FRESH copies of the base models."""
    return {name: deepcopy(model) for name, model in BASE_MODELS.items()}


# ============================================================
# 2. Evaluation
# ============================================================

def evaluate_binary(y_true, y_prob, y_pred) -> Dict[str, float]:
    """Return a dictionary of binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ============================================================
# 3. Helpers
# ============================================================

def make_time_mask(df_or_index, start: str, end: str) -> pd.Series:
    idx = df_or_index.index if isinstance(df_or_index, pd.DataFrame) else df_or_index
    return (idx >= start) & (idx <= end)


def build_supervised_dataset(
    df_assigned: pd.DataFrame,
    feature_cols: List[str],
    regime_col: str = "KMeans_Regime",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build supervised dataset:
        X_t = features at time t
        y_t = regime at time t+1
    """
    df = df_assigned.copy()
    df["target_regime"] = df[regime_col].shift(-1)

    supervised_cols = feature_cols + ["target_regime"]
    df_sup = df[supervised_cols].dropna()

    X_all = df_sup[feature_cols]
    y_all = df_sup["target_regime"].astype(int)

    return df_sup, X_all, y_all


def build_train_test_split(
    df_sup: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:

    mask_train = make_time_mask(df_sup, train_start, train_end)
    mask_test  = make_time_mask(df_sup, test_start, test_end)

    return (
        X_all[mask_train],
        X_all[mask_test],
        y_all[mask_train],
        y_all[mask_test],
        mask_train,
        mask_test,
    )

def build_majority_vote(preds_df: pd.DataFrame, split_name: str) -> pd.Series:
    """
    Hard-voting ensemble of model predictions for a given split.
    Returns a Series indexed by date with 0/1 regime labels.
    """
    df = preds_df[preds_df["split"] == split_name].copy()
    # shape: rows = dates * models, columns: ['split','model','date','y_true','y_pred','y_prob']
    pivot = df.pivot(index="date", columns="model", values="y_pred")
    vote = (pivot.sum(axis=1) >= 2).astype(int)
    return vote.sort_index()


# ============================================================
# 4. Train + Predict
# ============================================================

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_model(model, X_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


#********************* Not sure *****************************============================================================
# 5. Run full ML for one or multiple splits
# ============================================================

def run_splits(
    df_sup: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    splits: Dict[str, Tuple[str, str, str, str]],
    model_names: List[str] = None,
):
    """
    Run all models across all splits.

    splits dict format:
    {
        "split1": ("train_start", "train_end", "test_start", "test_end"),
        "split2": (...)
    }

    Returns:
        results_df : one row per model x split (metrics)
        preds_df   : full detailed predictions (for strategies)
    """

    all_results = []
    all_preds = []
    models = get_models()  # fresh models

    if model_names is None:
        model_names = list(models.keys())

    for split_name, (tr_start, tr_end, te_start, te_end) in splits.items():
        print(f"\n=== Running split: {split_name} ===")

        X_train, X_test, y_train, y_test, mask_tr, mask_te = build_train_test_split(
            df_sup, X_all, y_all,
            tr_start, tr_end,
            te_start, te_end,
        )

        test_index = df_sup.index[mask_te]

        # Loop over models
        for model_name in model_names:
            model = deepcopy(models[model_name])

            model = train_model(model, X_train, y_train)
            y_pred, y_prob = predict_model(model, X_test)

            # Metrics
            metrics = evaluate_binary(y_test, y_prob, y_pred)
            metrics.update({"split": split_name, "model": model_name})
            all_results.append(metrics)

            print(
                f"{model_name} → acc={metrics['accuracy']:.3f}, "
                f"bal_acc={metrics['balanced_accuracy']:.3f}, "
                f"f1={metrics['f1']:.3f}, auc={metrics['roc_auc']:.3f}"
            )

            # Detailed predictions
            df_pred = pd.DataFrame({
                "split": split_name,
                "model": model_name,
                "date": test_index,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
            })

            all_preds.append(df_pred)

    results_df = pd.DataFrame(all_results)
    preds_df = pd.concat(all_preds, ignore_index=True)

    return results_df, preds_df

