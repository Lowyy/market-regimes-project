import pandas as pd
import numpy as np


# ==========================================================================================
# 1. PERFORMANCE FUNCTION (Sharpe, Sortino, drawdown)
# ==========================================================================================

def perf_stats(returns: pd.Series, freq: int = 252):
    """Compute standard performance stats + Sortino ratio."""
    eq = (1 + returns).cumprod()

    mu = returns.mean() * freq
    sigma = returns.std() * np.sqrt(freq)

    downside = returns[returns < 0]
    downside_dev = (
        downside.std() * np.sqrt(freq)
        if len(downside) > 0 else np.nan
    )

    sharpe = mu / sigma if sigma > 0 else np.nan
    sortino = (
        mu / downside_dev
        if downside_dev and downside_dev > 0 else np.nan
    )

    max_dd = (eq / eq.cummax()).min() - 1

    return {
        "ann_return": mu,
        "ann_vol": sigma,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }

# ==========================================================================================
# 1B. EXPOSURE STATS (cash / invested / long / short)
# ==========================================================================================

def exposure_stats(positions: pd.Series) -> dict:
    """
    Compute share of days in cash / invested.
    positions are typically in {-1, 0, +1}.
    """
    p = positions.fillna(0.0)

    return {
        "cash_pct": float((p == 0).mean()),
        "invested_pct": float((p != 0).mean()),

    }

# ==========================================================================================
# 2. DATA ALIGNMENT
# ==========================================================================================

def align_data_for_strategies(df, pred_regime_today):
    """
    Align returns & features so strategies can use them cleanly.
    """
    idx = pred_regime_today.index

    data = {
        "ret": df.loc[idx, "daily_return"],
        "ret_next": df["daily_return"].shift(-1).loc[idx],
        "ret_5d": df.loc[idx, "5d_return"],

        # price / momentum features
        "ma20" : df.loc[idx, "MA_20"],
        "ma200": df.loc[idx, "MA_200"],
        "sma20": df.loc[idx, "slope_MA_20"],
        "sma50": df.loc[idx, "slope_MA_50"],
        "rsi": df.loc[idx, "RSI_20"],

        # volatility features
        "vix": df.loc[idx, "VIX_Close"],
        "vix_z": df.loc[idx, "VIX_Zscore_20d"],
        "vol_ratio": df.loc[idx, "vol_ratio"],

        # risk features
        "dd": df.loc[idx, "current_drawdown"],
        "vrp": df.loc[idx, "VRP_Proxy"],

        # regimes
        "reg0": (pred_regime_today == 0),
        "reg1": (pred_regime_today == 1)
    }

    return idx, data


# ==========================================================================================
# 3. STRATEGY DEFINITIONS
# ==========================================================================================

# Strategy 1 — baseline, long in regime 1
def strategy_1(idx, data):
    pos = pd.Series(0.0, index=idx)
    pos[data["reg0"]] = 1.0
    return pos


# Strategy 2 — momentum strategy
def strategy_2(idx, data):
    pos = pd.Series(0.0, index=idx)

    long_mom = data["reg0"] & (data["ma20"] > data["ma200"])
    short_mom = data["reg1"] & (data["sma20"] > 0)

    pos[long_mom] = 1.0
    pos[short_mom] = 1.0
    return pos


# Strategy 3A — regime 0 mean reversion
def strategy_3A(idx, data):
    pos = pd.Series(0.0, index=idx)

    long_3A = (
        (data["dd"] < -0.05) &
        (data["vrp"] > 0) &
        (data["vix_z"] < 0) &
        (data["vix"] > 30)
    )

    short_3A = (
        (data["vix_z"] < -1) &
        (data["vrp"] < 0)
    )

    pos[data["reg1"] & long_3A] = 1.0
    pos[data["reg1"] & short_3A] = -1.0
    return pos


# Strategy 3B — regime 1 filtered long
def strategy_3B(idx, data):
    pos = pd.Series(0.0, index=idx)

    filter_3B = (
        data["vix_z"].rolling(5).max() < 2
    )

    pos[data["reg0"] & filter_3B] = 1.0
    return pos


# Strategy 3 — combined A + B
def strategy_3(idx, data):
    return strategy_3A(idx, data) + strategy_3B(idx, data)


# ==========================================================================================
# 4. MAIN WRAPPER TO COMPUTE ALL STRATEGIES
# ==========================================================================================

def compute_strategy_positions(name, idx, data):
    """
    Returns a position series for a given strategy name.
    """
    mapping = {
        "strat1": strategy_1,
        "strat2": strategy_2,
        "strat3A": strategy_3A,
        "strat3B": strategy_3B,
        "strat3": strategy_3,
    }
    return mapping[name](idx, data)


def evaluate_strategy(positions, ret_next):
    """Compute performance dict for a strategy."""
    return perf_stats(positions * ret_next)


def run_all_strategies(pred_regime_today, df):
    """
    Computes positions + returns + performance for every strategy.
    Returns a dictionary with all results.
    """

    idx, data = align_data_for_strategies(df, pred_regime_today)

    results = {}
    strategies = ["strat1", "strat2", "strat3A", "strat3B", "strat3"]

    for strat in strategies:
        pos = compute_strategy_positions(strat, idx, data)
        ret = pos * data["ret_next"]

        results[strat] = {
            "positions": pos,
            "returns": ret,
            "performance": perf_stats(ret),
            "exposure": exposure_stats(pos),
        }

        # Buy & hold
    bh_positions = pd.Series(1.0, index=idx)
    bh_returns = data["ret_next"]

    results["buy_and_hold"] = {
        "positions": bh_positions,
        "returns": bh_returns,
        "performance": perf_stats(bh_returns),
        "exposure": exposure_stats(bh_positions),
    }

    return results
