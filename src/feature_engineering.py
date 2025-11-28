import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from hurst import compute_Hc

DPY = 252  # trading days per year
MAR_ANNUAL = 0.042  # Minimum Acceptable Return (T-Bill)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear regression slope over `window` points."""
    return series.rolling(window).apply(
        lambda x: np.polyfit(range(window), x, 1)[0],
        raw=False
    )


def calculate_rsi(close: pd.Series, window: int = 20) -> pd.Series:
    """Relative Strength Index (RSI) over a rolling window."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def adf_test(series: pd.Series) -> float:
    """Return ADF test p-value for a series."""
    result = adfuller(series.dropna())
    return result[1]


def kpss_test(series: pd.Series) -> float:
    """Return KPSS test p-value for a series."""
    result = kpss(series.dropna(), nlags="auto")
    return result[1]


def hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    """Rolling Hurst exponent using compute_Hc with kind='change'."""
    def hurst_calc(x):
        H, c, data = compute_Hc(x, kind='change', simplified=True)
        return H

    return series.rolling(window=window).apply(hurst_calc, raw=False)


def calculate_adx(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index (ADX) over `window`."""
    high = data['SP500_High']
    low = data['SP500_Low']
    close = data['SP500_Close']

    plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0)
    minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx


# -------------------------------------------------------------------
# Main feature engineering function
# -------------------------------------------------------------------

def add_spx_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all SP500 + VIX based features to the dataframe.

    Assumes df has at least:
        SP500_Open, SP500_High, SP500_Low, SP500_Close
        VIX_Open,   VIX_High,   VIX_Low,   VIX_Close
    """
    df = df.copy()

    # ------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------
    close = df['SP500_Close']

    df['daily_return'] = close.pct_change()
    df['log_daily_return'] = np.log(close / close.shift(1))
    df['5d_return'] = close.pct_change(5)
    df['20d_return'] = close.pct_change(20)

    # ------------------------------------------------------------
    # Volatility measures
    # ------------------------------------------------------------
    df['5d_volatility'] = df['log_daily_return'].rolling(5).std()
    df['20d_volatility'] = df['log_daily_return'].rolling(20).std()
    df['60d_volatility'] = df['log_daily_return'].rolling(60).std()
    df['vol_of_vol_20d'] = df['20d_volatility'].rolling(20).std()

    df['realized_quarticity_20d'] = df['log_daily_return'].rolling(20).apply(
        lambda x: np.mean(x**4)
    )
    df['realized_skew_20d'] = df['log_daily_return'].rolling(20).skew()
    df['realized_kurtosis_20d'] = df['log_daily_return'].rolling(20).kurt()

    ret_mean_20d = df['log_daily_return'].rolling(20).mean()
    df['return_Zscore_20d'] = (df['log_daily_return'] - ret_mean_20d) / df['20d_volatility']

    # ------------------------------------------------------------
    # Moving averages and trend
    # ------------------------------------------------------------
    df['MA_20'] = close.rolling(20).mean()
    df['MA_50'] = close.rolling(50).mean()
    df['MA_200'] = close.rolling(200).mean()

    df['slope_MA_20'] = rolling_slope(df['MA_20'], 20)
    df['slope_MA_50'] = rolling_slope(df['MA_50'], 50)
    df['slope_MA_200'] = rolling_slope(df['MA_200'], 200)

    df['price_MA_20'] = close / df['MA_20']
    df['price_MA_50'] = close / df['MA_50']
    df['price_MA_200'] = close / df['MA_200']

    # ------------------------------------------------------------
    # Drawdown and autocorrelation
    # ------------------------------------------------------------
    df['rolling_max'] = close.rolling(252).max()
    df['current_drawdown'] = (close - df['rolling_max']) / df['rolling_max']
    df['max_drawdown_252d'] = df['current_drawdown'].rolling(252).min()

    MAR_daily = (1 + MAR_ANNUAL)**(1 / DPY) - 1
    df['downside_dev_20d'] = np.sqrt(
        (np.minimum(df['daily_return'] - MAR_daily, 0)**2).rolling(20).mean()
    )

    df['autocorr_1d'] = df['log_daily_return'].rolling(50).apply(lambda x: x.autocorr(1))
    df['autocorr_5d'] = df['log_daily_return'].rolling(100).apply(lambda x: x.autocorr(5))
    df['autocorr_20d'] = df['log_daily_return'].rolling(200).apply(lambda x: x.autocorr(20))

    # ------------------------------------------------------------
    # VIX related features
    # ------------------------------------------------------------
    df['dVIX_1d'] = np.log(df['VIX_Close']).diff()
    df['dVIX_5d'] = df['dVIX_1d'].rolling(5).sum()

    df['VIX_Zscore_20d'] = (
        df['dVIX_1d'] - df['dVIX_1d'].rolling(20).mean()
    ) / df['dVIX_1d'].rolling(20).std()

    df['VIX_vol_20d'] = df['dVIX_1d'].rolling(20).std()

    df['realized_vol_20d_ann'] = df['20d_volatility'] * np.sqrt(DPY)
    df['VRP_Proxy'] = (df['VIX_Close'] / 100)**2 - df['realized_vol_20d_ann']**2

    # ------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------
    df['corr_ret_dVIX_20d'] = df['log_daily_return'].rolling(20).corr(df['dVIX_1d'])
    df['ret_x_VIX'] = df['log_daily_return'] * df['dVIX_1d']
    df['vol_x_VIX'] = df['20d_volatility'] * df['dVIX_1d']
    df['vol_ratio'] = df['realized_vol_20d_ann'] / (df['VIX_Close'] / 100)
    df['VIX_Zscore_X_DD'] = df['VIX_Zscore_20d'] * df['current_drawdown']

    # ------------------------------------------------------------
    # RSI indicator
    # ------------------------------------------------------------
    df['RSI_20'] = calculate_rsi(close, window=20)

    # ------------------------------------------------------------
    # Hurst exponent feature
    # ------------------------------------------------------------
    df['hurst_100d'] = hurst_exponent(df['log_daily_return'], 100)

    # ------------------------------------------------------------
    # ADX feature
    # ------------------------------------------------------------
    df['ADX_14'] = calculate_adx(df, window=14)

    return df



