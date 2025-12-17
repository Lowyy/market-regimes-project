import os
from pathlib import Path

import pandas as pd

from src import data_loader, processing, feature_engineering


def test_feature_engineering_after_parquet_roundtrip():
    """
    Full pipeline:

    1. Download raw data
    2. Prepare market data (rename/flatten)
    3. Save processed data as Parquet
    4. Load processed data back
    5. Run feature engineering
    6. Check that key features exist and look sane
    """

    # 1) Download a small sample
    df_raw = data_loader.download_raw_data(
        start_date="2018-01-01",
        end_date="2020-01-01",
    )

    assert isinstance(df_raw, pd.DataFrame)
    assert not df_raw.empty

    # 2) Prepare market data
    df_proc = processing.prepare_market_data(df_raw)

    assert "SP500_Close" in df_proc.columns
    assert "VIX_Close" in df_proc.columns

    # 3) Save processed to Parquet
    fname = "test_processed_feat.parquet"
    proc_path = data_loader.save_processed_data(df_proc, filename=fname)
    assert Path(proc_path).exists()

    # 4) Load processed again
    df_proc_loaded = data_loader.load_processed_data(filename=fname)
    assert isinstance(df_proc_loaded, pd.DataFrame)
    assert not df_proc_loaded.empty

    # 5) Add features on the LOADED data
    df_feat = feature_engineering.add_spx_vix_features(df_proc_loaded)

    assert isinstance(df_feat, pd.DataFrame)
    assert not df_feat.empty

    # 6) Check a few representative features
    expected_features = [
        # returns
        "daily_return",
        "log_daily_return",
        "5d_return",
        "20d_return",
        # volatility
        "5d_volatility",
        "20d_volatility",
        "60d_volatility",
        "vol_of_vol_20d",
        "realized_quarticity_20d",
        "realized_skew_20d",
        "realized_kurtosis_20d",
        "return_Zscore_20d",
        # trend / MAs
        "MA_20",
        "MA_50",
        "MA_200",
        "slope_MA_20",
        "slope_MA_50",
        "slope_MA_200",
        # drawdown & risk
        "current_drawdown",
        "max_drawdown_252d",
        "downside_dev_20d",
        # VIX / vol risk premium
        "dVIX_1d",
        "VIX_Zscore_20d",
        "realized_vol_20d_ann",
        "VRP_Proxy",
        # interactions / tech indicators
        "vol_ratio",
        "RSI_20",
        "hurst_100d",
        "ADX_14",
    ]

    for col in expected_features:
        assert col in df_feat.columns, f"Missing feature column: {col}"

    # Cleanup test file
    try:
        os.remove(proc_path)
    except Exception:
        pass

    print("\nfeature_engineering after parquet roundtrip test passed")


if __name__ == "__main__":
    test_feature_engineering_after_parquet_roundtrip()
