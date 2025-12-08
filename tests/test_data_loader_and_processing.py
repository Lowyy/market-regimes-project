import os
from pathlib import Path
import pandas as pd

from src import data_loader, processing


def test_full_data_loader_pipeline():
    """
    Test the entire data loader workflow:

    1. Download raw data
    2. Save raw data → load raw data
    3. Process data
    4. Save processed → load processed
    """

    # 1. Download raw SPX + VIX data
    df_raw = data_loader.download_raw_data(
        start_date="2020-01-01",
        end_date="2020-03-01",
    )

    assert isinstance(df_raw, pd.DataFrame)
    assert not df_raw.empty
    assert "^GSPC" in df_raw.columns[0]  or "^VIX" in df_raw.columns[-1]

    # 2. Save raw data and load it back
    raw_path = data_loader.save_raw_data(df_raw, filename="test_raw.parquet")
    assert Path(raw_path).exists()

    df_raw_loaded = data_loader.load_raw_data(filename="test_raw.parquet")
    assert isinstance(df_raw_loaded, pd.DataFrame)
    assert not df_raw_loaded.empty

    # 3. Process data (renaming, flattening, drop NaN)
    df_proc = processing.prepare_market_data(df_raw_loaded)

    assert isinstance(df_proc, pd.DataFrame)
    assert not df_proc.empty
    assert "SP500_Close" in df_proc.columns
    assert "VIX_Close" in df_proc.columns
    assert "SP500_Open" in df_proc.columns

    # 4. Save processed and load again
    proc_path = data_loader.save_processed_data(df_proc, filename="test_processed.parquet")
    assert Path(proc_path).exists()

    df_proc_loaded = data_loader.load_processed_data(filename="test_processed.parquet")

    assert isinstance(df_proc_loaded, pd.DataFrame)
    assert not df_proc_loaded.empty
    assert "SP500_Close" in df_proc_loaded.columns

    # ------------------------------------------------------------
    # Cleanup test files
    # ------------------------------------------------------------
    try:
        os.remove(raw_path)
        os.remove(proc_path)
    except Exception:
        pass

    print("\n✓ data_loader pipeline test passed")

if __name__ == "__main__":
    test_full_data_loader_pipeline()
