import pandas as pd

def prepare_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw concatenated SP500 + VIX dataframe from data_loader.download_raw_data
    and:
    - rename tickers to SP500 / VIX
    - flatten MultiIndex columns to SP500_Close, VIX_Close, etc.
    - sort index, drop obvious NaNs
    """
    df = df.copy()
    df = df.rename(columns={'^GSPC': 'SP500', '^VIX': 'VIX'})

    df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]

    df = df.sort_index()
    df = df.dropna()

    return df