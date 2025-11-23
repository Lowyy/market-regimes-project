import pandas as pd
import numpy as np

from pathlib import Path
from src.data_loader import RAW_DIR, PROCESSED_DIR, load_raw_data

#****************************************************************
# 1. Basic Cleaning
#****************************************************************

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - Keep only useful columns (Open, Close)
    - Rename to clean names: SP500, VIX
    """
    df = df.rename(columns=({'^GSPC': 'SP500','^VIX': 'VIX'}))
    df = df.drop(['High', 'Low', 'Volume'], axis=1, level=0)
    df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    df_clean = df.dropna()

    return df_clean

#****************************************************************
# 2. Feature Engineering
#****************************************************************




