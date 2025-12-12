
from pathlib import Path
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent # Obtaining the project root directory
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def ensure_directories():
    """Ensure that the necessary data directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_raw_data(start_date: str, end_date: str) -> pd.DataFrame:
    """ Download daily data for S&P 500 and VIX from Yahoo Finance
    and return a merged dataframe."""
    ensure_directories()
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=True)
    vix = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)

    # Concatenate dataframes
    df = pd.concat([sp500, vix], axis=1)

    return df

# We use Parquet because it is a better file format for storing data, being more efficient in terms of both
# storage space and speed when reading and writing data compared to csv.

def save_raw_data(df: pd.DataFrame, filename: str = "raw_data.parquet") -> Path:
    """Save raw data to Parquet."""
    ensure_directories()
    raw_file_path = RAW_DIR / filename
    df.to_parquet(raw_file_path)
    print(f"Raw data saved to {raw_file_path}")
    return raw_file_path


def load_raw_data(filename: str = "raw_data.parquet") -> pd.DataFrame:
    """Load raw data from Parquet."""
    raw_file_path = RAW_DIR / filename
    df = pd.read_parquet(raw_file_path)
    print(f"Raw data loaded from {raw_file_path}")
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "processed_data.parquet") -> Path:
    """Save processed data to Parquet."""
    ensure_directories()
    processed_file_path = PROCESSED_DIR / filename
    df.to_parquet(processed_file_path)
    print(f"Processed data saved to {processed_file_path}")
    return processed_file_path


def load_processed_data(filename: str = "processed_data.parquet") -> pd.DataFrame:
    """Load processed data from Parquet."""
    processed_file_path = PROCESSED_DIR / filename
    df = pd.read_parquet(processed_file_path)
    print(f"Processed data loaded from {processed_file_path}")
    return df