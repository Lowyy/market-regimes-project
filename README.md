
# Market Regimes and Strategy Performance

## Project Overview

This project studies how changing market conditions influence the performance of simple quantitative trading strategies. The core idea is to identify market regimes from historical data using unsupervised learning, assess whether these regimes can be predicted one day ahead using supervised models, and evaluate how different strategies behave across regimes.

The analysis focuses on daily data from the S&P 500 and the VIX. Market regimes are extracted from a set of engineered return, volatility, and momentum features. These regimes are then used both directly and through predicted signals to drive rule-based trading strategies.

The project is structured as a reproducible pipeline, with exploratory analysis conducted in notebooks and the final logic implemented in modular source files.

---

## Repository Structure

```
FinalProject/
│
├── data/
│   ├── raw/            # downloaded raw market data (Parquet)
│   └── processed/      # processed and feature-engineered data
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_exploring_features.ipynb
│   ├── 03_exploring_clustering.ipynb
│   ├── 04_exploring_ML.ipynb
│   ├── 05_exploring_strats.ipynb
│   └── 06_exploration_tests_src.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── strategies.py
│   ├── reporting.py
│   └── clustering/
│       ├── core.py
│       └── sweeps.py
│
├── tests/              # unit tests for each pipeline component
│
├── results/            # generated figures and tables
│
├── main.py             # end-to-end execution script
├── README.md
├── PROPOSAL.md
├── AI_USAGE.md
└── requirements.txt
```

---

## Methodology

1. **Data collection**
   Daily S&P 500 and VIX data are downloaded from Yahoo Finance and stored locally.

2. **Feature engineering**
   A broad set of features is constructed, including returns, realized volatility, drawdowns, momentum indicators, volatility risk premium proxies, and interaction terms.

3. **Unsupervised learning (regimes)**
   Market regimes are identified using K-Means and Gaussian Mixture Models. Both raw scaled features and PCA-reduced representations are evaluated, with clustering quality assessed using standard criteria.

4. **Supervised learning (regime prediction)**
   Logistic Regression, Random Forest, and Gradient Boosting models are trained to predict tomorrow’s regime based on today’s features. Time-ordered train/test splits are used to avoid look-ahead bias.

5. **Strategy backtesting**
   Several rule-based strategies are tested, including a baseline regime strategy, a momentum strategy, and a more tuned combined strategy. Performance is evaluated using standard risk-adjusted metrics.

6. **Reporting**
   All key results (PCA diagnostics, regime visualization, ML performance, and strategy equity curves) are automatically saved to the `results/` directory.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Lowyy/market-regimes-project
cd market-regimes-project
```

### 2. Create a clean virtual environment

```bash
python -m venv new_venv
source new_venv/bin/activate        # macOS / Linux
new_venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project relies on standard scientific Python libraries. Parquet support requires `pyarrow`, which is included in the dependency list.

---

## Usage

### Run the full pipeline

From the project root:

```bash
python main.py
```

This will:

* download raw data and save it to `data/raw/`
* process and engineer features, then save it to `data/processed/`
* identify market regimes
* train and evaluate predictive models
* backtest strategies
* generate figures and tables in `results/`

Execution time depends on machine performance and may take several minutes.

---

## Results

All outputs produced by `main.py` are saved automatically in the `results/` directory, including:

* PCA variance plots and loadings
* regime visualizations on the S&P 500
* machine learning performance tables
* confusion matrices and ROC curves
* strategy performance summaries and equity curves

The `results/` folder is intentionally not tracked in version control and is regenerated locally. But parts of its results will be included in the report.

---

## Testing

Unit tests are provided for each major module. To run the test suite:

```bash
pytest tests/
```

Tests are designed to validate data loading, feature construction, clustering logic, modeling outputs, and strategy execution.

---

## Notes

* Exploratory notebooks are provided for transparency and intuition building.
* The final implementation logic resides entirely in `src/` and `main.py`.
* All modeling and backtesting respects time ordering to avoid data leakage.
