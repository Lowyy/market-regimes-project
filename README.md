
# Market Regimes and Strategy Performance

## Project Overview

This project investigates whether market regimes inferred from historical price and volatility dynamics can be used to condition and improve systematic trading strategies relative to a buy-and-hold benchmark on the S&P 500.

Using daily data from the S&P 500 and the VIX, a broad set of return, volatility, momentum, and risk-related features is constructed. These features are analysed in both a raw scaled space and a PCA-reduced space and used to identify latent market regimes through unsupervised learning methods, including K-Means and Gaussian Mixture Models. The resulting regimes are examined to assess whether they capture distinct market environments and contain information relevant for trading decisions.

Beyond regime identification, the project studies whether market regimes can be predicted one day ahead using supervised learning models trained on current market features. Logistic Regression, Random Forest, and Gradient Boosting models are evaluated using time-based walk-forward splits, and an ensemble approach is used to stabilise regime predictions.

Finally, several rule-based trading strategies are designed to adapt their exposure based on the prevailing or predicted regime. Their performance is compared to a buy-and-hold strategy using standard risk-adjusted metrics. The objective is not only to test simple strategies, but to assess whether regime information can be integrated into systematic decision-making in a way that leads to more robust performance across different market conditions.

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
   Market regimes are identified using K-Means and Gaussian Mixture Models. Both raw scaled features and PCA-reduced representations are evaluated, with clustering quality assessed using standard criteria. The final pipeline relies on the approach that provided the most stable and interpretable regimes, with full comparisons documented in the exploratory notebooks and report.

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
