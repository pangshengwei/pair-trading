# Pair Trading System - Session Notes

## Overview

This document summarizes the full development session for the pair-trading backtesting system, including design decisions, technical explanations, key concepts, and lessons learned.

---

## 1. Project Background

Built a complete Python-based automated pair trading workflow from scratch. The system:
- Fetches historical stock data from **yfinance** (free, no API key required)
- Runs statistical cointegration tests to identify tradeable pairs
- Generates technical indicators (z-score, RSI, Bollinger Bands, volume filters)
- Simulates a backtest with realistic transaction costs
- Produces 6 professional charts per pair

### Trading Pairs

| Pair Name | Ticker 1 | Ticker 2 | Sector |
|-----------|----------|----------|--------|
| TRGP_EPD | TRGP | EPD | Energy / Midstream |
| STNG_HAFN | STNG | HAFN | Shipping / Tankers |
| SBLK_GNK | SBLK | GNK | Shipping / Dry Bulk |

---

## 2. Project Architecture

```
pair-trading/
├── config/
│   ├── pairs_config.py          # Ticker pairs, date range, cache settings
│   └── strategy_config.py       # All strategy parameters (thresholds, windows)
├── src/
│   ├── data/
│   │   └── data_fetcher.py      # yfinance data fetching + Parquet caching
│   ├── analysis/
│   │   ├── cointegration.py     # Engle-Granger test, ADF test, OLS regression
│   │   ├── spread.py            # Spread calculation
│   │   └── indicators.py        # Z-score, RSI, Bollinger Bands, volume analysis
│   ├── strategy/
│   │   ├── base_strategy.py     # Abstract base class
│   │   ├── signal_generator.py  # Entry/exit signal logic
│   │   └── pair_strategy.py     # Main strategy: filter stack + signal generation
│   ├── backtest/
│   │   ├── position.py          # Position tracking (open/close)
│   │   ├── portfolio.py         # Cash, equity curve, trade execution
│   │   ├── metrics.py           # Sharpe, Sortino, Max Drawdown, Win Rate
│   │   └── backtester.py        # Main engine: day-by-day simulation
│   └── visualization/
│       └── plotter.py           # 6 charts per pair
├── results/plots/               # Output charts organized by pair
├── data/cache/                  # Parquet cached data
├── main.py                      # Entry point
└── requirements.txt
```

### 8-Step Execution Workflow

```
Step 1: Load config (pairs_config.py, strategy_config.py)
          |
Step 2: Fetch data via yfinance → cache as Parquet
          |
Step 3: Cointegration test (Engle-Granger + ADF)
          |
Step 4: Calculate hedge ratio (OLS regression) + spread
          |
Step 5: Generate indicators (z-score, RSI, volume, Bollinger)
          |
Step 6: Generate signals (z-score threshold → filter stack)
          |
Step 7: Run backtest (Portfolio simulation with commission + slippage)
          |
Step 8: Generate charts (6 plots per pair) → results/plots/
```

---

## 3. Data Storage: Parquet vs CSV

### Why Parquet?

| Feature | Parquet | CSV |
|---------|---------|-----|
| File size | 5-10x smaller (compressed) | Large |
| Read speed | Fast (columnar format) | Slow on large files |
| Data types | Preserved (int, float, datetime) | Everything is string |
| Partial reads | Read specific columns only | Must load full file |
| Tooling | Python, DuckDB, Spark | Excel, any text editor |

### How to View Parquet Files

```python
# Option 1: pandas
import pandas as pd
df = pd.read_parquet('./data/cache/TRGP_2020-01-01_2024-12-31.parquet')
print(df.head(10))
df.to_csv('temp.csv')    # Convert to CSV for Excel

# Option 2: Excel export
df.to_excel('output.xlsx', index=True)

# Option 3: Quick CLI viewer
# python view_cache.py ./data/cache/TRGP_2020-01-01_2024-12-31.parquet
```

---

## 4. Statistical Concepts

### 4a. Augmented Dickey-Fuller (ADF) Test

The ADF test checks whether a time series is **stationary** (mean and variance constant over time).

- **Null hypothesis**: Series has a unit root (non-stationary, random walk)
- **Alternative hypothesis**: Series is stationary (mean-reverting)
- **Decision rule**: If `p-value < 0.05` → reject null → series IS stationary

This is applied to the **spread** (residuals), not the raw prices.

### 4b. Cointegration (Engle-Granger Method)

Two non-stationary series are **cointegrated** if their linear combination is stationary.

```
Steps:
1. Regress: stock1 = β × stock2 + α + ε
2. Extract residuals (spread): ε = stock1 - β × stock2 - α
3. Run ADF test on residuals
4. If p < 0.05 → residuals are stationary → COINTEGRATED
```

**What cointegration means in practice:**
- Both stocks may individually wander (trend up or down)
- But their **spread** oscillates around a stable mean
- When the spread deviates, it tends to revert → tradeable signal
- Like two people on a rope: each wanders, but the rope keeps them tethered

**Important clarification:**
- Cointegration is NOT about low residuals — it's about **stationary** residuals
- The spread can be large in absolute value and still be stationary

### 4c. Hedge Ratio

The hedge ratio β is the **slope (gradient)** of the regression line:

```
Stock1 = β × Stock2 + α
```

- β tells you how many units of Stock2 to trade per unit of Stock1
- Makes the position **market-neutral** (removes common market exposure)
- Only profits from spread convergence, not market direction

**Example:**
- β = 7.99 for TRGP/EPD
- Buy $10,000 TRGP → must also sell $79,900 EPD
- This is why TRGP/EPD couldn't be traded with $100K capital!

### 4d. Spread = Regression Residual

```
Spread = Stock1 - β × Stock2 - α
       = Actual Stock1 - Predicted Stock1
       = OLS Regression residual
```

The spread captures how far the pair has deviated from its long-run equilibrium.

### 4e. Z-Score of Spread

```
z = (spread - rolling_mean) / rolling_std
```

Normalizes the spread relative to recent history:
- `z > +2.0` → Spread is abnormally high → SHORT the spread
- `z < -2.0` → Spread is abnormally low → LONG the spread
- `|z| < 0.5` → Spread is near mean → EXIT position

---

## 5. Trading Signals Explained

### What "Short the Spread" Means

**Condition:** Z-score > +2.0 (spread is too high, Stock1 overvalued vs Stock2)

```
Action:
  SELL Stock1 (it's too expensive)
  BUY  Stock2 (it's relatively cheap)

Expected outcome:
  Spread narrows (Stock1 falls OR Stock2 rises)
  Close positions when z-score < 0.5
```

**Numerical example:**
```
TRGP = $120, EPD = $15, β = 8.0
Spread = 120 - 8.0 × 15 = $0 (normal)

Next day: TRGP = $125, EPD = $15
Spread = 125 - 8.0 × 15 = $5 (z-score = +2.5 → too high!)

Trade: SELL 83 TRGP @ $120, BUY 667 EPD @ $15

Three days later: TRGP = $119, EPD = $15.10
Close: BUY 83 TRGP @ $119, SELL 667 EPD @ $15.10

Profit:
  TRGP short: 83 × ($120 - $119) = +$83
  EPD long:   667 × ($15.10 - $15) = +$67
  Total: +$150
```

### What "Long the Spread" Means

**Condition:** Z-score < -2.0 (spread is too low, Stock1 undervalued vs Stock2)

```
Action:
  BUY  Stock1 (it's too cheap)
  SELL Stock2 (it's relatively expensive)

Expected outcome:
  Spread widens (Stock1 rises OR Stock2 falls)
  Close positions when z-score > -0.5
```

**Intuition (rubber band analogy):**
- Stretch the rubber band too far (+2 sigma) → It snaps back → Short
- Compress the rubber band (-2 sigma) → It expands → Long

---

## 6. Indicators Used

### Z-Score (Primary Signal)
```python
rolling_mean = spread.rolling(20).mean()
rolling_std  = spread.rolling(20).std()
zscore = (spread - rolling_mean) / rolling_std
```

### RSI (Confirmation Filter)
```python
delta = prices.diff()
gains = delta.where(delta > 0, 0)
losses = -delta.where(delta < 0, 0)
rs = gains.rolling(14).mean() / losses.rolling(14).mean()
rsi = 100 - (100 / (1 + rs))
```
- RSI > 70 → overbought (don't go long)
- RSI < 30 → oversold (don't go short)

### Volume Filter
- Requires minimum daily volume > 100,000 shares
- Detects abnormal volume (> 2x 20-day MA) as confirmation

### Bollinger Bands
- 20-day MA ± 2 standard deviations
- Additional context for spread positioning

---

## 7. Strategy Parameters (config/strategy_config.py)

```python
STRATEGY_PARAMS = {
    # Cointegration
    'cointegration_window': 252,       # 1 year lookback for cointegration test
    'hedge_ratio_window': 60,          # 60-day rolling hedge ratio
    'zscore_window': 20,               # 20-day rolling z-score

    # Signal thresholds
    'entry_threshold': 2.0,            # Enter when |z| > 2.0
    'exit_threshold': 0.5,             # Exit when |z| < 0.5
    'stop_loss_threshold': 3.0,        # Force exit when |z| > 3.0

    # Position sizing
    'position_size': 0.3,              # Use 30% of capital per trade

    # Volume filters
    'min_volume_threshold': 100000,    # Min daily volume
    'volume_ma_window': 20,
    'abnormal_volume_multiplier': 2.0,

    # Technical indicators
    'rsi_window': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'sma_window': 50,
    'bollinger_window': 20,
    'bollinger_std': 2.0,

    # Feature toggles
    'use_volume_filter': True,
    'use_rsi_filter': True,
    'use_trend_filter': False,
}
```

---

## 8. Backtest Configuration

```python
BACKTEST_PARAMS = {
    'initial_capital': 100000,         # $100,000 starting capital
    'commission_per_share': 0.005,     # $0.005/share commission
    'slippage_bps': 5,                 # 5 basis points slippage
}
```

### How to Run

```bash
# Run full backtest (all pairs)
python main.py

# View cached Parquet data
python -c "import pandas as pd; df = pd.read_parquet('./data/cache/TRGP_2020-01-01_2024-12-31.parquet'); print(df.head())"
```

---

## 9. Chart Interpretation Guide

Each pair generates 6 charts in `results/plots/<PAIR_NAME>/`:

### 1. equity_curve.png
- Shows portfolio value over time
- **Good:** Steadily rising, step-like increases when trades close
- **Bad:** Flat (no trades) or declining

### 2. drawdown.png
- Shows peak-to-trough portfolio decline
- **Good:** Max drawdown < 10%, quick recovery
- **Bad:** Deep troughs (> 20%), prolonged recovery

### 3. spread_zscore.png (2 panels)
- Top: Raw spread over time
  - **Good:** Oscillates around 0 (stationary)
  - **Bad:** Trends upward/downward (non-stationary, not cointegrated)
- Bottom: Z-score with trade markers
  - Green ↑ = Entry signals
  - Red ↓ = Exit signals
  - Dashed red lines = Entry threshold (±2.0)
  - Dashed green lines = Exit threshold (±0.5)

### 4. returns_distribution.png
- Histogram of daily returns
- **Good:** Slightly positive skew, fat tails acceptable
- **Bad:** Strongly negative skew, large negative outliers

### 5. price_series.png
- Raw price of both stocks over time
- **Good:** Both stocks move together, similar growth rates
- **Bad:** One stock dramatically outperforms (diverging trajectories)

### 6. performance_summary.png
- 4-panel summary: Total Return, Sharpe Ratio, Max Drawdown, Win Rate
- **Good benchmarks:** Sharpe > 1.0, Win Rate > 50%, Return > 0%

---

## 10. Backtest Results Summary

### TRGP_EPD ❌ (Not Cointegrated)

| Metric | Value |
|--------|-------|
| Trades | 0 |
| Return | 0% |
| Equity curve | Flat |
| Root cause | TRGP grew 6.7x ($30→$200); EPD grew 2x ($15→$30). Diverging trajectories. |

**Why not cointegrated:**
- TRGP: Natural gas pipelines, benefited from fracking boom (explosive growth)
- EPD: Mature midstream infrastructure (slow, dividend-focused growth)
- Same sector name ("Energy Midstream") does NOT guarantee cointegration
- Spread trended upward → not stationary → ADF test failed

### STNG_HAFN ❌ (Data Quality Issues)

| Metric | Value |
|--------|-------|
| Trades | 0 |
| Return | 0% |
| Root cause | HAFN had 85.4% missing values; cointegration test failed with NaN errors |

### SBLK_GNK ✅ (Cointegrated and Profitable!)

| Metric | Value |
|--------|-------|
| Cointegration p-value | 0.0195 (< 0.05) ✅ |
| Trades | Multiple |
| Return | ~20% ($100K → $120K) |
| Equity curve | Step-like growth from 2020 to 2023 |

**Why cointegrated:**
- Both are dry bulk shippers (same business model, same commodity cycle)
- Both exposed to iron ore and coal shipping rates
- When shipping rates rise, BOTH stocks rise and fall together
- Spread oscillates around mean → stationary → mean-reverting

---

## 11. Bugs Encountered and Fixed

### Bug 1: pandas-ta installation failure
- **Error:** `Could not find a version that satisfies the requirement pandas-ta>=0.3.14b`
- **Fix:** Removed pandas-ta from requirements.txt; re-implemented all indicators (RSI, Bollinger, etc.) from scratch using numpy/pandas

### Bug 2: yfinance MultiIndex columns
- **Error:** `KeyError: 'Adj Close'` (data returned as MultiIndex DataFrame)
- **Fix:** Used `.squeeze()` to convert 2D column to 1D Series
```python
adj_close1 = data1['Adj Close'].squeeze()
volume1 = data1['Volume'].squeeze()
```

### Bug 3: DataFrame construction without index
- **Error:** `ValueError: If using all scalar values, you must pass an index`
- **Fix:** Added `index=data1.index` when constructing DataFrame

### Bug 4: Insufficient capital for TRGP_EPD
- **Warning:** `Insufficient capital: need $134693.96, have $100000.00`
- **Cause:** Hedge ratio of 7.99 required ~$80K EPD per $10K TRGP
- **Status:** Expected behavior; pair correctly skipped

---

## 12. Other Cointegration Techniques (For Future Reference)

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **Kalman Filter** | Dynamic, adaptive hedge ratio | Faster adaptation to regime changes |
| **Johansen Test** | Multi-asset cointegration | Finding cointegrated portfolios of 3+ stocks |
| **Distance Method** | Euclidean distance between normalized prices | Simpler, no statistical assumptions |
| **Copula Models** | Models joint distribution of returns | Captures non-linear dependencies |
| **LSTM/Transformer** | ML-based spread prediction | Handles non-linear patterns |
| **ETF Arbitrage** | Trade ETF vs constituent basket | Index tracking errors |
| **Cross-sectional Mean Reversion** | Stocks vs sector average | Broader universe, more signals |

---

## 13. Key Lessons Learned

1. **Same sector ≠ cointegrated.** Always test statistically. TRGP and EPD are both "energy midstream" but not cointegrated because they serve different sub-markets.

2. **The flat equity curve is correct behavior.** If a pair fails cointegration, the system should produce zero trades, not forced trades.

3. **Hedge ratio determines capital requirements.** A high β (e.g., 7.99) means you need far more capital for the leg-2 position. Always check capital adequacy.

4. **Data quality matters.** HAFN had 85% missing data, making it untradeable regardless of statistical tests.

5. **yfinance is free and reliable**, but its API returns MultiIndex DataFrames when downloading. Use `.squeeze()` or handle this explicitly.

6. **Parquet > CSV** for financial time series: smaller, faster, type-preserving. Convert to CSV only when needed for Excel or human inspection.

7. **Stationary spread ≠ small spread.** The ADF test cares about mean-reversion behavior, not the magnitude of the spread.
