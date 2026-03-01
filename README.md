# Pair Trading Automated System

A Python-based automated pair trading system that implements statistical arbitrage strategies using cointegration analysis and mean reversion.

## Features

- **Free Data Source**: Uses yfinance (NO API key required!)
- **Statistical Analysis**: Cointegration testing (Engle-Granger, ADF)
- **Technical Indicators**: Z-score, RSI, volume analysis
- **Advanced Filters**: Volume and RSI confirmation filters
- **Complete Backtesting**: Full backtesting engine with transaction costs
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, and more
- **Visualization**: Automated chart generation for analysis

## Trading Pairs

The system analyzes three cointegrated pairs:

1. **TRGP/EPD** - Energy Midstream
2. **STNG/HAFN** - Shipping Tankers
3. **SBLK/GNK** - Shipping Dry Bulk

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import yfinance; print('yfinance OK')"
python -c "import pandas; print('pandas OK')"
python -c "import statsmodels; print('statsmodels OK')"
```

## Usage

### Run Complete Backtest

```bash
python main.py
```

This will:
1. Fetch historical data for all pairs (2020-2024)
2. Test for cointegration
3. Run backtest with the configured strategy
4. Generate performance metrics
5. Create charts and save to `./results/plots/`

### Configuration

Edit configuration files to customize the strategy:

#### Pairs Configuration ([config/pairs_config.py](config/pairs_config.py))

```python
DATA_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'price_field': 'Adj Close',
}
```

#### Strategy Configuration ([config/strategy_config.py](config/strategy_config.py))

```python
STRATEGY_PARAMS = {
    'entry_threshold': 2.0,      # Enter when |z-score| > 2.0
    'exit_threshold': 0.5,       # Exit when |z-score| < 0.5
    'stop_loss_threshold': 3.0,  # Force exit when |z-score| > 3.0
    'zscore_window': 20,         # 20 days for z-score

    # Filters
    'use_volume_filter': True,
    'use_rsi_filter': True,
    'min_volume_threshold': 100000,
}

BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'commission_per_share': 0.005,
    'slippage_bps': 5,
}
```

## Strategy Logic

The pair trading strategy follows these steps:

### 1. Cointegration Testing
- Tests if two stocks are cointegrated using Engle-Granger test
- Calculates hedge ratio (β) via OLS regression
- Only trades pairs with p-value < 0.05

### 2. Spread Calculation
```
Spread = Price1 - β × Price2
Z-score = (Spread - Mean) / Std Dev
```

### 3. Signal Generation
- **Long Spread** when z-score < -2.0 (buy stock1, sell stock2)
- **Short Spread** when z-score > 2.0 (sell stock1, buy stock2)

### 4. Filters
- **Volume Filter**: Both stocks must have > 100K daily volume
- **RSI Filter**: Confirms overbought/oversold conditions

### 5. Exit Rules
- **Mean Reversion**: Exit when |z-score| < 0.5
- **Stop Loss**: Force exit when |z-score| > 3.0

## Output

### Performance Metrics
- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Number of Trades

### Generated Charts
Each pair gets 6 charts saved to `./results/plots/{PAIR_NAME}/`:

1. **equity_curve.png** - Portfolio value over time
2. **drawdown.png** - Underwater equity curve
3. **spread_zscore.png** - Spread, z-score, and trade markers
4. **returns_distribution.png** - Histogram of daily returns
5. **price_series.png** - Both stock prices
6. **performance_summary.png** - Key metrics visualization

## Project Structure

```
pair-trading/
├── config/
│   ├── pairs_config.py        # Trading pairs configuration
│   └── strategy_config.py     # Strategy parameters
├── src/
│   ├── data/
│   │   ├── data_fetcher.py    # yfinance data fetching
│   │   └── data_processor.py  # Data cleaning and alignment
│   ├── analysis/
│   │   ├── cointegration.py   # Cointegration tests
│   │   ├── spread.py          # Spread calculation
│   │   └── indicators.py      # Technical indicators
│   ├── strategy/
│   │   ├── base_strategy.py   # Abstract strategy base
│   │   ├── signal_generator.py # Signal filtering
│   │   └── pair_strategy.py   # Main strategy implementation
│   ├── backtest/
│   │   ├── position.py        # Position management
│   │   ├── portfolio.py       # Portfolio tracking
│   │   ├── metrics.py         # Performance metrics
│   │   └── backtester.py      # Backtesting engine
│   ├── visualization/
│   │   └── plotter.py         # Chart generation
│   └── utils/
│       └── logger.py          # Logging configuration
├── results/                   # Output directory
├── data/cache/               # Cached market data
├── main.py                   # Main entry point
├── requirements.txt
└── README.md
```

## Key Modules

### Data Module
- Fetches historical data from Yahoo Finance (FREE, no API key!)
- Caches data locally to avoid repeated downloads
- Handles missing data and alignment

### Analysis Module
- **Cointegration**: Tests statistical relationships between pairs
- **Spread**: Calculates and normalizes price spreads
- **Indicators**: Z-score, RSI, volume analysis

### Strategy Module
- Implements mean reversion pair trading logic
- Applies volume and technical indicator filters
- Generates entry/exit signals

### Backtest Module
- Simulates trading with realistic costs (commission + slippage)
- Tracks positions and portfolio state
- Calculates comprehensive performance metrics

### Visualization Module
- Generates publication-quality charts
- Visualizes spread behavior and trade signals
- Plots performance metrics

## Example Output

```
============================================================
PAIR TRADING BACKTEST SYSTEM
============================================================

Processing pair: TRGP_EPD
Fetching historical data...
Data fetched: 1258 rows from 2020-01-02 to 2024-12-31

Testing cointegration...
Cointegration test results:
  - P-value: 0.0234
  - Hedge ratio (β): 1.4523
  - Cointegrated: True

Running backtest...

BACKTEST RESULTS
Initial Capital: $100,000.00
Final Value: $115,234.56
Total Return: 15.23%
Sharpe Ratio: 1.34
Max Drawdown: -8.45%
Number of Trades: 42
Win Rate: 57.14%
```

## Customization

### Add New Pairs

Edit [config/pairs_config.py](config/pairs_config.py):

```python
TRADING_PAIRS.append({
    'name': 'XOM_CVX',
    'ticker1': 'XOM',
    'ticker2': 'CVX',
    'sector': 'Energy/Oil Majors'
})
```

### Adjust Strategy Parameters

Edit [config/strategy_config.py](config/strategy_config.py) to tune:
- Entry/exit thresholds
- Z-score calculation window
- Filter parameters
- Position sizing

## Performance Considerations

- **Backtest Period**: 2020-2024 (5 years)
- **Transaction Costs**: $0.005 per share + 5 bps slippage
- **Position Size**: 30% of capital per pair
- **Data Frequency**: Daily bars

## Limitations

- Uses daily data (no intraday trading)
- Does not account for:
  - Borrowing costs for short positions
  - Margin requirements
  - Overnight funding
- Historical cointegration may not persist in the future
- Past performance does not guarantee future results

## Contributing

Feel free to extend the system with:
- Additional technical indicators
- Machine learning for dynamic thresholds
- Real-time trading integration
- Portfolio optimization across multiple pairs

## License

MIT License - Feel free to use and modify for your own trading research.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves risk of loss. Always do your own research and consult with financial professionals before trading.
