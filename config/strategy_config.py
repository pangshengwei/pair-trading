"""
Strategy Configuration

Defines all parameters for the pair trading strategy including
statistical thresholds, technical indicators, and backtest settings.
"""

STRATEGY_PARAMS = {
    # Statistical Parameters
    'cointegration_window': 252,      # 1 year for cointegration test
    'hedge_ratio_window': 60,         # 60 days rolling window
    'zscore_window': 20,              # 20 days for z-score
    'entry_threshold': 2.0,           # Enter when |z-score| > 2.0
    'exit_threshold': 0.5,            # Exit when |z-score| < 0.5
    'stop_loss_threshold': 3.0,       # Force exit when |z-score| > 3.0
    'position_size': 0.3,             # 30% capital per pair

    # Volume Filter Parameters
    'min_volume_threshold': 100000,   # Minimum daily volume per stock
    'volume_ma_window': 20,           # Volume moving average window
    'abnormal_volume_multiplier': 2.0, # Flag volume > 2x average

    # Technical Indicator Parameters
    'rsi_window': 14,                 # RSI calculation window
    'rsi_overbought': 70,             # RSI overbought threshold
    'rsi_oversold': 30,               # RSI oversold threshold
    'sma_window': 50,                 # Simple MA for trend detection
    'bollinger_window': 20,           # Bollinger Bands window
    'bollinger_std': 2.0,             # Bollinger Bands std deviation

    # Filter Flags
    'use_volume_filter': True,        # Enable volume filtering
    'use_rsi_filter': True,           # Enable RSI confirmation
    'use_trend_filter': False,        # Enable trend filter (optional)
}

BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'commission_per_share': 0.005,    # $0.005 per share
    'slippage_bps': 5,                # 5 basis points
}
