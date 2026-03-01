"""
Signal Generator Module

Generates and filters trading signals based on multiple criteria.
"""

import numpy as np
import pandas as pd
from loguru import logger


def generate_entry_signals(zscore, threshold=2.0):
    """
    Generate entry signals based on z-score threshold.

    Args:
        zscore: Z-score series
        threshold: Entry threshold (default: 2.0)

    Returns:
        pandas Series with signals:
        - 1 for long spread (zscore < -threshold)
        - -1 for short spread (zscore > threshold)
        - 0 for no signal
    """
    signals = pd.Series(0, index=zscore.index)

    # Long spread when zscore is very negative (spread undervalued)
    signals[zscore < -threshold] = 1

    # Short spread when zscore is very positive (spread overvalued)
    signals[zscore > threshold] = -1

    num_long = (signals == 1).sum()
    num_short = (signals == -1).sum()
    logger.debug(f"Generated {num_long} long and {num_short} short entry signals")

    return signals


def generate_exit_signals(zscore, threshold=0.5, current_position=0):
    """
    Generate exit signals when z-score reverts to mean.

    Args:
        zscore: Z-score series
        threshold: Exit threshold (default: 0.5)
        current_position: Current position (1=long, -1=short, 0=flat)

    Returns:
        pandas Series with boolean exit flags
    """
    # Exit when |zscore| crosses below threshold (mean reversion)
    exit_signals = abs(zscore) < threshold

    # Also exit on stop loss (z-score diverges further)
    # This is handled separately in pair_strategy

    return exit_signals


def apply_volume_filter(signals, volume1, volume2, min_volume=100000):
    """
    Filter out signals where volume is below minimum threshold.

    Args:
        signals: Original signal series
        volume1: Volume for first stock
        volume2: Volume for second stock
        min_volume: Minimum volume threshold

    Returns:
        Filtered signal series
    """
    # Check if both stocks have sufficient volume
    sufficient_volume = (volume1 >= min_volume) & (volume2 >= min_volume)

    # Keep only signals with sufficient volume
    filtered_signals = signals.copy()
    filtered_signals[~sufficient_volume] = 0

    num_filtered = (signals != 0).sum() - (filtered_signals != 0).sum()
    logger.info(f"Volume filter removed {num_filtered} signals")

    return filtered_signals


def apply_rsi_filter(signals, rsi1, rsi2, overbought=70, oversold=30):
    """
    Filter signals based on RSI confirmation.

    For long spread signals: Check if stock1 is oversold OR stock2 is overbought
    For short spread signals: Check if stock1 is overbought OR stock2 is oversold

    Args:
        signals: Original signal series
        rsi1: RSI for first stock
        rsi2: RSI for second stock
        overbought: RSI overbought threshold (default: 70)
        oversold: RSI oversold threshold (default: 30)

    Returns:
        Filtered signal series
    """
    filtered_signals = signals.copy()

    # For long spread signals (buy stock1, sell stock2)
    # Confirm if stock1 is oversold or stock2 is overbought
    long_signals = signals == 1
    long_confirmed = (rsi1 < oversold) | (rsi2 > overbought)
    filtered_signals[long_signals & ~long_confirmed] = 0

    # For short spread signals (sell stock1, buy stock2)
    # Confirm if stock1 is overbought or stock2 is oversold
    short_signals = signals == -1
    short_confirmed = (rsi1 > overbought) | (rsi2 < oversold)
    filtered_signals[short_signals & ~short_confirmed] = 0

    num_filtered = (signals != 0).sum() - (filtered_signals != 0).sum()
    logger.info(f"RSI filter removed {num_filtered} signals")

    return filtered_signals


def apply_trend_filter(signals, prices1, prices2, ma_window=50):
    """
    Filter signals against strong trends.

    Avoid trading when either stock is in a strong trend (price far from MA).

    Args:
        signals: Original signal series
        prices1: Price series for first stock
        prices2: Price series for second stock
        ma_window: Moving average window for trend detection

    Returns:
        Filtered signal series
    """
    # Calculate moving averages
    ma1 = prices1.rolling(window=ma_window).mean()
    ma2 = prices2.rolling(window=ma_window).mean()

    # Check if prices are close to MA (within 5%)
    near_ma1 = abs(prices1 - ma1) / ma1 < 0.05
    near_ma2 = abs(prices2 - ma2) / ma2 < 0.05

    # Keep signals only when both stocks are near their MAs
    filtered_signals = signals.copy()
    filtered_signals[~(near_ma1 & near_ma2)] = 0

    num_filtered = (signals != 0).sum() - (filtered_signals != 0).sum()
    logger.info(f"Trend filter removed {num_filtered} signals")

    return filtered_signals


def combine_filters(signals, filters_dict):
    """
    Apply multiple filters to signals.

    Args:
        signals: Original signal series
        filters_dict: Dict with filter configurations
            Example: {
                'volume': {'volume1': ..., 'volume2': ..., 'min_volume': 100000},
                'rsi': {'rsi1': ..., 'rsi2': ..., 'overbought': 70, 'oversold': 30},
                'trend': {'prices1': ..., 'prices2': ..., 'ma_window': 50}
            }

    Returns:
        Filtered signal series
    """
    filtered_signals = signals.copy()

    # Apply volume filter
    if 'volume' in filters_dict:
        vol_config = filters_dict['volume']
        filtered_signals = apply_volume_filter(
            filtered_signals,
            vol_config['volume1'],
            vol_config['volume2'],
            vol_config.get('min_volume', 100000)
        )

    # Apply RSI filter
    if 'rsi' in filters_dict:
        rsi_config = filters_dict['rsi']
        filtered_signals = apply_rsi_filter(
            filtered_signals,
            rsi_config['rsi1'],
            rsi_config['rsi2'],
            rsi_config.get('overbought', 70),
            rsi_config.get('oversold', 30)
        )

    # Apply trend filter
    if 'trend' in filters_dict:
        trend_config = filters_dict['trend']
        filtered_signals = apply_trend_filter(
            filtered_signals,
            trend_config['prices1'],
            trend_config['prices2'],
            trend_config.get('ma_window', 50)
        )

    total_filtered = (signals != 0).sum() - (filtered_signals != 0).sum()
    logger.info(f"Total signals filtered: {total_filtered}")

    return filtered_signals
