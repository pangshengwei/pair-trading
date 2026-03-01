"""
Technical Indicators Module

Implements z-score, volume indicators, and technical indicators (RSI, MA, etc.).
"""

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# Statistical Indicators
# ============================================================================

def calculate_zscore(series, window=20):
    """
    Calculate z-score of a series.

    Z-score = (value - rolling_mean) / rolling_std

    Args:
        series: Input series (pandas Series)
        window: Rolling window size

    Returns:
        pandas Series of z-scores
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    zscore = (series - rolling_mean) / rolling_std

    logger.debug(f"Calculated z-score (window={window})")
    return zscore


def rolling_mean(series, window):
    """Calculate rolling mean."""
    return series.rolling(window=window).mean()


def rolling_std(series, window):
    """Calculate rolling standard deviation."""
    return series.rolling(window=window).std()


# ============================================================================
# Volume Indicators
# ============================================================================

def calculate_volume_ratio(volume1, volume2):
    """
    Calculate ratio of trading volumes between two stocks.

    Args:
        volume1: Volume series for first stock
        volume2: Volume series for second stock

    Returns:
        pandas Series of volume ratios
    """
    volume_ratio = volume1 / volume2
    return volume_ratio


def calculate_average_volume(volume, window=20):
    """
    Calculate rolling average volume.

    Args:
        volume: Volume series
        window: Rolling window size

    Returns:
        pandas Series of average volume
    """
    avg_volume = volume.rolling(window=window).mean()
    return avg_volume


def detect_abnormal_volume(volume, threshold=2.0, window=20):
    """
    Flag volume spikes that exceed threshold times the average.

    Args:
        volume: Volume series
        threshold: Multiplier threshold (e.g., 2.0 for 2x average)
        window: Window for average calculation

    Returns:
        pandas Series of boolean flags (True = abnormal)
    """
    avg_volume = calculate_average_volume(volume, window)
    abnormal = volume > (threshold * avg_volume)

    num_abnormal = abnormal.sum()
    logger.debug(f"Detected {num_abnormal} abnormal volume days (>{threshold}x average)")

    return abnormal


def check_liquidity(volume, min_threshold=100000):
    """
    Check if volume meets minimum liquidity threshold.

    Args:
        volume: Volume series
        min_threshold: Minimum daily volume threshold

    Returns:
        pandas Series of boolean flags (True = meets threshold)
    """
    sufficient_liquidity = volume >= min_threshold

    pct_sufficient = sufficient_liquidity.mean() * 100
    logger.debug(f"{pct_sufficient:.1f}% of days meet liquidity threshold (>{min_threshold})")

    return sufficient_liquidity


# ============================================================================
# Price-Based Technical Indicators
# ============================================================================

def calculate_sma(prices, window=20):
    """
    Calculate Simple Moving Average.

    Args:
        prices: Price series
        window: Window size

    Returns:
        pandas Series of SMA values
    """
    return prices.rolling(window=window).mean()


def calculate_ema(prices, window=20):
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Price series
        window: Window size

    Returns:
        pandas Series of EMA values
    """
    return prices.ewm(span=window, adjust=False).mean()


def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index (RSI).

    RSI ranges from 0 to 100:
    - RSI > 70: Overbought
    - RSI < 30: Oversold

    Args:
        prices: Price series
        window: Window size (typically 14)

    Returns:
        pandas Series of RSI values
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_bands(series, window=20, num_std=2.0):
    """
    Calculate Bollinger Bands.

    Args:
        series: Input series (typically spread)
        window: Rolling window size
        num_std: Number of standard deviations for bands

    Returns:
        dict with keys: 'middle', 'upper', 'lower'
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return {
        'middle': middle,
        'upper': upper,
        'lower': lower
    }


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        dict with keys: 'macd', 'signal', 'histogram'
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


# ============================================================================
# Helper Functions
# ============================================================================

def get_indicator_summary(prices, volume):
    """
    Calculate a comprehensive set of indicators for a stock.

    Args:
        prices: Price series
        volume: Volume series

    Returns:
        dict with all indicators
    """
    indicators = {
        'sma_20': calculate_sma(prices, 20),
        'sma_50': calculate_sma(prices, 50),
        'ema_20': calculate_ema(prices, 20),
        'rsi_14': calculate_rsi(prices, 14),
        'volume_ma_20': calculate_average_volume(volume, 20),
    }

    bollinger = calculate_bollinger_bands(prices, 20, 2.0)
    indicators.update({
        'bb_upper': bollinger['upper'],
        'bb_middle': bollinger['middle'],
        'bb_lower': bollinger['lower']
    })

    macd = calculate_macd(prices, 12, 26, 9)
    indicators.update({
        'macd': macd['macd'],
        'macd_signal': macd['signal'],
        'macd_hist': macd['histogram']
    })

    return indicators
