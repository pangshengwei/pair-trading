"""
Spread Calculation Module

Calculates and normalizes the spread between pair trading stocks.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_spread(price1, price2, hedge_ratio, alpha=0):
    """
    Calculate spread between two price series.

    Spread = price1 - hedge_ratio * price2 - alpha

    Args:
        price1: First price series (pandas Series)
        price2: Second price series (pandas Series)
        hedge_ratio: Hedge ratio (beta) from OLS regression
        alpha: Intercept from OLS regression (default: 0)

    Returns:
        pandas Series of spread values
    """
    spread = price1 - hedge_ratio * price2 - alpha
    logger.debug(f"Calculated spread: mean={spread.mean():.2f}, std={spread.std():.2f}")
    return spread


def normalize_spread(spread):
    """
    Normalize spread to zero mean and unit variance.

    Args:
        spread: Spread series (pandas Series)

    Returns:
        Normalized spread series
    """
    normalized = (spread - spread.mean()) / spread.std()
    logger.debug(f"Normalized spread: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    return normalized


def calculate_rolling_spread(price1, price2, hedge_ratio_series):
    """
    Calculate spread using a rolling hedge ratio.

    Args:
        price1: First price series
        price2: Second price series
        hedge_ratio_series: Series of rolling hedge ratios (aligned with prices)

    Returns:
        pandas Series of spread values
    """
    # Align all series
    common_idx = price1.index.intersection(price2.index).intersection(hedge_ratio_series.index)

    p1_aligned = price1.loc[common_idx]
    p2_aligned = price2.loc[common_idx]
    hr_aligned = hedge_ratio_series.loc[common_idx]

    # Calculate spread
    spread = p1_aligned - hr_aligned * p2_aligned

    logger.debug(f"Calculated rolling spread with {len(spread)} points")
    return spread


def calculate_spread_ratio(price1, price2):
    """
    Calculate spread as a ratio instead of difference.

    Spread ratio = price1 / price2

    Args:
        price1: First price series
        price2: Second price series

    Returns:
        pandas Series of spread ratios
    """
    spread_ratio = price1 / price2
    logger.debug(f"Calculated spread ratio: mean={spread_ratio.mean():.4f}")
    return spread_ratio


def get_spread_statistics(spread):
    """
    Calculate descriptive statistics for spread.

    Args:
        spread: Spread series

    Returns:
        dict with statistics
    """
    stats = {
        'mean': spread.mean(),
        'std': spread.std(),
        'min': spread.min(),
        'max': spread.max(),
        'median': spread.median(),
        'skew': spread.skew(),
        'kurtosis': spread.kurtosis()
    }

    logger.info(f"Spread statistics: μ={stats['mean']:.2f}, σ={stats['std']:.2f}")
    return stats
