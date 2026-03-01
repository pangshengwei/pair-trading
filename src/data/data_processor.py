"""
Data Processor Module

Handles data cleaning, alignment, and preprocessing.
"""

import pandas as pd
import numpy as np
from loguru import logger


def align_timestamps(df1, df2):
    """
    Align two DataFrames by their timestamps (indices).

    Args:
        df1: First DataFrame with datetime index
        df2: Second DataFrame with datetime index

    Returns:
        Tuple of aligned (df1, df2)
    """
    # Find common dates
    common_dates = df1.index.intersection(df2.index)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between the two series")

    df1_aligned = df1.loc[common_dates]
    df2_aligned = df2.loc[common_dates]

    logger.info(f"Aligned data: {len(common_dates)} common dates")
    return df1_aligned, df2_aligned


def handle_missing_data(df, method='ffill', limit=None):
    """
    Handle missing values in DataFrame.

    Args:
        df: DataFrame with potential missing values
        method: Method to use ('ffill', 'bfill', 'interpolate', 'drop')
        limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with missing values handled
    """
    if method == 'ffill':
        df = df.fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        df = df.fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        df = df.interpolate(method='linear', limit=limit)
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Drop any remaining NaNs
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with missing data")

    return df


def calculate_returns(df, method='log'):
    """
    Calculate returns from price series.

    Args:
        df: DataFrame with price columns
        method: 'log' for log returns or 'simple' for simple returns

    Returns:
        DataFrame with return columns
    """
    returns_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        if 'Close' in col or 'Price' in col:
            if method == 'log':
                returns_df[f'Return_{col}'] = np.log(df[col] / df[col].shift(1))
            elif method == 'simple':
                returns_df[f'Return_{col}'] = df[col].pct_change()
            else:
                raise ValueError(f"Unknown method: {method}")

    # Drop first row with NaN returns
    returns_df = returns_df.dropna()
    logger.info(f"Calculated {method} returns for {len(returns_df.columns)} series")

    return returns_df


def resample_data(df, frequency='1D'):
    """
    Resample data to a different frequency.

    Args:
        df: DataFrame with datetime index
        frequency: Target frequency ('1D', '1H', '1W', etc.)

    Returns:
        Resampled DataFrame
    """
    # Resample prices using last value
    price_cols = [col for col in df.columns if 'Close' in col or 'Price' in col]
    volume_cols = [col for col in df.columns if 'Volume' in col]

    resampled = pd.DataFrame()

    if price_cols:
        resampled = pd.concat([resampled, df[price_cols].resample(frequency).last()], axis=1)

    if volume_cols:
        resampled = pd.concat([resampled, df[volume_cols].resample(frequency).sum()], axis=1)

    resampled = resampled.dropna()
    logger.info(f"Resampled data to {frequency}: {len(resampled)} rows")

    return resampled


def normalize_series(series):
    """
    Normalize a series to have zero mean and unit variance.

    Args:
        series: pandas Series

    Returns:
        Normalized series
    """
    return (series - series.mean()) / series.std()


def winsorize_series(series, lower_pct=0.01, upper_pct=0.99):
    """
    Winsorize extreme values in a series.

    Args:
        series: pandas Series
        lower_pct: Lower percentile threshold
        upper_pct: Upper percentile threshold

    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)

    return series.clip(lower=lower_bound, upper=upper_bound)
