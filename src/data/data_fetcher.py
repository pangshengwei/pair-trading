"""
Data Fetcher Module

Fetches historical price and volume data from yfinance.
No API key required - completely FREE!
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from loguru import logger
import os


class YFinanceDataFetcher:
    """
    Fetches historical stock data from Yahoo Finance using yfinance.
    """

    def __init__(self, cache_dir='./data/cache', use_cache=True):
        """
        Initialize the data fetcher.

        Args:
            cache_dir: Directory to cache downloaded data
            use_cache: Whether to use cached data if available
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_pair_data(self, ticker1, ticker2, start_date, end_date):
        """
        Fetch historical data for a pair of tickers.

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: [date, Close_ticker1, Close_ticker2,
                                     Volume_ticker1, Volume_ticker2]
        """
        logger.info(f"Fetching data for pair: {ticker1}/{ticker2}")

        # Try to load from cache
        if self.use_cache:
            cached_data = self._load_from_cache(ticker1, ticker2, start_date, end_date)
            if cached_data is not None:
                logger.info(f"Loaded data from cache for {ticker1}/{ticker2}")
                return cached_data

        # Fetch data from yfinance
        try:
            data1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)
            data2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)

            if data1.empty or data2.empty:
                raise ValueError(f"No data returned for {ticker1} or {ticker2}")

            # yfinance returns regular columns for single ticker downloads
            # Extract Adj Close and Volume directly and squeeze to 1D
            adj_close1 = data1['Adj Close'].squeeze() if 'Adj Close' in data1.columns else data1['Close'].squeeze()
            volume1 = data1['Volume'].squeeze()

            adj_close2 = data2['Adj Close'].squeeze() if 'Adj Close' in data2.columns else data2['Close'].squeeze()
            volume2 = data2['Volume'].squeeze()

            # Combine data with explicit index
            df = pd.DataFrame({
                f'Close_{ticker1}': adj_close1,
                f'Close_{ticker2}': adj_close2,
                f'Volume_{ticker1}': volume1,
                f'Volume_{ticker2}': volume2
            }, index=data1.index)

            # Validate data
            self.validate_data(df)

            # Save to cache
            if self.use_cache:
                self._save_to_cache(ticker1, ticker2, start_date, end_date, df)

            logger.info(f"Successfully fetched {len(df)} rows for {ticker1}/{ticker2}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker1}/{ticker2}: {e}")
            raise

    def fetch_multiple_pairs(self, pairs_list, start_date, end_date):
        """
        Fetch data for multiple pairs.

        Args:
            pairs_list: List of dicts with 'ticker1' and 'ticker2' keys
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping pair names to DataFrames
        """
        results = {}
        for pair in pairs_list:
            pair_name = f"{pair['ticker1']}_{pair['ticker2']}"
            try:
                data = self.fetch_pair_data(
                    pair['ticker1'],
                    pair['ticker2'],
                    start_date,
                    end_date
                )
                results[pair_name] = data
            except Exception as e:
                logger.error(f"Failed to fetch {pair_name}: {e}")
                results[pair_name] = None

        return results

    def validate_data(self, df):
        """
        Validate the fetched data for quality issues.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If data quality issues are found
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        for col, pct in missing_pct.items():
            if pct > 10:  # More than 10% missing
                logger.warning(f"Column {col} has {pct:.1f}% missing values")

        # Check for duplicate indices
        if df.index.duplicated().any():
            logger.warning("DataFrame has duplicate date indices")
            df = df[~df.index.duplicated(keep='first')]

        # Check for reasonable price values
        for col in df.columns:
            if 'Close' in col:
                if (df[col] <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive prices")

        logger.info("Data validation passed")

    def _get_cache_filename(self, ticker1, ticker2, start_date, end_date):
        """Generate cache filename for a pair."""
        return os.path.join(
            self.cache_dir,
            f"{ticker1}_{ticker2}_{start_date}_{end_date}.parquet"
        )

    def _save_to_cache(self, ticker1, ticker2, start_date, end_date, df):
        """Save DataFrame to cache."""
        try:
            filename = self._get_cache_filename(ticker1, ticker2, start_date, end_date)
            df.to_parquet(filename)
            logger.debug(f"Saved data to cache: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, ticker1, ticker2, start_date, end_date):
        """Load DataFrame from cache if it exists."""
        try:
            filename = self._get_cache_filename(ticker1, ticker2, start_date, end_date)
            if os.path.exists(filename):
                return pd.read_parquet(filename)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        return None
