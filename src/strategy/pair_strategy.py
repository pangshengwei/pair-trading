"""
Pair Trading Strategy Module

Implements mean reversion pair trading strategy with filtering.
"""

import pandas as pd
import numpy as np
from loguru import logger

from .base_strategy import BaseStrategy
from .signal_generator import (
    generate_entry_signals,
    generate_exit_signals,
    combine_filters
)
from ..analysis.cointegration import CointegrationAnalyzer
from ..analysis.spread import calculate_spread
from ..analysis.indicators import (
    calculate_zscore,
    calculate_rsi,
    check_liquidity
)


class PairTradingStrategy(BaseStrategy):
    """
    Mean reversion pair trading strategy.

    Strategy logic:
    1. Calculate spread between two cointegrated stocks
    2. Normalize spread to z-score
    3. Generate entry signals when |z-score| > entry_threshold
    4. Apply volume and RSI filters
    5. Exit when |z-score| < exit_threshold or hits stop loss
    """

    def __init__(self, ticker1, ticker2, **params):
        """
        Initialize pair trading strategy.

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            **params: Strategy parameters (from config)
        """
        super().__init__(**params)
        self.ticker1 = ticker1
        self.ticker2 = ticker2

        # Extract key parameters
        self.entry_threshold = self.get_param('entry_threshold', 2.0)
        self.exit_threshold = self.get_param('exit_threshold', 0.5)
        self.stop_loss_threshold = self.get_param('stop_loss_threshold', 3.0)
        self.zscore_window = self.get_param('zscore_window', 20)
        self.hedge_ratio_window = self.get_param('hedge_ratio_window', 60)

        # Filter flags
        self.use_volume_filter = self.get_param('use_volume_filter', True)
        self.use_rsi_filter = self.get_param('use_rsi_filter', True)
        self.use_trend_filter = self.get_param('use_trend_filter', False)

        # Filter parameters
        self.min_volume = self.get_param('min_volume_threshold', 100000)
        self.rsi_window = self.get_param('rsi_window', 14)
        self.rsi_overbought = self.get_param('rsi_overbought', 70)
        self.rsi_oversold = self.get_param('rsi_oversold', 30)

        logger.info(f"Initialized PairTradingStrategy for {ticker1}/{ticker2}")

    def generate_signals(self, data):
        """
        Generate trading signals from price data.

        Args:
            data: DataFrame with columns:
                  - Close_{ticker1}, Close_{ticker2}
                  - Volume_{ticker1}, Volume_{ticker2}

        Returns:
            DataFrame with signals and indicators
        """
        logger.info(f"Generating signals for {self.ticker1}/{self.ticker2}")

        # Extract price and volume series
        price1 = data[f'Close_{self.ticker1}']
        price2 = data[f'Close_{self.ticker2}']
        volume1 = data[f'Volume_{self.ticker1}']
        volume2 = data[f'Volume_{self.ticker2}']

        # Test cointegration and calculate hedge ratio
        coint_analyzer = CointegrationAnalyzer()
        coint_result = coint_analyzer.engle_granger_test(price1, price2)

        if not coint_result['cointegrated']:
            logger.warning(f"Pair {self.ticker1}/{self.ticker2} is NOT cointegrated!")

        hedge_ratio = coint_result['hedge_ratio']
        alpha = coint_result['alpha']

        # Calculate spread
        spread = calculate_spread(price1, price2, hedge_ratio, alpha)

        # Calculate z-score
        zscore = calculate_zscore(spread, window=self.zscore_window)

        # Generate preliminary entry signals
        signals = generate_entry_signals(zscore, threshold=self.entry_threshold)

        # Apply filters
        if self.use_volume_filter or self.use_rsi_filter:
            filters_dict = {}

            if self.use_volume_filter:
                filters_dict['volume'] = {
                    'volume1': volume1,
                    'volume2': volume2,
                    'min_volume': self.min_volume
                }

            if self.use_rsi_filter:
                rsi1 = calculate_rsi(price1, window=self.rsi_window)
                rsi2 = calculate_rsi(price2, window=self.rsi_window)
                filters_dict['rsi'] = {
                    'rsi1': rsi1,
                    'rsi2': rsi2,
                    'overbought': self.rsi_overbought,
                    'oversold': self.rsi_oversold
                }

            if self.use_trend_filter:
                filters_dict['trend'] = {
                    'prices1': price1,
                    'prices2': price2,
                    'ma_window': self.get_param('sma_window', 50)
                }

            # Apply all filters
            signals = combine_filters(signals, filters_dict)

        # Create results DataFrame
        results = pd.DataFrame({
            'price1': price1,
            'price2': price2,
            'volume1': volume1,
            'volume2': volume2,
            'spread': spread,
            'zscore': zscore,
            'signal': signals,
            'hedge_ratio': hedge_ratio
        })

        # Add RSI if calculated
        if self.use_rsi_filter:
            results['rsi1'] = rsi1
            results['rsi2'] = rsi2

        self.signals = results
        logger.info(f"Generated {(signals != 0).sum()} signals")

        return results

    def calculate_positions(self, signals_df):
        """
        Convert signals to positions with entry/exit logic.

        Args:
            signals_df: DataFrame from generate_signals()

        Returns:
            DataFrame with position information
        """
        logger.info("Calculating positions from signals")

        positions = []
        current_position = 0  # 0=flat, 1=long spread, -1=short spread
        entry_date = None
        entry_zscore = None

        for date, row in signals_df.iterrows():
            signal = row['signal']
            zscore = row['zscore']

            # Check for entry
            if current_position == 0 and signal != 0:
                current_position = signal
                entry_date = date
                entry_zscore = zscore
                positions.append({
                    'date': date,
                    'action': 'ENTRY',
                    'position': current_position,
                    'zscore': zscore,
                    'price1': row['price1'],
                    'price2': row['price2']
                })
                logger.debug(f"Entry: {date}, position={current_position}, z={zscore:.2f}")

            # Check for exit
            elif current_position != 0:
                # Exit on mean reversion
                if abs(zscore) < self.exit_threshold:
                    positions.append({
                        'date': date,
                        'action': 'EXIT',
                        'position': 0,
                        'zscore': zscore,
                        'price1': row['price1'],
                        'price2': row['price2'],
                        'exit_reason': 'mean_reversion'
                    })
                    logger.debug(f"Exit (reversion): {date}, z={zscore:.2f}")
                    current_position = 0
                    entry_date = None

                # Exit on stop loss
                elif abs(zscore) > self.stop_loss_threshold:
                    positions.append({
                        'date': date,
                        'action': 'EXIT',
                        'position': 0,
                        'zscore': zscore,
                        'price1': row['price1'],
                        'price2': row['price2'],
                        'exit_reason': 'stop_loss'
                    })
                    logger.warning(f"Exit (stop loss): {date}, z={zscore:.2f}")
                    current_position = 0
                    entry_date = None

        positions_df = pd.DataFrame(positions)

        if len(positions_df) > 0:
            positions_df.set_index('date', inplace=True)
            logger.info(f"Generated {len(positions_df)} position changes")
        else:
            logger.warning("No positions generated!")

        self.positions = positions_df
        return positions_df

    def get_summary(self):
        """
        Get strategy summary statistics.

        Returns:
            dict with summary info
        """
        if self.signals is None:
            return {"error": "No signals generated yet"}

        summary = {
            'ticker1': self.ticker1,
            'ticker2': self.ticker2,
            'total_signals': (self.signals['signal'] != 0).sum(),
            'long_signals': (self.signals['signal'] == 1).sum(),
            'short_signals': (self.signals['signal'] == -1).sum(),
            'avg_zscore': self.signals['zscore'].mean(),
            'max_zscore': self.signals['zscore'].max(),
            'min_zscore': self.signals['zscore'].min(),
        }

        if self.positions is not None and len(self.positions) > 0:
            summary['num_trades'] = (self.positions['action'] == 'ENTRY').sum()
            summary['num_exits'] = (self.positions['action'] == 'EXIT').sum()

        return summary
