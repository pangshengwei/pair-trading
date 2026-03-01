"""
Backtester Module

Main backtesting engine for pair trading strategies.
"""

import pandas as pd
import numpy as np
from loguru import logger

from .portfolio import Portfolio
from .metrics import calculate_all_metrics


class Backtester:
    """
    Backtesting engine for pair trading strategies.
    """

    def __init__(self, initial_capital=100000, commission_per_share=0.005, slippage_bps=5):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission_per_share: Commission per share
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

        logger.info(f"Initialized Backtester with ${initial_capital:,.0f} capital")

    def run(self, strategy, data):
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy object (e.g., PairTradingStrategy)
            data: DataFrame with price and volume data

        Returns:
            dict with backtest results
        """
        logger.info(f"Running backtest for {strategy.ticker1}/{strategy.ticker2}")

        # Generate signals
        signals_df = strategy.generate_signals(data)

        # Calculate positions
        positions_df = strategy.calculate_positions(signals_df)

        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_per_share=self.commission_per_share,
            slippage_bps=self.slippage_bps
        )

        # Simulate trading
        self._simulate_trading(portfolio, signals_df, positions_df, strategy)

        # Calculate metrics
        equity_curve_df = portfolio.get_equity_curve_df()
        closed_positions_df = portfolio.get_closed_positions_df()
        trades_df = portfolio.get_trades_df()

        metrics = calculate_all_metrics(
            equity_curve_df,
            closed_positions_df,
            self.initial_capital
        )

        # Compile results
        results = {
            'strategy': strategy,
            'signals': signals_df,
            'positions': positions_df,
            'equity_curve': equity_curve_df,
            'closed_positions': closed_positions_df,
            'trades': trades_df,
            'metrics': metrics,
            'portfolio_summary': portfolio.get_summary(),
            # Add these for visualization
            'ticker1': strategy.ticker1,
            'ticker2': strategy.ticker2,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown_pct'],
            'win_rate': metrics['win_rate'],
            'num_trades': metrics['num_trades']
        }

        logger.info(f"Backtest complete: Return={metrics['total_return']:.2f}%, "
                   f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                   f"MaxDD={metrics['max_drawdown_pct']:.2f}%")

        return results

    def _simulate_trading(self, portfolio, signals_df, positions_df, strategy):
        """
        Simulate trading day by day.

        Args:
            portfolio: Portfolio object
            signals_df: DataFrame with signals and indicators
            positions_df: DataFrame with position entries/exits
            strategy: Strategy object
        """
        logger.info("Simulating trading...")

        # Get position size parameter
        position_size = strategy.get_param('position_size', 0.3)

        # Track current state
        open_pair = None

        # Iterate through each day
        for date, row in signals_df.iterrows():
            # Update prices for open positions
            prices_dict = {
                strategy.ticker1: row['price1'],
                strategy.ticker2: row['price2']
            }
            portfolio.update_positions(date, prices_dict)

            # Check if we have a position action on this date
            if positions_df is not None and len(positions_df) > 0 and date in positions_df.index:
                pos_action = positions_df.loc[date]

                if pos_action['action'] == 'ENTRY':
                    # Open new position
                    if open_pair is None:
                        position_type = pos_action['position']  # 1 or -1
                        hedge_ratio = row['hedge_ratio']

                        # Calculate quantities
                        quantities = self._calculate_quantities(
                            portfolio,
                            row['price1'],
                            row['price2'],
                            hedge_ratio,
                            position_type,
                            position_size
                        )

                        if quantities is not None:
                            q1, q2 = quantities

                            # Execute trade
                            success = portfolio.execute_trade(
                                strategy.ticker1, strategy.ticker2,
                                q1, q2,
                                row['price1'], row['price2'],
                                date,
                                action='open',
                                position_type='long_spread' if position_type == 1 else 'short_spread'
                            )

                            if success:
                                open_pair = f"{strategy.ticker1}_{strategy.ticker2}"

                elif pos_action['action'] == 'EXIT':
                    # Close existing position
                    if open_pair is not None:
                        exit_reason = pos_action.get('exit_reason', 'signal')

                        success = portfolio.execute_trade(
                            strategy.ticker1, strategy.ticker2,
                            0, 0,  # Not used for closing
                            row['price1'], row['price2'],
                            date,
                            action='close',
                            exit_reason=exit_reason
                        )

                        if success:
                            open_pair = None

        # Close any remaining open positions at the end
        if open_pair is not None:
            last_date = signals_df.index[-1]
            last_row = signals_df.loc[last_date]

            portfolio.execute_trade(
                strategy.ticker1, strategy.ticker2,
                0, 0,
                last_row['price1'], last_row['price2'],
                last_date,
                action='close',
                exit_reason='end_of_backtest'
            )

        logger.info(f"Simulation complete: {len(portfolio.closed_positions)} trades executed")

    def _calculate_quantities(self, portfolio, price1, price2, hedge_ratio, position_type, position_size):
        """
        Calculate share quantities for a pair trade.

        Args:
            portfolio: Portfolio object
            price1: Price of first stock
            price2: Price of second stock
            hedge_ratio: Hedge ratio (beta)
            position_type: 1 for long spread, -1 for short spread
            position_size: Fraction of portfolio to allocate

        Returns:
            tuple: (quantity1, quantity2) or None if insufficient capital
        """
        # Calculate capital to allocate
        available_capital = portfolio.cash * position_size

        # For pair trading, we want equal dollar amounts in both legs
        # Spread = price1 - hedge_ratio * price2
        # For long spread: buy stock1, sell stock2
        # For short spread: sell stock1, buy stock2

        # Calculate quantities to have equal dollar exposure
        # Capital split: half to each leg
        capital_per_leg = available_capital / 2

        # Calculate shares
        # For stock1
        shares1 = int(capital_per_leg / price1)

        # For stock2, adjust by hedge ratio
        dollar_amount2 = shares1 * price1 * hedge_ratio
        shares2 = int(dollar_amount2 / price2)

        # Apply position direction
        if position_type == 1:  # Long spread
            quantity1 = shares1  # Long stock1
            quantity2 = -shares2  # Short stock2
        else:  # Short spread
            quantity1 = -shares1  # Short stock1
            quantity2 = shares2  # Long stock2

        # Check if we have enough capital
        required_capital = abs(quantity1 * price1) + abs(quantity2 * price2)

        if required_capital > portfolio.cash:
            logger.warning(f"Insufficient capital: need ${required_capital:.2f}, have ${portfolio.cash:.2f}")
            return None

        return (quantity1, quantity2)
