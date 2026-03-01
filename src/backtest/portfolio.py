"""
Portfolio Management Module

Tracks portfolio state, cash, positions, and equity curve.
"""

import pandas as pd
import numpy as np
from loguru import logger
from .position import Position


class Portfolio:
    """
    Manages portfolio state during backtest.
    """

    def __init__(self, initial_capital=100000, commission_per_share=0.005, slippage_bps=5):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash
            commission_per_share: Commission per share traded
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps / 10000  # Convert bps to decimal

        # Track positions
        self.positions = {}  # {pair_name: Position}
        self.closed_positions = []

        # Track equity curve
        self.equity_curve = []
        self.dates = []

        # Track trades
        self.trades = []

        logger.info(f"Initialized portfolio with ${initial_capital:,.0f}")

    def execute_trade(self, ticker1, ticker2, quantity1, quantity2,
                     price1, price2, date, action='open', position_type='long_spread',
                     exit_reason=None):
        """
        Execute a trade (open or close position).

        Args:
            ticker1: First ticker
            ticker2: Second ticker
            quantity1: Quantity for first stock (+ for long, - for short)
            quantity2: Quantity for second stock
            price1: Price for first stock
            price2: Price for second stock
            date: Trade date
            action: 'open' or 'close'
            position_type: 'long_spread' or 'short_spread'
            exit_reason: Reason for closing (if action='close')

        Returns:
            bool: True if trade executed successfully
        """
        pair_name = f"{ticker1}_{ticker2}"

        if action == 'open':
            # Check if we already have a position for this pair
            if pair_name in self.positions:
                logger.warning(f"Position already exists for {pair_name}")
                return False

            # Apply slippage
            exec_price1 = price1 * (1 + self.slippage_bps if quantity1 > 0 else 1 - self.slippage_bps)
            exec_price2 = price2 * (1 + self.slippage_bps if quantity2 > 0 else 1 - self.slippage_bps)

            # Calculate costs
            commission1 = abs(quantity1) * self.commission_per_share
            commission2 = abs(quantity2) * self.commission_per_share
            total_commission = commission1 + commission2

            # Calculate required capital
            required_capital = abs(quantity1 * exec_price1) + abs(quantity2 * exec_price2) + total_commission

            # Check if we have enough cash
            if required_capital > self.cash:
                logger.warning(f"Insufficient cash for {pair_name}: need ${required_capital:.2f}, have ${self.cash:.2f}")
                return False

            # Create position
            position = Position(
                ticker1, ticker2, quantity1, quantity2,
                exec_price1, exec_price2, date, position_type
            )

            # Update cash
            self.cash -= required_capital

            # Store position
            self.positions[pair_name] = position

            # Record trade
            self.trades.append({
                'date': date,
                'pair': pair_name,
                'action': 'OPEN',
                'position_type': position_type,
                'quantity1': quantity1,
                'quantity2': quantity2,
                'price1': exec_price1,
                'price2': exec_price2,
                'commission': total_commission,
                'cash_after': self.cash
            })

            logger.info(f"Opened {position_type} for {pair_name}: ${required_capital:.2f}")
            return True

        elif action == 'close':
            # Check if position exists
            if pair_name not in self.positions:
                logger.warning(f"No position to close for {pair_name}")
                return False

            position = self.positions[pair_name]

            # Apply slippage (opposite direction for closing)
            exec_price1 = price1 * (1 - self.slippage_bps if position.quantity1 > 0 else 1 + self.slippage_bps)
            exec_price2 = price2 * (1 - self.slippage_bps if position.quantity2 > 0 else 1 + self.slippage_bps)

            # Calculate commission for closing
            commission1 = abs(position.quantity1) * self.commission_per_share
            commission2 = abs(position.quantity2) * self.commission_per_share
            total_commission = commission1 + commission2

            # Close position
            position.close(exec_price1, exec_price2, date, exit_reason)

            # Calculate proceeds from closing
            proceeds1 = -position.quantity1 * exec_price1  # Negative quantity means we receive cash
            proceeds2 = -position.quantity2 * exec_price2
            total_proceeds = proceeds1 + proceeds2 - total_commission

            # Update cash
            self.cash += (position.entry_value + position.realized_pnl - total_commission)

            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[pair_name]

            # Record trade
            self.trades.append({
                'date': date,
                'pair': pair_name,
                'action': 'CLOSE',
                'exit_reason': exit_reason,
                'quantity1': position.quantity1,
                'quantity2': position.quantity2,
                'price1': exec_price1,
                'price2': exec_price2,
                'pnl': position.realized_pnl,
                'commission': total_commission,
                'cash_after': self.cash
            })

            logger.info(f"Closed {pair_name}: P&L=${position.realized_pnl:.2f}")
            return True

        return False

    def update_positions(self, date, prices_dict):
        """
        Update all open positions with current prices.

        Args:
            date: Current date
            prices_dict: Dict mapping ticker to current price
        """
        for pair_name, position in self.positions.items():
            if position.ticker1 in prices_dict and position.ticker2 in prices_dict:
                price1 = prices_dict[position.ticker1]
                price2 = prices_dict[position.ticker2]
                position.update_prices(price1, price2)

        # Update equity curve
        portfolio_value = self.get_portfolio_value()
        self.equity_curve.append(portfolio_value)
        self.dates.append(date)

    def get_portfolio_value(self):
        """
        Calculate total portfolio value (cash + positions).

        Returns:
            float: Total portfolio value
        """
        positions_value = sum(
            pos.entry_value + pos.unrealized_pnl
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_equity_curve_df(self):
        """
        Get equity curve as DataFrame.

        Returns:
            pandas DataFrame with dates and portfolio values
        """
        return pd.DataFrame({
            'date': self.dates,
            'portfolio_value': self.equity_curve
        }).set_index('date')

    def get_trades_df(self):
        """
        Get trades as DataFrame.

        Returns:
            pandas DataFrame with trade history
        """
        if len(self.trades) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_closed_positions_df(self):
        """
        Get closed positions as DataFrame.

        Returns:
            pandas DataFrame with closed position details
        """
        if len(self.closed_positions) == 0:
            return pd.DataFrame()
        return pd.DataFrame([pos.to_dict() for pos in self.closed_positions])

    def get_summary(self):
        """
        Get portfolio summary statistics.

        Returns:
            dict with summary statistics
        """
        current_value = self.get_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital

        summary = {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'total_return': total_return * 100,
            'total_pnl': current_value - self.initial_capital,
            'num_open_positions': len(self.positions),
            'num_closed_positions': len(self.closed_positions),
            'num_trades': len(self.trades)
        }

        return summary
