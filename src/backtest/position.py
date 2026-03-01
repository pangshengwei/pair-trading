"""
Position Management Module

Handles individual position tracking and P&L calculation.
"""

import pandas as pd
from datetime import datetime
from loguru import logger


class Position:
    """
    Represents a single trading position (pair of stocks).
    """

    def __init__(self, ticker1, ticker2, quantity1, quantity2,
                 entry_price1, entry_price2, entry_date, position_type):
        """
        Initialize a position.

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            quantity1: Quantity of first stock (positive for long, negative for short)
            quantity2: Quantity of second stock
            entry_price1: Entry price for first stock
            entry_price2: Entry price for second stock
            entry_date: Date position was entered
            position_type: 'long_spread' or 'short_spread'
        """
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.quantity1 = quantity1
        self.quantity2 = quantity2
        self.entry_price1 = entry_price1
        self.entry_price2 = entry_price2
        self.entry_date = entry_date
        self.position_type = position_type

        # Current prices (updated daily)
        self.current_price1 = entry_price1
        self.current_price2 = entry_price2

        # Exit information
        self.exit_price1 = None
        self.exit_price2 = None
        self.exit_date = None
        self.exit_reason = None

        # P&L
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Calculate initial value
        self.entry_value = abs(quantity1 * entry_price1) + abs(quantity2 * entry_price2)

        logger.debug(f"Opened {position_type} position: {ticker1}/{ticker2} on {entry_date}")

    def update_prices(self, price1, price2):
        """
        Update current market prices and recalculate unrealized P&L.

        Args:
            price1: Current price for first stock
            price2: Current price for second stock
        """
        self.current_price1 = price1
        self.current_price2 = price2
        self.unrealized_pnl = self.calculate_pnl()

    def calculate_pnl(self):
        """
        Calculate current P&L.

        Returns:
            float: Current P&L
        """
        # P&L for stock 1
        pnl1 = self.quantity1 * (self.current_price1 - self.entry_price1)

        # P&L for stock 2
        pnl2 = self.quantity2 * (self.current_price2 - self.entry_price2)

        # Total P&L
        total_pnl = pnl1 + pnl2

        return total_pnl

    def close(self, exit_price1, exit_price2, exit_date, exit_reason='manual'):
        """
        Close the position and calculate realized P&L.

        Args:
            exit_price1: Exit price for first stock
            exit_price2: Exit price for second stock
            exit_date: Date position was closed
            exit_reason: Reason for closing ('mean_reversion', 'stop_loss', 'manual')
        """
        self.exit_price1 = exit_price1
        self.exit_price2 = exit_price2
        self.exit_date = exit_date
        self.exit_reason = exit_reason

        # Calculate realized P&L
        self.update_prices(exit_price1, exit_price2)
        self.realized_pnl = self.calculate_pnl()
        self.unrealized_pnl = 0.0

        logger.debug(f"Closed position: {self.ticker1}/{self.ticker2}, "
                    f"P&L=${self.realized_pnl:.2f}, reason={exit_reason}")

    def is_open(self):
        """Check if position is still open."""
        return self.exit_date is None

    def get_return(self):
        """
        Calculate return as percentage of entry value.

        Returns:
            float: Return percentage
        """
        if self.entry_value == 0:
            return 0.0

        if self.is_open():
            return (self.unrealized_pnl / self.entry_value) * 100
        else:
            return (self.realized_pnl / self.entry_value) * 100

    def get_holding_period(self, current_date=None):
        """
        Get holding period in days.

        Args:
            current_date: Current date (for open positions)

        Returns:
            int: Holding period in days
        """
        if self.is_open():
            if current_date is None:
                current_date = datetime.now()
            return (current_date - self.entry_date).days
        else:
            return (self.exit_date - self.entry_date).days

    def to_dict(self):
        """
        Convert position to dictionary.

        Returns:
            dict with position details
        """
        return {
            'ticker1': self.ticker1,
            'ticker2': self.ticker2,
            'position_type': self.position_type,
            'quantity1': self.quantity1,
            'quantity2': self.quantity2,
            'entry_price1': self.entry_price1,
            'entry_price2': self.entry_price2,
            'entry_date': self.entry_date,
            'entry_value': self.entry_value,
            'exit_price1': self.exit_price1,
            'exit_price2': self.exit_price2,
            'exit_date': self.exit_date,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'return_pct': self.get_return(),
            'holding_period': self.get_holding_period() if not self.is_open() else None
        }
