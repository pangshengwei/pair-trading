"""
Base Strategy Module

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All concrete strategies must implement:
    - generate_signals()
    - calculate_positions()
    """

    def __init__(self, **params):
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
        self.signals = None
        self.positions = None
        logger.info(f"Initialized {self.__class__.__name__} with params: {params}")

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals from data.

        Args:
            data: DataFrame with price and indicator data

        Returns:
            pandas Series with signals:
            - 1 for long
            - -1 for short
            - 0 for no position/exit
        """
        pass

    @abstractmethod
    def calculate_positions(self, signals):
        """
        Convert signals to actual positions.

        Args:
            signals: Series of trading signals

        Returns:
            DataFrame with position details
        """
        pass

    def validate_params(self, required_params):
        """
        Validate that required parameters are present.

        Args:
            required_params: List of required parameter names

        Raises:
            ValueError: If any required parameter is missing
        """
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

    def get_param(self, name, default=None):
        """
        Get parameter value with optional default.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value
        """
        return self.params.get(name, default)
