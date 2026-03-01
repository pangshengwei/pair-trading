"""
Cointegration Analysis Module

Implements cointegration tests (Engle-Granger, ADF) and hedge ratio calculation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from loguru import logger


class CointegrationAnalyzer:
    """
    Analyzes cointegration relationship between two price series.
    """

    def __init__(self):
        pass

    def engle_granger_test(self, series1, series2):
        """
        Perform Engle-Granger cointegration test.

        Steps:
        1. Regress series1 on series2 to get hedge ratio (beta)
        2. Calculate residuals (spread)
        3. Test residuals for stationarity using ADF test

        Args:
            series1: First price series (pandas Series)
            series2: Second price series (pandas Series)

        Returns:
            dict with keys: 'cointegrated', 'p_value', 'hedge_ratio',
                           'adf_statistic', 'critical_values', 'residuals'
        """
        logger.info("Running Engle-Granger cointegration test")

        # Calculate hedge ratio via OLS
        hedge_ratio, alpha = self.calculate_hedge_ratio(series1, series2)

        # Calculate residuals (spread)
        residuals = series1 - hedge_ratio * series2 - alpha

        # Test residuals for stationarity
        adf_result = self.adf_test(residuals)

        # Check if cointegrated (p-value < 0.05)
        cointegrated = adf_result['p_value'] < 0.05

        result = {
            'cointegrated': cointegrated,
            'p_value': adf_result['p_value'],
            'hedge_ratio': hedge_ratio,
            'alpha': alpha,
            'adf_statistic': adf_result['adf_statistic'],
            'critical_values': adf_result['critical_values'],
            'residuals': residuals
        }

        if cointegrated:
            logger.info(f"Series are cointegrated (p={result['p_value']:.4f}, β={hedge_ratio:.4f})")
        else:
            logger.warning(f"Series are NOT cointegrated (p={result['p_value']:.4f})")

        return result

    def calculate_hedge_ratio(self, series1, series2):
        """
        Calculate hedge ratio via OLS regression: series1 = beta * series2 + alpha.

        Args:
            series1: Dependent variable (pandas Series)
            series2: Independent variable (pandas Series)

        Returns:
            tuple: (hedge_ratio (beta), intercept (alpha))
        """
        # Prepare data
        X = add_constant(series2.values)
        y = series1.values

        # Run OLS regression
        model = OLS(y, X)
        results = model.fit()

        alpha = results.params[0]
        beta = results.params[1]

        logger.debug(f"Hedge ratio: β={beta:.4f}, α={alpha:.4f}, R²={results.rsquared:.4f}")

        return beta, alpha

    def rolling_hedge_ratio(self, series1, series2, window=60):
        """
        Calculate rolling hedge ratio.

        Args:
            series1: First price series
            series2: Second price series
            window: Rolling window size in days

        Returns:
            pandas Series of rolling hedge ratios
        """
        hedge_ratios = []
        dates = []

        for i in range(window, len(series1)):
            s1_window = series1.iloc[i-window:i]
            s2_window = series2.iloc[i-window:i]

            beta, _ = self.calculate_hedge_ratio(s1_window, s2_window)
            hedge_ratios.append(beta)
            dates.append(series1.index[i])

        rolling_hr = pd.Series(hedge_ratios, index=dates)
        logger.debug(f"Calculated rolling hedge ratio (window={window})")

        return rolling_hr

    def adf_test(self, series, regression='c'):
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            series: Time series to test (pandas Series)
            regression: Type of regression ('c', 'ct', 'ctt', 'n')

        Returns:
            dict with keys: 'adf_statistic', 'p_value', 'critical_values',
                           'stationary'
        """
        # Run ADF test
        adf_stat, p_value, _, _, critical_values, _ = adfuller(
            series.dropna(),
            regression=regression
        )

        stationary = p_value < 0.05

        result = {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'stationary': stationary
        }

        return result

    def half_life(self, residuals):
        """
        Calculate half-life of mean reversion for the spread.

        Args:
            residuals: Spread residuals series

        Returns:
            float: Half-life in days
        """
        # Fit AR(1) model: residuals[t] = alpha + beta * residuals[t-1] + error
        lagged_residuals = residuals.shift(1).dropna()
        residuals_aligned = residuals[1:]

        X = add_constant(lagged_residuals.values)
        y = residuals_aligned.values

        model = OLS(y, X)
        results = model.fit()

        beta = results.params[1]

        # Calculate half-life
        if beta < 1 and beta > 0:
            half_life = -np.log(2) / np.log(beta)
            logger.info(f"Half-life of mean reversion: {half_life:.2f} days")
            return half_life
        else:
            logger.warning(f"Invalid beta for half-life calculation: {beta:.4f}")
            return np.inf
