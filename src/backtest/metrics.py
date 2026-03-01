"""
Performance Metrics Module

Calculates various performance metrics for backtest results.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_returns(equity_curve):
    """
    Calculate returns from equity curve.

    Args:
        equity_curve: Series or DataFrame with portfolio values

    Returns:
        pandas Series of returns
    """
    if isinstance(equity_curve, pd.DataFrame):
        equity_curve = equity_curve['portfolio_value']

    returns = equity_curve.pct_change().dropna()
    return returns


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert annual risk-free rate to period rate
    rf_period = risk_free_rate / periods_per_year

    # Calculate excess returns
    excess_returns = returns - rf_period

    # Calculate Sharpe ratio
    if excess_returns.std() == 0:
        return 0.0

    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate Sortino ratio (downside risk-adjusted return).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        float: Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period

    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: Series or DataFrame with portfolio values

    Returns:
        dict with 'max_drawdown', 'max_drawdown_pct', 'peak_date', 'trough_date'
    """
    if isinstance(equity_curve, pd.DataFrame):
        equity_curve = equity_curve['portfolio_value']

    if len(equity_curve) == 0:
        return {'max_drawdown': 0, 'max_drawdown_pct': 0}

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = equity_curve - running_max
    drawdown_pct = (drawdown / running_max) * 100

    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_pct = drawdown_pct.min()

    # Find peak and trough dates
    trough_date = drawdown.idxmin()
    peak_date = running_max[:trough_date].idxmax()

    result = {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'peak_date': peak_date,
        'trough_date': trough_date
    }

    return result


def calculate_win_rate(trades_df):
    """
    Calculate win rate from trades.

    Args:
        trades_df: DataFrame with closed positions

    Returns:
        float: Win rate as percentage
    """
    if len(trades_df) == 0:
        return 0.0

    if 'realized_pnl' not in trades_df.columns:
        logger.warning("No realized_pnl column in trades")
        return 0.0

    winning_trades = (trades_df['realized_pnl'] > 0).sum()
    total_trades = len(trades_df)

    win_rate = (winning_trades / total_trades) * 100

    return win_rate


def calculate_profit_factor(trades_df):
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades_df: DataFrame with closed positions

    Returns:
        float: Profit factor
    """
    if len(trades_df) == 0 or 'realized_pnl' not in trades_df.columns:
        return 0.0

    gross_profit = trades_df[trades_df['realized_pnl'] > 0]['realized_pnl'].sum()
    gross_loss = abs(trades_df[trades_df['realized_pnl'] < 0]['realized_pnl'].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    profit_factor = gross_profit / gross_loss

    return profit_factor


def calculate_calmar_ratio(returns, max_drawdown_pct, periods_per_year=252):
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of returns
        max_drawdown_pct: Maximum drawdown percentage
        periods_per_year: Number of periods per year

    Returns:
        float: Calmar ratio
    """
    if len(returns) == 0 or max_drawdown_pct == 0:
        return 0.0

    # Annualized return
    total_return = (1 + returns).prod() - 1
    num_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / num_years) - 1

    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown_pct / 100)

    return calmar


def calculate_all_metrics(equity_curve, trades_df, initial_capital, risk_free_rate=0.02):
    """
    Calculate all performance metrics.

    Args:
        equity_curve: DataFrame or Series with portfolio values
        trades_df: DataFrame with closed positions
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate

    Returns:
        dict with all metrics
    """
    # Calculate returns
    returns = calculate_returns(equity_curve)

    # Calculate drawdown
    dd_info = calculate_max_drawdown(equity_curve)

    # Get final value
    if isinstance(equity_curve, pd.DataFrame):
        final_value = equity_curve['portfolio_value'].iloc[-1]
    else:
        final_value = equity_curve.iloc[-1]

    # Calculate total return
    total_return = (final_value - initial_capital) / initial_capital

    # Calculate metrics
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return * 100,  # As percentage
        'annualized_return': ((1 + total_return) ** (252 / len(returns)) - 1) * 100,
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': dd_info['max_drawdown'],
        'max_drawdown_pct': dd_info['max_drawdown_pct'],
        'calmar_ratio': calculate_calmar_ratio(returns, dd_info['max_drawdown_pct']),
        'volatility': returns.std() * np.sqrt(252) * 100,  # Annualized
        'num_trades': len(trades_df) if len(trades_df) > 0 else 0,
        'win_rate': calculate_win_rate(trades_df),
        'profit_factor': calculate_profit_factor(trades_df),
    }

    # Add trade statistics if available
    if len(trades_df) > 0 and 'realized_pnl' in trades_df.columns:
        metrics['avg_trade_pnl'] = trades_df['realized_pnl'].mean()
        metrics['max_trade_win'] = trades_df['realized_pnl'].max()
        metrics['max_trade_loss'] = trades_df['realized_pnl'].min()

    if len(trades_df) > 0 and 'return_pct' in trades_df.columns:
        metrics['avg_trade_return'] = trades_df['return_pct'].mean()

    logger.info(f"Calculated metrics: Return={metrics['total_return']:.2f}%, Sharpe={metrics['sharpe_ratio']:.2f}")

    return metrics
