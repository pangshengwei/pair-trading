"""
Visualization Module

Generates charts and plots for backtest results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_equity_curve(equity_curve_df, pair_name, output_path=None):
    """Plot equity curve."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity_curve_df.index, equity_curve_df['portfolio_value'], linewidth=2)
    ax.set_title(f'Equity Curve - {pair_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Saved equity curve to {output_path}")

    plt.close()


def plot_drawdown(equity_curve_df, pair_name, output_path=None):
    """Plot drawdown."""
    values = equity_curve_df['portfolio_value']
    running_max = values.expanding().max()
    drawdown = (values - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax.set_title(f'Drawdown - {pair_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Saved drawdown to {output_path}")

    plt.close()


def plot_spread_and_zscore(signals_df, positions_df, pair_name, output_path=None):
    """Plot spread, z-score, and trade markers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot spread
    ax1.plot(signals_df.index, signals_df['spread'], linewidth=1, label='Spread')
    ax1.set_title(f'Spread - {pair_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot z-score
    ax2.plot(signals_df.index, signals_df['zscore'], linewidth=1, label='Z-Score', color='blue')
    ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Entry Threshold')
    ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Exit Threshold')
    ax2.axhline(y=-0.5, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

    # Add trade markers
    if positions_df is not None and len(positions_df) > 0:
        entries = positions_df[positions_df['action'] == 'ENTRY']
        exits = positions_df[positions_df['action'] == 'EXIT']

        ax2.scatter(entries.index, entries['zscore'], marker='^', color='green',
                   s=100, label='Entry', zorder=5)
        ax2.scatter(exits.index, exits['zscore'], marker='v', color='red',
                   s=100, label='Exit', zorder=5)

    ax2.set_title(f'Z-Score with Trade Signals - {pair_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Saved spread/zscore to {output_path}")

    plt.close()


def plot_returns_distribution(equity_curve_df, pair_name, output_path=None):
    """Plot returns distribution."""
    returns = equity_curve_df['portfolio_value'].pct_change().dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(returns * 100, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(returns.mean() * 100, color='r', linestyle='--',
              linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax.set_title(f'Returns Distribution - {pair_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Saved returns distribution to {output_path}")

    plt.close()


def plot_price_series(signals_df, pair_name, ticker1, ticker2, output_path=None):
    """Plot both stock prices."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(signals_df.index, signals_df['price1'], label=ticker1, linewidth=1.5)
    ax.plot(signals_df.index, signals_df['price2'], label=ticker2, linewidth=1.5)
    ax.set_title(f'Price Series - {pair_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Saved price series to {output_path}")

    plt.close()


def plot_performance_summary(metrics, pair_name, output_path=None):
    """Plot performance metrics summary."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Summary - {pair_name}', fontsize=16, fontweight='bold')

    # Total return
    ax1.bar(['Total Return'], [metrics['total_return']], color='skyblue')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Total Return')
    ax1.grid(True, alpha=0.3, axis='y')

    # Sharpe ratio
    ax2.bar(['Sharpe Ratio'], [metrics['sharpe_ratio']], color='lightgreen')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Sharpe Ratio')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Max drawdown
    ax3.bar(['Max Drawdown'], [metrics['max_drawdown_pct']], color='salmon')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_title('Maximum Drawdown')
    ax3.grid(True, alpha=0.3, axis='y')

    # Win rate
    ax4.bar(['Win Rate'], [metrics['win_rate']], color='gold')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate')
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Saved performance summary to {output_path}")

    plt.close()


def save_all_plots(backtest_result, output_dir):
    """Save all plots for a backtest result."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pair_name = f"{backtest_result['ticker1']}_{backtest_result['ticker2']}"

    logger.info(f"Generating charts for {pair_name}...")

    # 1. Equity curve
    plot_equity_curve(
        backtest_result['equity_curve'],
        pair_name,
        output_path / 'equity_curve.png'
    )

    # 2. Drawdown
    plot_drawdown(
        backtest_result['equity_curve'],
        pair_name,
        output_path / 'drawdown.png'
    )

    # 3. Spread and z-score
    plot_spread_and_zscore(
        backtest_result['signals'],
        backtest_result['positions'],
        pair_name,
        output_path / 'spread_zscore.png'
    )

    # 4. Returns distribution
    plot_returns_distribution(
        backtest_result['equity_curve'],
        pair_name,
        output_path / 'returns_distribution.png'
    )

    # 5. Price series
    plot_price_series(
        backtest_result['signals'],
        pair_name,
        backtest_result['ticker1'],
        backtest_result['ticker2'],
        output_path / 'price_series.png'
    )

    # 6. Performance summary
    plot_performance_summary(
        backtest_result['metrics'],
        pair_name,
        output_path / 'performance_summary.png'
    )

    logger.info(f"Charts saved to {output_dir}")
