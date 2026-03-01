"""
Main Entry Point for Pair Trading System

Run backtests on configured trading pairs and generate reports.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.utils.logger import setup_logger
from src.data.data_fetcher import YFinanceDataFetcher
from src.analysis.cointegration import CointegrationAnalyzer
from src.strategy.pair_strategy import PairTradingStrategy
from src.backtest.backtester import Backtester
from src.visualization.plotter import save_all_plots
from config.pairs_config import TRADING_PAIRS, DATA_CONFIG
from config.strategy_config import STRATEGY_PARAMS, BACKTEST_PARAMS


def main():
    """Main execution function."""

    # Setup logger
    setup_logger(log_level="INFO")

    logger.info("="*60)
    logger.info("PAIR TRADING BACKTEST SYSTEM")
    logger.info("="*60)

    # Initialize data fetcher
    fetcher = YFinanceDataFetcher(
        cache_dir=DATA_CONFIG.get('cache_dir', './data/cache'),
        use_cache=DATA_CONFIG.get('cache_enabled', True)
    )

    # Create output directories
    os.makedirs('./results/plots', exist_ok=True)
    os.makedirs('./results/reports', exist_ok=True)

    # Store results for all pairs
    all_results = {}

    # Process each pair
    for pair_config in TRADING_PAIRS:
        pair_name = pair_config['name']
        ticker1 = pair_config['ticker1']
        ticker2 = pair_config['ticker2']

        logger.info("\n" + "="*60)
        logger.info(f"Processing pair: {pair_name}")
        logger.info(f"Tickers: {ticker1} / {ticker2}")
        logger.info(f"Sector: {pair_config['sector']}")
        logger.info("="*60)

        try:
            # 1. Fetch historical data
            logger.info("Fetching historical data...")
            data = fetcher.fetch_pair_data(
                ticker1,
                ticker2,
                DATA_CONFIG['start_date'],
                DATA_CONFIG['end_date']
            )

            logger.info(f"Data fetched: {len(data)} rows from {data.index[0]} to {data.index[-1]}")

            # 2. Test cointegration
            logger.info("Testing cointegration...")
            analyzer = CointegrationAnalyzer()

            price1 = data[f'Close_{ticker1}']
            price2 = data[f'Close_{ticker2}']

            coint_result = analyzer.engle_granger_test(price1, price2)

            logger.info(f"Cointegration test results:")
            logger.info(f"  - P-value: {coint_result['p_value']:.4f}")
            logger.info(f"  - Hedge ratio (β): {coint_result['hedge_ratio']:.4f}")
            logger.info(f"  - Cointegrated: {coint_result['cointegrated']}")

            if not coint_result['cointegrated']:
                logger.warning(f"WARNING: Pair {pair_name} is NOT cointegrated at 95% confidence level!")
                logger.warning("Results may not be reliable for mean reversion trading.")

            # 3. Build strategy
            logger.info("Building trading strategy...")
            strategy = PairTradingStrategy(
                ticker1=ticker1,
                ticker2=ticker2,
                **STRATEGY_PARAMS
            )

            # 4. Run backtest
            logger.info("Running backtest...")
            backtester = Backtester(**BACKTEST_PARAMS)
            backtest_result = backtester.run(strategy, data)

            # Store results
            all_results[pair_name] = backtest_result

            # 5. Print summary
            logger.info("\n" + "-"*60)
            logger.info("BACKTEST RESULTS")
            logger.info("-"*60)

            metrics = backtest_result['metrics']

            logger.info(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
            logger.info(f"Final Value: ${metrics['final_value']:,.2f}")
            logger.info(f"Total Return: {metrics['total_return']:.2f}%")
            logger.info(f"Annualized Return: {metrics['annualized_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            logger.info(f"Volatility (annual): {metrics['volatility']:.2f}%")
            logger.info(f"Number of Trades: {metrics['num_trades']}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")

            if metrics['num_trades'] > 0:
                logger.info(f"Avg Trade P&L: ${metrics.get('avg_trade_pnl', 0):.2f}")
                logger.info(f"Max Trade Win: ${metrics.get('max_trade_win', 0):.2f}")
                logger.info(f"Max Trade Loss: ${metrics.get('max_trade_loss', 0):.2f}")

            # 6. Generate charts
            logger.info("\n" + "-"*60)
            logger.info("Generating charts...")
            logger.info("-"*60)

            output_dir = f"./results/plots/{pair_name}"
            save_all_plots(backtest_result, output_dir)

            logger.info(f"Charts saved to: {output_dir}")

        except Exception as e:
            logger.error(f"Error processing {pair_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("SUMMARY - ALL PAIRS")
    logger.info("="*60)

    if all_results:
        summary_data = []
        for pair_name, result in all_results.items():
            metrics = result['metrics']
            summary_data.append({
                'Pair': pair_name,
                'Return (%)': f"{metrics['total_return']:.2f}",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD (%)': f"{metrics['max_drawdown_pct']:.2f}",
                'Trades': metrics['num_trades'],
                'Win Rate (%)': f"{metrics['win_rate']:.2f}"
            })

        # Print as table
        logger.info("\nPerformance Comparison:")
        for row in summary_data:
            logger.info(f"  {row['Pair']:15} | Return: {row['Return (%)']: >8} | "
                       f"Sharpe: {row['Sharpe']: >5} | MaxDD: {row['Max DD (%)']: >7} | "
                       f"Trades: {row['Trades']: >4} | WinRate: {row['Win Rate (%)']: >5}")

    logger.info("\n" + "="*60)
    logger.info("Backtest complete!")
    logger.info(f"Results saved to: ./results/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
