"""
Trading Pairs Configuration

Defines the trading pairs to be analyzed and backtested,
along with data fetching parameters.
"""

TRADING_PAIRS = [
    {
        'name': 'TRGP_EPD',
        'ticker1': 'TRGP',
        'ticker2': 'EPD',
        'sector': 'Energy/Midstream'
    },
    {
        'name': 'STNG_HAFN',
        'ticker1': 'STNG',
        'ticker2': 'HAFN',
        'sector': 'Shipping/Tankers'
    },
    {
        'name': 'SBLK_GNK',
        'ticker1': 'SBLK',
        'ticker2': 'GNK',
        'sector': 'Shipping/Dry Bulk'
    }
]

DATA_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'price_field': 'Adj Close',  # Use adjusted close for splits/dividends
    'cache_enabled': True,
    'cache_dir': './data/cache'
}
