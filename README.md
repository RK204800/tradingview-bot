# TradingView Bot - Automated Strategy Backtesting

## Overview
Scrape TradingView community indicators → Convert Pine scripts to Python → Backtest on BTC → Log results

## Process Flow
```
1. Scrape TradingView community indicators (Editor's Picks)
2. Extract Pine Script code
3. Convert Pine → Python using AI
4. Fetch BTC historical data (free API)
5. Run backtest with $1M capital
6. Log results to CSV
7. Push to GitHub
```

## Data Sources
- **Indicators:** TradingView Community Scripts
- **BTC Data:** Binance public API (free)
- **Results:** CSV in `/results`

## Output CSV Format
| Column | Description |
|--------|-------------|
| indicator_name | TradingView indicator name |
| python_filename | Converted Python filename |
| roi | Return on investment % |
| max_drawdown | Maximum drawdown % |
| sharpe | Sharpe ratio |
| sortino | Sortino ratio |
| expected_value | Expected value per trade |
| num_trades | Total number of trades |

## Usage
```bash
# Run backtest
python main.py

# Convert Pine script
python converters/pine_converter.py
```

## Requirements
- Python 3.8+
- requests
- pandas
- numpy
- ta (technical analysis)
