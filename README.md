# TradingView Bot - Automated Strategy Backtesting

Scrape TradingView community indicators → Convert Pine scripts to Python → Backtest on BTC → Log results

## Overview

This bot automates the process of testing TradingView indicators against historical BTC data:

1. **Scrape** TradingView community indicators (Editor's Picks)
2. **Convert** Pine Script to Python using pattern matching
3. **Backtest** on BTC data from Binance
4. **Analyze** results with multiple strategies and risk management

## Features

### Strategies Implemented
- **SMA Crossover** - Buy when fast SMA crosses above slow SMA
- **RSI** - Buy oversold (30), sell overbought (70)
- **MACD** - Buy when MACD crosses above signal line
- **Bollinger Bands** - Buy at lower band, sell at upper band
- **Bollinger Bounce** - Buy at lower band with confirmation
- **Combo MA+RSI** - Trend following with RSI filter
- **MACD+RSI** - MACD signals filtered by RSI
- **Triple MA** - Buy when all MAs align upward

### Risk Management
- Stop loss (2%, 3%, 5%)
- Take profit (4%, 6%, 10%)
- Position sizing (50%, 75%, 100%)

### Timeframes
- 15m, 1h, 4h, 1d

## Process Flow

```
1. Fetch BTC data from Binance API
2. Calculate technical indicators (RSI, MACD, BB, etc.)
3. Generate trading signals based on strategy
4. Run backtest with configurable risk parameters
5. Calculate metrics (ROI, Sharpe, Drawdown, Win Rate)
6. Save results to CSV
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### List available strategies
```bash
python main.py --list
```

### Test a single strategy
```bash
python main.py --strategy sma_crossover
```

### Test all strategies on 1h timeframe
```bash
python main.py
```

### Test across all timeframes
```bash
python main.py --timeframes
```

### Test risk management configurations
```bash
python main.py --strategy sma_crossover --risk-test
```

### Run comprehensive tests
```bash
python main.py --all
```

## Configuration

Edit `CONFIG` in `main.py`:

```python
CONFIG = {
    "capital": 1_000_000,  # $1M
    "symbol": "BTCUSDT",
    "interval": "1h",
}
```

## Results

### Latest Backtest Results (BTCUSDT, 180 days)

| Timeframe | Best Strategy | ROI | Drawdown | Sharpe | Trades |
|-----------|---------------|-----|----------|--------|--------|
| 15m | combo_ma_rsi | +0.10% | -1.76% | 0.03 | 13 |
| 1h | sma_crossover | **+5.54%** | -7.68% | 0.40 | 19 |
| 4h | sma_crossover | -8.18% | -23.38% | -0.25 | 20 |
| 1d | combo_ma_rsi | 0.00% | 0.00% | 0.00 | 0 |

### Risk Management Impact (SMA Crossover, 1h)

| Configuration | ROI | Drawdown | Sharpe | Trades |
|---------------|-----|----------|--------|--------|
| No RM | -9.86% | -22.53% | -0.32 | 17 |
| 2% Stop Loss | -6.66% | -19.78% | -0.26 | 17 |
| **2% SL + 4% TP** | **+0.51%** | -9.49% | 0.05 | 17 |
| 3% SL + 6% TP | -2.50% | -14.83% | -0.10 | 17 |

### Key Findings

1. **Best overall**: SMA Crossover on 1h timeframe (+5.54% ROI)
2. **Risk management helps**: Adding 2% SL + 4% TP improved results from -9.86% to +0.51%
3. **Shorter timeframes** tend to have lower drawdowns but more trades
4. **Most strategies lose money** - this confirms the difficulty of beating buy-and-hold

## Output CSV Format

| Column | Description |
|--------|-------------|
| indicator_name | TradingView indicator name |
| strategy | Strategy used |
| roi | Return on investment % |
| drawdown | Maximum drawdown % |
| sharpe | Sharpe ratio |
| sortino | Sortino ratio |
| expected_value | Expected value per trade |
| num_trades | Total number of trades |
| win_rate | Win rate % |
| stop_loss_triggers | Stop loss triggered (bool) |
| take_profit_triggers | Take profit triggered (bool) |
| timestamp | Test timestamp |

## Pine Script Converter

The converter (`converters/pine_converter.py`) provides:

- Pattern matching for common Pine Script functions
- Indicator mapping (SMA, EMA, RSI, MACD, Bollinger Bands)
- Strategy template generation
- AI prompt generation for LLM conversion

### Convert Pine Script
```bash
python converters/pine_converter.py --pine "// Pine code here" --name "My Indicator" --strategy sma_crossover
```

## Requirements

- Python 3.8+
- requests
- pandas
- numpy
- ta (technical analysis library)

## Disclaimer

**Past performance does not guarantee future results.** This tool is for educational and research purposes only. Always do your own research before making investment decisions.

## License

MIT License
