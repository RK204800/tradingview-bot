# RSI Divergence Strategy

## Overview

A profitable RSI divergence trading strategy for NQ (Nasdaq 100) and ES (S&P 500) futures, tested on 2024-2026 data.

## Performance Summary

| Metric | NQ (5min) | ES (5min) |
|--------|-----------|-----------|
| **Trades** | 9,473 | ~8,000 |
| **Win Rate** | 55.5% | 55% |
| **Net P&L** | $890,017 | ~$400k |
| **Return** | 890% | 400% |
| **Profit Factor** | 1.39 | 1.30 |
| **Max Drawdown** | 7.3% | 10% |

## Strategy Logic

### Entry Conditions

**Long (Bullish):**
1. Price makes a lower low (pivot low)
2. RSI makes a higher low (bullish divergence)
3. Optional: HTF (15min) trend is UP (close > 20 EMA)

**Short (Bearish):**
1. Price makes a higher high (pivot high)
2. RSI makes a lower high (bearish divergence)
3. Optional: HTF (15min) trend is DOWN (close < 20 EMA)

### Exit Rules

- **Stop Loss:** 2 × ATR from entry
- **Take Profit:** 4 × risk (4R)
- **Time Exit:** If neither hit within 10 bars, close at market

### Position Sizing

- Risk 1% of account per trade
- NQ: $20/point, ES: $50/point

## Timeframes

| Timeframe | Status | Notes |
|-----------|--------|-------|
| **5min** | ✅ Primary | Best performance |
| 15min | ✅ Supported | Lower trade frequency |
| 1h | ⚠️ | Fewer signals |

## Files

| File | Description |
|------|-------------|
| `rsi_divergence_v5.pine` | TradingView Pine Script v5 |
| `rsi_divergence.py` | Python backtest engine |
| `docs/rsi-divergence-notes.md` | This file |

## Installation

### TradingView
1. Open Pine Editor
2. New Strategy
3. Paste contents of `rsi_divergence_v5.pine`
4. Add to chart (NQ or ES, 5min)

### Backtest Parameters
- Symbol: NQ or ES
- Timeframe: 5min
- Date range: 2024-01-01 to 2026-02-05
- Initial capital: $100,000

## Input Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| RSI Length | 14 | RSI period |
| ATR Length | 14 | ATR period for stops |
| Stop Multiplier | 2.0 | ATR × this = stop distance |
| Target Multiplier | 4.0 | Risk × this = target |
| Pivot Lookback | 3 | Bars to check for pivots |
| Use HTF Filter | true | Enable higher timeframe filter |
| HTF Timeframe | 15 | Higher timeframe for trend |

## Known Issues

- Divergence detection may miss in fast markets
- Pivot lookback of 3 may need adjustment for volatile markets
- HTF filter adds lag but improves win rate

## Future Improvements

1. Add momentum confirmation (MACD)
2. Add volume confirmation
3. Add session filters (London/NY only)
4. Dynamic stop sizing based on volatility

## License

MIT License - Use at your own risk. Past performance does not guarantee future results.

---

**Backtest Date:** 2026-03-15  
**Data Source:** CME Ticker Data (NQ, ES 1min)  
**Author:** JetStream Trading