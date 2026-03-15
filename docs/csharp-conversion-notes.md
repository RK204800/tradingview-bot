# C# Conversion Notes for RSI Divergence Strategy
## For NinjaTrader / Cloud Code Implementation

---

## Strategy Overview

- **Name:** RSI Divergence Strategy
- **Instruments:** NQ (Nasdaq 100), ES (S&P 500) Futures
- **Timeframe:** 5-minute bars
- **Entry:** RSI bullish/bearish divergence + optional HTF trend filter
- **Exit:** 2× ATR stop loss, 4× R take profit

---

## Key Variables to Implement

### Input Parameters (NinjaScript Properties)
```csharp
[Parameter("RSI Length", DefaultValue = 14)]
public int RsiLength { get; set; }

[Parameter("ATR Length", DefaultValue = 14)]
public int AtrLength { get; set; }

[Parameter("Stop ATR Multiplier", DefaultValue = 2.0)]
public double StopMult { get; set; }

[Parameter("Target R Multiple", DefaultValue = 4.0)]
public double TargetMult { get; set; }

[Parameter("Pivot Lookback", DefaultValue = 3)]
public int PivotLookback { get; set; }

[Parameter("Use HTF Filter", DefaultValue = true)]
public bool UseHtfFilter { get; set; }

[Parameter("HTF Timeframe", DefaultValue = "15")]
public string HtfTimeframe { get; set; }
```

---

## Required Indicator Calculations

### 1. RSI
```csharp
// In OnStateChange() - State.DataLoaded
RSI = RSI(Close, RsiLength);
```

### 2. ATR
```csharp
ATR = ATR(AtrLength);
```

### 3. Pivot Points
```csharp
// Custom calculation needed - find local min/max over lookback period
public bool IsPivotLow(int bar, int lookback)
{
    double low = Low[bar];
    for (int i = 1; i <= lookback; i++)
    {
        if (Low[bar - i] < low || Low[bar + i] < low)
            return false;
    }
    return true;
}
```

### 4. Divergence Detection
```csharp
// Need to track previous pivot values
private double _lastPivotLowRSI;
private double _lastPivotLowPrice;
private double _lastPivotHighRSI;
private double _lastPivotHighPrice;

// Check in OnBarUpdate()
if (IsPivotLow(CurrentBar, PivotLookback))
{
    double currentRSI = RSI[0];
    double currentPrice = Low[0];
    
    if (!double.IsNaN(_lastPivotLowRSI) && 
        currentRSI > _lastPivotLowRSI && 
        currentPrice < _lastPivotLowPrice)
    {
        // Bullish divergence!
    }
    
    _lastPivotLowRSI = currentRSI;
    _lastPivotLowPrice = currentPrice;
}
```

### 5. Optional HTF Filter
```csharp
// For HTF, use BarsInProgress or request additional data
private Series<double> htfClose;
private Series<double> htfEMA;

// In OnStateChange() - State.DataLoaded
AddDataSeries("NQ 09-25", HtfTimeframe);
htfClose = DataSeries2;  // HTF close
htfEMA = EMA(htfClose, 20);
```

---

## Entry Logic

```csharp
protected override void OnBarUpdate()
{
    // Only calculate on primary series
    if (BarsInProgress != 0) return;
    
    if (CurrentBar < Math.Max(RsiLength, PivotLookback + 5)) return;
    
    bool bullishDiv = CheckBullishDivergence();
    bool bearishDiv = CheckBearishDivergence();
    
    // HTF filter
    bool trendUp = !UseHtfFilter || (htfClose[0] > htfEMA[0]);
    bool trendDown = !UseHtfFilter || (htfClose[0] < htfEMA[0]);
    
    // Entry signals
    bool longSignal = bullishDiv && trendUp;
    bool shortSignal = bearishDiv && trendDown;
    
    // Execute on signal
    if (longSignal && Position.MarketPosition == MarketPosition.Flat)
    {
        double entry = Close[0];
        double stop = entry - ATR[0] * StopMult;
        double target = entry + ATR[0] * StopMult * TargetMult;
        
        EnterLong("RSI Div Long", "Long", CalculateQuantity(entry - stop));
        SetStopLoss("Long", stop, false);
        SetProfitTarget("Long", target, false);
    }
    
    if (shortSignal && Position.MarketPosition == MarketPosition.Flat)
    {
        double entry = Close[0];
        double stop = entry + ATR[0] * StopMult;
        double target = entry - ATR[0] * StopMult * TargetMult;
        
        EnterShort("RSI Div Short", "Short", CalculateQuantity(stop - entry));
        SetStopLoss("Short", stop, false);
        SetProfitTarget("Short", target, false);
    }
}
```

---

## Position Sizing

```csharp
private double CalculateQuantity(double riskAmount)
{
    double tickValue = Instrument.MasterInstrument.PointValue / Instrument.MasterInstrument.TickSize;
    double riskPerContract = riskAmount * tickValue;
    return Account.BuyingPower * 0.01 / riskPerContract; // 1% risk
}
```

---

## Important Notes

### 1. Multi-Series Handling
- Need to handle HTF data request properly
- Use `AddDataSeries()` for HTF
- Access via `BarsArray[1]` for HTF data

### 2. Convergence Issues
- If HTF is different from primary, need to sync bars
- Use `BarsArray[1].GetTime()` to align

### 3. Performance
- Pivot calculation in OnBarUpdate is slow
- Consider pre-calculating or using Running Max/Min

### 4. Broker Settings
- NQ: $20/point, ES: $50/point
- Set proper commission (~$2-4 round trip)

---

## Testing Checklist

- [ ] Backtest on 2024-2026 NQ data
- [ ] Backtest on 2024-2026 ES data
- [ ] Verify win rate matches (~55%)
- [ ] Check profit factor (~1.4)
- [ ] Verify drawdown (<10%)
- [ ] Paper trade for 1 week
- [ ] Forward test in sim

---

## Contact

For questions about the strategy logic:
- Strategy creator: JetStream Trading
- GitHub: https://github.com/RK204800/tradingview-bot