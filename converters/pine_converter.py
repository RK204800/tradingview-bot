#!/usr/bin/env python3
"""
Pine Script to Python Converter
Uses AI to convert TradingView Pine Script to Python backtestable code
"""

import os
import re
import json
from pathlib import Path

# Pine to Python mapping
INDICATOR_MAPPINGS = {
    # Moving Averages
    "sma": "ta.sma",
    "ema": "ta.ema",
    "rma": "ta.rma",
    "wma": "ta.wma",
    "vwma": "ta.vwma",
    "hma": "ta.hma",
    "dema": "ta.dema",
    "tema": "ta.tema",
    "trix": "ta.trix",
    
    # Oscillators
    "rsi": "ta.rsi",
    "stoch": "ta.stoch",
    "macd": "ta.macd",
    "cci": "ta.cci",
    "willr": "ta.willr",
    "adx": "ta.adx",
    "atr": "ta.atr",
    "ao": "ta.ao",  # Awesome Oscillator
    "mom": "ta.mom",  # Momentum
    "roc": "ta.roc",  # Rate of Change
    
    # Bollinger Bands
    "bb": "ta.bbands",
    
    # Volume
    "obv": "ta.obv",
    "mfi": "ta.mfi",
    
    # Custom
    "pivot": "pivot_points",
    "supres": "support_resistance",
}

def convert_pine_to_python(pine_code, indicator_name):
    """
    Convert Pine Script to Python backtest code
    
    Args:
        pine_code: Original Pine Script source
        indicator_name: Name of the indicator
    
    Returns:
        Python code as string
    """
    lines = pine_code.split("\n")
    python_lines = []
    
    # Header
    python_lines.append("# Auto-converted from Pine Script")
    python_lines.append(f"# Original: {indicator_name}")
    python_lines.append("")
    python_lines.append("import pandas as pd")
    python_lines.append("import numpy as np")
    python_lines.append("import ta")
    python_lines.append("from ta.volatility import BollingerBands")
    python_lines.append("from ta.trend import MACD, SMAIndicator, EMAIndicator")
    python_lines.append("from ta.momentum import RSIIndicator")
    python_lines.append("")
    python_lines.append("")
    python_lines.append("def calculate_indicators(df):")
    python_lines.append("    \"\"\"Calculate indicators from OHLCV data\"\"\"")
    python_lines.append("    result = df.copy()")
    
    # Simple conversion - detect common patterns
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith("//"):
            continue
            
        # Detect indicator calls
        for pine_func, python_module in INDICATOR_MAPPINGS.items():
            if f"{pine_func}(" in line.lower():
                # Extract parameters
                params_match = re.search(r'\((.*?)\)', line)
                if params_match:
                    params = params_match.group(1)
                    # Convert Pine source to 'close'
                    params = params.replace("close", "df['close']")
                    params = params.replace("high", "df['high']")
                    params = params.replace("low", "df['low']")
                    params = params.replace("open", "df['open']")
                    params = params.replace("volume", "df['volume']")
                    
                    python_lines.append(f"    result['{pine_func}'] = {python_module}({params})")
    
    python_lines.append("    return result")
    python_lines.append("")
    python_lines.append("")
    python_lines.append("def generate_signals(df):")
    python_lines.append("    \"\"\"Generate buy/sell signals\"\"\"")
    python_lines.append("    signals = pd.Series(index=df.index)")
    python_lines.append("    signals[:] = 0  # 0 = hold")
    python_lines.append("    # Add signal logic here")
    python_lines.append("    return signals")
    
    return "\n".join(python_lines)

def save_converted_code(python_code, output_dir="converters"):
    """Save converted Python code to file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{output_dir}/converted_strategy.py"
    with open(filename, "w") as f:
        f.write(python_code)
    
    print(f"Saved converted code to {filename}")
    return filename

def convert_with_ai(pine_code, indicator_name):
    """
    Use AI to convert Pine Script to Python
    This is more accurate than regex mapping
    
    Returns a prompt for the AI to do the conversion
    """
    prompt = f"""Convert this TradingView Pine Script to Python backtest code:

Original Pine Script ({indicator_name}):
```
{pine_code}
```

Requirements:
1. Use pandas and numpy for data handling
2. Use 'ta' library for technical indicators
3. Create a backtest function that calculates:
   - Entry/exit signals
   - P&L for each trade
   - Total ROI
   - Max drawdown
   - Sharpe ratio
   - Sortino ratio
4. Return results as a dictionary

Write clean, commented Python code.
"""
    return prompt

def main():
    """Test the converter"""
    sample_pine = """
//@version=5
indicator("Sample SMA", overlay=true)
sma20 = ta.sma(close, 20)
plot(sma20, color=color.blue)
"""
    
    result = convert_pine_to_python(sample_pine, "Sample SMA")
    print(result)

if __name__ == "__main__":
    main()
