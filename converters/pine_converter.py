#!/usr/bin/env python3
"""
Enhanced Pine Script to Python Converter
Advanced conversion with proper parsing and multiple strategy templates
"""

import os
import re
import json
from pathlib import Path

# Pine to Python indicator mappings
INDICATOR_MAPPINGS = {
    # Moving Averages
    "sma": ("ta.sma", "close", "period"),
    "ema": ("ta.ema", "close", "period"),
    "rma": ("ta.ema", "close", "period"),  # RMA is similar to EMA
    "wma": ("ta.wma", "close", "period"),
    "vwma": ("ta.volume_weighted_avg_price", "close", "period"),
    "hma": ("ta.hull", "close", "period"),
    "dema": ("ta.dema", "close", "period"),
    "tema": ("ta.tema", "close", "period"),
    "trix": ("ta.trix", "close", "period"),
    
    # Oscillators
    "rsi": ("ta.rsi", "close", "period"),
    "stoch": ("ta.stoch", "high,low,close", "k,d"),
    "macd": ("ta.macd", "close", "fast,signal,slow"),
    "cci": ("ta.cci", "high,low,close", "period"),
    "williams_r": ("ta.williams_r", "high,low,close", "period"),
    "adx": ("ta.adx", "high,low,close", "period"),
    "atr": ("ta.atr", "high,low,close", "period"),
    "ao": ("ta.ao", "high,low", ""),  # Awesome Oscillator
    "mom": ("ta.momentum", "close", "period"),
    "roc": ("ta.roc", "close", "period"),
    "stochrsi": ("ta.stochrsi", "close", "period"),
    
    # Bollinger Bands
    "bb": ("BollingerBands", "close", "period, std_dev"),
    "bbands": ("BollingerBands", "close", "period, std_dev"),
    
    # Volume
    "obv": ("ta.obv", "close", ""),
    "mfi": ("ta.mfi", "high,low,close,volume", "period"),
    
    # Ichimoku
    "ichimoku": ("ichimoku", "high,low", "conversion,base,span_b"),
    
    # VWAP
    "vwap": ("ta.vwap", "close", ""),
    
    # Custom
    "pivot": ("pivot_points", "high,low", "type"),
    "supres": ("support_resistance", "high,low", ""),
}


def parse_pine_version(code):
    """Detect Pine Script version"""
    match = re.search(r'@version=(\d+)', code)
    return int(match.group(1)) if match else 4


def parse_pine_indicators(code):
    """Extract all indicators from Pine Script"""
    indicators = []
    
    # Find indicator declarations
    patterns = [
        r'indicator\s*\(\s*"([^"]+)"',  # indicator("name")
        r'strategy\s*\(\s*"([^"]+)"',   # strategy("name")
        r'plot\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)',  # plot(var)
        r'plotshape\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)',  # plotshape(var)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            indicators.append(match)
    
    return indicators


def parse_ta_function(call):
    """Parse a ta.* function call and extract parameters"""
    # Extract function name and parameters
    match = re.match(r'ta\.(\w+)\s*\((.*)\)', call.strip())
    if not match:
        return None
    
    func_name = match.group(1)
    params_str = match.group(2)
    
    # Parse parameters
    params = []
    depth = 0
    current = ""
    for char in params_str:
        if char in '([':
            depth += 1
        elif char in ')]':
            depth -= 1
        if char == ',' and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += char
    if current:
        params.append(current.strip())
    
    return {
        'function': func_name,
        'params': params
    }


def convert_source_to_series(source):
    """Convert Pine source (close, high, low, etc.) to Python"""
    source = source.strip().lower()
    
    mappings = {
        'close': "df['close']",
        'open': "df['open']",
        'high': "df['high']",
        'low': "df['low']",
        'volume': "df['volume']",
        'hl2': "(df['high'] + df['low']) / 2",
        'hlc3': "(df['high'] + df['low'] + df['close']) / 3",
        'ohlc4': "(df['open'] + df['high'] + df['low'] + df['close']) / 4",
    }
    
    return mappings.get(source, source)


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
    python_lines.append("#!/usr/bin/env python3")
    python_lines.append(f"# Auto-converted from Pine Script")
    python_lines.append(f"# Original: {indicator_name}")
    python_lines.append("")
    python_lines.append("import pandas as pd")
    python_lines.append("import numpy as np")
    python_lines.append("import ta")
    python_lines.append("from ta.volatility import BollingerBands")
    python_lines.append("from ta.trend import MACD, SMAIndicator, EMAIndicator")
    python_lines.append("from ta.momentum import RSIIndicator")
    python_lines.append("")
    
    # Detect version
    version = parse_pine_version(pine_code)
    python_lines.append(f"# Pine Script version: {version}")
    python_lines.append("")
    
    # Extract indicators
    indicators = parse_pine_indicators(pine_code)
    python_lines.append(f"# Detected indicators: {indicators}")
    python_lines.append("")
    
    # Helper functions
    python_lines.append("def calculate_indicators(df):")
    python_lines.append('    """Calculate indicators from OHLCV data"""')
    python_lines.append("    result = df.copy()")
    
    # Parse ta.* functions
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith("//"):
            continue
        
        # Look for ta.* function calls
        if 'ta.' in line.lower():
            # Try to parse as indicator calculation
            parsed = parse_ta_function(line)
            if parsed:
                func = parsed['function']
                
                # Generate appropriate Python code
                if func == 'sma':
                    period = parsed['params'][1] if len(parsed['params']) > 1 else '20'
                    python_lines.append(f"    result['sma_{period}'] = ta.sma(result['close'], {period})")
                
                elif func == 'ema':
                    period = parsed['params'][1] if len(parsed['params']) > 1 else '12'
                    python_lines.append(f"    result['ema_{period}'] = ta.ema(result['close'], {period})")
                
                elif func == 'rsi':
                    period = parsed['params'][1] if len(parsed['params']) > 1 else '14'
                    python_lines.append(f"    result['rsi_{period}'] = ta.rsi(result['close'], {period})")
                
                elif func == 'macd':
                    python_lines.append("    result['macd'] = ta.macd(result['close'])")
                
                elif func == 'atr':
                    period = parsed['params'][1] if len(parsed['params']) > 1 else '14'
                    python_lines.append(f"    result['atr_{period}'] = ta.atr(result['high'], result['low'], result['close'], {period})")
    
    python_lines.append("    return result")
    python_lines.append("")
    
    # Signal generation
    python_lines.append("")
    python_lines.append("def generate_signals(df):")
    python_lines.append('    """Generate buy/sell signals based on indicators"""')
    python_lines.append("    signals = pd.Series(0, index=df.index)")
    python_lines.append("    ")
    python_lines.append("    # Add your signal logic here")
    python_lines.append("    # signals[df['indicator'] > threshold] = 1")
    python_lines.append("    # signals[df['indicator'] < threshold] = -1")
    python_lines.append("    ")
    python_lines.append("    return signals")
    
    return "\n".join(python_lines)


def generate_strategy_template(strategy_type, indicator_name):
    """Generate a complete strategy template"""
    
    templates = {
        'sma_crossover': '''#!/usr/bin/env python3
"""
SMA Crossover Strategy
Generated from: {name}
"""

import pandas as pd
import numpy as np
import ta
from backtests.backtester import run_backtest, fetch_btc_data

def calculate_indicators(df):
    df = df.copy()
    df['sma_fast'] = ta.sma(df['close'], 20)
    df['sma_slow'] = ta.sma(df['close'], 50)
    return df

def generate_signals(df):
    signals = pd.Series(0, index=df.index)
    
    # Buy: fast SMA crosses above slow SMA
    buy = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
    signals[buy] = 1
    
    # Sell: fast SMA crosses below slow SMA
    sell = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))
    signals[sell] = -1
    
    return signals

def run():
    df = fetch_btc_data('BTCUSDT', '1h', 365)
    df = calculate_indicators(df)
    signals = generate_signals(df)
    results = run_backtest(df, signals, initial_capital=1_000_000)
    print(f"ROI: {{results['total_return_pct']:+.2f}}%")
    return results

if __name__ == "__main__":
    run()
'''.format(name=indicator_name),

        'rsi': '''#!/usr/bin/env python3
"""
RSI Strategy
Generated from: {name}
"""

import pandas as pd
import numpy as np
import ta
from backtests.backtester import run_backtest, fetch_btc_data

def calculate_indicators(df):
    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], 14)
    return df

def generate_signals(df, oversold=30, overbought=70):
    signals = pd.Series(0, index=df.index)
    
    # Buy: RSI crosses above oversold
    buy = (df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold)
    signals[buy] = 1
    
    # Sell: RSI crosses below overbought
    sell = (df['rsi'] < overbought) & (df['rsi'].shift(1) >= overbought)
    signals[sell] = -1
    
    return signals

def run():
    df = fetch_btc_data('BTCUSDT', '1h', 365)
    df = calculate_indicators(df)
    signals = generate_signals(df)
    results = run_backtest(df, signals, initial_capital=1_000_000)
    print(f"ROI: {{results['total_return_pct']:+.2f}}%")
    return results

if __name__ == "__main__":
    run()
'''.format(name=indicator_name),

        'macd': '''#!/usr/bin/env python3
"""
MACD Strategy
Generated from: {name}
"""

import pandas as pd
import numpy as np
import ta
from backtests.backtester import run_backtest, fetch_btc_data

def calculate_indicators(df):
    df = df.copy()
    macd = ta.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df

def generate_signals(df):
    signals = pd.Series(0, index=df.index)
    
    # Buy: MACD crosses above signal
    buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    signals[buy] = 1
    
    # Sell: MACD crosses below signal
    sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    signals[sell] = -1
    
    return signals

def run():
    df = fetch_btc_data('BTCUSDT', '1h', 365)
    df = calculate_indicators(df)
    signals = generate_signals(df)
    results = run_backtest(df, signals, initial_capital=1_000_000)
    print(f"ROI: {{results['total_return_pct']:+.2f}}%")
    return results

if __name__ == "__main__":
    run()
'''.format(name=indicator_name),

        'bollinger': '''#!/usr/bin/env python3
"""
Bollinger Bands Strategy
Generated from: {name}
"""

import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands
from backtests.backtester import run_backtest, fetch_btc_data

def calculate_indicators(df):
    df = df.copy()
    bb = BollingerBands(df['close'], 20, 2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    return df

def generate_signals(df):
    signals = pd.Series(0, index=df.index)
    
    # Buy: Price touches lower band
    buy = (df['close'] <= df['bb_lower']) & (df['close'].shift(1) > df['bb_lower'].shift(1))
    signals[buy] = 1
    
    # Sell: Price touches upper band
    sell = (df['close'] >= df['bb_upper']) & (df['close'].shift(1) < df['bb_upper'].shift(1))
    signals[sell] = -1
    
    return signals

def run():
    df = fetch_btc_data('BTCUSDT', '1h', 365)
    df = calculate_indicators(df)
    signals = generate_signals(df)
    results = run_backtest(df, signals, initial_capital=1_000_000)
    print(f"ROI: {{results['total_return_pct']:+.2f}}%")
    return results

if __name__ == "__main__":
    run()
'''.format(name=indicator_name),
    }
    
    return templates.get(strategy_type, "# Strategy template not found")


def save_converted_code(python_code, filename, output_dir="converters"):
    """Save converted Python code to file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = f"{output_dir}/{filename}"
    with open(filepath, "w") as f:
        f.write(python_code)
    
    print(f"Saved converted code to {filepath}")
    return filepath


def convert_with_ai_prompt(pine_code, indicator_name, target_strategy='sma_crossover'):
    """
    Generate AI prompt for Pine Script conversion
    """
    prompt = f"""Convert this TradingView Pine Script to a complete Python backtest:

Original Indicator: {indicator_name}

```pine
{pine_code}
```

Instructions:
1. Use pandas for data handling
2. Use 'ta' library for technical indicators
3. Generate signals: 1 = buy, -1 = sell, 0 = hold
4. Use the backtest function to calculate ROI, drawdown, Sharpe
5. Include proper error handling

Strategy Type: {target_strategy}

Return ONLY valid Python code, no explanations.
"""
    return prompt


def analyze_pine_script(pine_code):
    """Analyze Pine Script and suggest appropriate strategy"""
    code_lower = pine_code.lower()
    
    # Check for indicators present
    has_sma = 'sma(' in code_lower or 'ta.sma' in code_lower
    has_ema = 'ema(' in code_lower or 'ta.ema' in code_lower
    has_rsi = 'rsi(' in code_lower or 'ta.rsi' in code_lower
    has_macd = 'macd(' in code_lower or 'ta.macd' in code_lower
    has_bb = 'bb(' in code_lower or 'bollinger' in code_lower
    has_volume = 'volume' in code_lower
    
    suggestions = []
    if has_sma:
        suggestions.append('sma_crossover')
    if has_rsi:
        suggestions.append('rsi')
    if has_macd:
        suggestions.append('macd')
    if has_bb:
        suggestions.append('bollinger')
    
    if not suggestions:
        suggestions = ['sma_crossover']  # Default
    
    return suggestions


def main():
    """Test the converter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pine Script Converter")
    parser.add_argument("--pine", type=str, help="Pine Script code")
    parser.add_argument("--name", type=str, default="Unknown", help="Indicator name")
    parser.add_argument("--strategy", type=str, help="Target strategy type")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--analyze", action="store_true", help="Analyze and suggest strategies")
    args = parser.parse_args()
    
    if args.analyze and args.pine:
        suggestions = analyze_pine_script(args.pine)
        print(f"Suggested strategies: {suggestions}")
        return
    
    if args.pine:
        if args.strategy:
            # Generate template
            code = generate_strategy_template(args.strategy, args.name)
        else:
            # Auto-convert
            code = convert_pine_to_python(args.pine, args.name)
        
        print(code)
        
        if args.output:
            save_converted_code(code, args.output)
    else:
        # Demo
        sample_pine = """
//@version=5
indicator("My SMA", overlay=true)
sma20 = ta.sma(close, 20)
sma50 = ta.sma(close, 50)
plot(sma20, color=color.blue)
plot(sma50, color=color.red)

// Crossover signal
bull = ta.crossover(sma20, smna50)
bear = ta.crossunder(sma20, smna50)
plotshape(bull, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(bear, "Sell", shape.triangledown, location.abovebar, color.red)
"""
        print("Sample conversion:")
        print(convert_pine_to_python(sample_pine, "My SMA"))


if __name__ == "__main__":
    main()
