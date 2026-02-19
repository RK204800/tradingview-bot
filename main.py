#!/usr/bin/env python3
"""
TradingView Bot - Main Orchestrator
Coordinates scraping, conversion, and backtesting
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrapers.tv_scraper import get_indicators, save_indicators
from converters.pine_converter import convert_pine_to_python
from backtests.backtester import fetch_btc_data, run_backtest, save_results

# Configuration
CONFIG = {
    "capital": 1_000_000,  # $1M
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1h",
    "max_indicators": 100,
}

def load_indicators():
    """Load cached indicators or scrape new ones"""
    indicators_file = "data/indicators.json"
    
    if os.path.exists(indicators_file):
        with open(indicators_file, "r") as f:
            indicators = json.load(f)
        print(f"Loaded {len(indicators)} indicators from cache")
        return indicators
    else:
        print("Scraping new indicators...")
        return get_indicators()

def process_indicator(indicator):
    """
    Process a single indicator:
    1. Convert Pine to Python
    2. Run backtest
    3. Save results
    """
    indicator_name = indicator.get("name", "Unknown")
    print(f"\n{'='*50}")
    print(f"Processing: {indicator_name}")
    print(f"{'='*50}")
    
    # For now, generate test signals
    # In production, this would use the converted Pine code
    try:
        # Fetch BTC data (cache it)
        df_file = "data/btc_data.csv"
        if os.path.exists(df_file):
            import pandas as pd
            df = pd.read_csv(df_file, index_col=0, parse_dates=True)
            print(f"Loaded BTC data from cache: {len(df)} candles")
        else:
            print("Fetching BTC data from Binance...")
            df = fetch_btc_data(
                symbol=CONFIG["symbol"],
                interval=CONFIG["interval"],
                days=365
            )
            if df is not None:
                df.to_csv(df_file)
                print(f"Cached BTC data")
        
        if df is None:
            print("Failed to get BTC data, skipping")
            return None
        
        # Generate simple test signals (SMA crossover)
        import pandas as pd
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[df["sma_20"] > df["sma_50"]] = 1   # Buy signal
        signals[df["sma_20"] <= df["sma_50"]] = -1  # Sell signal
        
        # Run backtest
        results = run_backtest(df, signals, CONFIG["capital"])
        
        print(f"Results: ROI={results['total_return_pct']:.2f}%, "
              f"Drawdown={results['max_drawdown_pct']:.2f}%, "
              f"Sharpe={results['sharpe_ratio']:.2f}")
        
        # Save results
        save_results(results, indicator_name)
        
        return results
        
    except Exception as e:
        print(f"Error processing {indicator_name}: {e}")
        return None

def run_batch(limit=10):
    """Process a batch of indicators"""
    indicators = load_indicators()
    
    # Filter to get usable indicators
    usable = [ind for ind in indicators if ind.get("type") == "script|pine"]
    usable = usable[:limit]
    
    print(f"\nProcessing {len(usable)} indicators...")
    
    results = []
    for i, indicator in enumerate(usable):
        print(f"\n[{i+1}/{len(usable)}]")
        result = process_indicator(indicator)
        if result:
            results.append(result)
        
        # Rate limiting
        time.sleep(1)
    
    print(f"\n{'='*50}")
    print(f"Completed {len(results)} backtests")
    print(f"{'='*50}")
    
    # Summary
    if results:
        avg_roi = sum(r["total_return_pct"] for r in results) / len(results)
        best = max(results, key=lambda x: x["total_return_pct"])
        print(f"Average ROI: {avg_roi:.2f}%")
        print(f"Best: {best['indicator_name']} ({best['total_return_pct']:.2f}%)")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TradingView Bot")
    parser.add_argument("--limit", type=int, default=10, help="Number of indicators to process")
    parser.add_argument("--scrape", action="store_true", help="Force fresh scrape")
    args = parser.parse_args()
    
    if args.scrape:
        print("Fresh scrape...")
        indicators = get_indicators()
        save_indicators(indicators)
    
    run_batch(limit=args.limit)

if __name__ == "__main__":
    main()
