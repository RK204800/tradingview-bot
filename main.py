#!/usr/bin/env python3
"""
TradingView Bot - Main Orchestrator
Coordinates scraping, conversion, and backtesting
Enhanced with multiple strategies and timeframes
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtests.backtester import (
    fetch_btc_data, calculate_indicators, run_strategy, run_all_strategies,
    STRATEGIES, save_results, test_timeframes, run_with_risk_management
)

# Configuration
CONFIG = {
    "capital": 1_000_000,  # $1M
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1h",
    "max_indicators": 100,
}


def run_strategy_test(strategy_name, interval="1h", days=365):
    """Run a single strategy test"""
    print(f"\n{'='*60}")
    print(f"Testing: {strategy_name} on {interval}")
    print(f"{'='*60}")
    
    # Fetch data
    df_file = f"data/btc_{interval}.csv"
    if os.path.exists(df_file):
        import pandas as pd
        df = pd.read_csv(df_file, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} candles from cache")
    else:
        print(f"Fetching {interval} data from Binance...")
        df = fetch_btc_data(
            symbol=CONFIG["symbol"],
            interval=interval,
            days=days
        )
        if df is not None:
            df.to_csv(df_file)
            print(f"Cached data to {df_file}")
    
    if df is None:
        print("Failed to get data")
        return None
    
    # Run strategy
    results = run_strategy(strategy_name, df, CONFIG["capital"])
    
    print(f"\nResults:")
    print(f"  ROI: {results['total_return_pct']:+.2f}%")
    print(f"  Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"  Sortino: {results['sortino_ratio']:.2f}")
    print(f"  Trades: {results['num_trades']}")
    print(f"  Win Rate: {results['win_rate_pct']:.1f}%")
    
    return results


def run_multi_strategy_test(interval="1h", days=365):
    """Run all strategies and find the best one"""
    print(f"\n{'='*60}")
    print(f"Running all strategies on {interval}")
    print(f"{'='*60}")
    
    # Fetch data
    df_file = f"data/btc_{interval}.csv"
    if os.path.exists(df_file):
        import pandas as pd
        df = pd.read_csv(df_file, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} candles from cache")
    else:
        print(f"Fetching {interval} data from Binance...")
        df = fetch_btc_data(
            symbol=CONFIG["symbol"],
            interval=interval,
            days=days
        )
        if df is not None:
            df.to_csv(df_file)
    
    if df is None:
        print("Failed to get data")
        return []
    
    # Run all strategies
    results = run_all_strategies(df, CONFIG["capital"])
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['total_return_pct'])
        print(f"\n{'='*60}")
        print(f"BEST STRATEGY: {best['strategy']}")
        print(f"ROI: {best['total_return_pct']:+.2f}%")
        print(f"Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"Drawdown: {best['max_drawdown_pct']:.2f}%")
        print(f"{'='*60}")
        
        # Save best result
        save_results(best, f"best_{interval}")
    
    return results


def run_timeframe_comparison():
    """Compare strategies across timeframes"""
    intervals = ["15m", "1h", "4h", "1d"]
    all_results = {}
    
    for interval in intervals:
        print(f"\n{'#'*60}")
        print(f"## Testing {interval} timeframe")
        print(f"{'#'*60}")
        
        results = run_multi_strategy_test(interval=interval, days=180)
        all_results[interval] = results
        
        time.sleep(1)  # Rate limit
    
    # Summary
    print(f"\n{'='*60}")
    print("TIMEFRAME COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for interval, results in all_results.items():
        if results:
            best = max(results, key=lambda x: x['total_return_pct'])
            print(f"{interval:5s}: {best['strategy']:20s} ROI: {best['total_return_pct']:+8.2f}%  Sharpe: {best['sharpe_ratio']:6.2f}")
    
    return all_results


def run_risk_management_test(strategy_name, interval="1h"):
    """Test strategy with different risk management configs"""
    df_file = f"data/btc_{interval}.csv"
    if os.path.exists(df_file):
        import pandas as pd
        df = pd.read_csv(df_file, index_col=0, parse_dates=True)
    else:
        df = fetch_btc_data(symbol=CONFIG["symbol"], interval=interval, days=365)
    
    if df is None:
        print("Failed to get data")
        return []
    
    return run_with_risk_management(df, strategy_name, CONFIG["capital"])


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TradingView Bot")
    parser.add_argument("--strategy", type=str, help="Single strategy to test (see --list)")
    parser.add_argument("--list", action="store_true", help="List available strategies")
    parser.add_argument("--interval", type=str, default="1h", help="Timeframe (15m, 1h, 4h, 1d)")
    parser.add_argument("--timeframes", action="store_true", help="Test all timeframes")
    parser.add_argument("--risk-test", action="store_true", help="Test risk management")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable strategies:")
        for i, name in enumerate(STRATEGIES.keys(), 1):
            print(f"  {i}. {name}")
        return
    
    if args.all:
        # Run comprehensive test
        print("Running comprehensive tests...")
        run_timeframe_comparison()
        
        # Test risk management on best strategies
        print("\n" + "="*60)
        print("RISK MANAGEMENT TESTS")
        print("="*60)
        
        for strategy in ['sma_crossover', 'macd', 'rsi']:
            print(f"\n--- {strategy} ---")
            run_risk_management_test(strategy, "1h")
    elif args.timeframes:
        run_timeframe_comparison()
    elif args.risk_test:
        if not args.strategy:
            print("Please specify --strategy for risk management test")
            return
        run_risk_management_test(args.strategy, args.interval)
    elif args.strategy:
        run_strategy_test(args.strategy, args.interval)
    else:
        # Default: run all strategies on 1h
        run_multi_strategy_test(args.interval)


if __name__ == "__main__":
    main()
