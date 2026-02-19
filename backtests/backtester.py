#!/usr/bin/env python3
"""
Backtester for TradingView Strategies
Fetches BTC data and runs backtests on converted Python strategies
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Binance API (free, no auth needed for public data)
BINANCE_API = "https://api.binance.com/api/v3"

# Backtest parameters
INITIAL_CAPITAL = 1_000_000  # $1M since BTC > $100k
COMMISSION = 0.001  # 0.1% trading fee

def fetch_btc_data(symbol="BTCUSDT", interval="1h", days=365):
    """
    Fetch historical BTC data from Binance
    
    Args:
        symbol: Trading pair (default BTCUSDT)
        interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
        days: Number of days of history
    
    Returns:
        DataFrame with OHLCV data
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": 1000  # Max candles per request
    }
    
    try:
        response = requests.get(f"{BINANCE_API}/klines", params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        
        # Convert to numeric and create datetime
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        print(f"Fetched {len(df)} candles for {symbol}")
        return df[["open", "high", "low", "close", "volume"]]
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def run_backtest(df, signals, initial_capital=INITIAL_CAPITAL, commission=COMMISSION):
    """
    Run backtest on strategy signals
    
    Args:
        df: OHLCV DataFrame
        signals: Series of 1 (buy), -1 (sell), 0 (hold)
        initial_capital: Starting capital
        commission: Trading commission as decimal
    
    Returns:
        Dictionary with backtest results
    """
    capital = initial_capital
    position = 0  # Number of BTC held
    position_entry = 0
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        price = df["close"].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Buy signal
        if signal == 1 and position == 0:
            # Buy BTC with all capital
            position = (capital * (1 - commission)) / price
            position_entry = price
            capital = 0
            trades.append({
                "entry_time": df.index[i],
                "entry_price": price,
                "type": "BUY"
            })
        
        # Sell signal
        elif signal == -1 and position > 0:
            # Sell all BTC
            capital = position * price * (1 - commission)
            trades.append({
                "exit_time": df.index[i],
                "exit_price": price,
                "pnl": (price - position_entry) * position,
                "roi": ((price - position_entry) / position_entry) * 100,
                "type": "SELL"
            })
            position = 0
        
        # Track equity
        equity = capital + (position * price)
        equity_curve.append(equity)
    
    # Calculate final metrics
    final_equity = capital + (position * df["close"].iloc[-1] if position > 0 else 0)
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Calculate drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdowns = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdowns.min()
    
    # Calculate Sharpe (assuming 0% risk-free rate)
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Calculate Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Win rate
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    
    # Expected value per trade
    avg_pnl = np.mean([t.get("pnl", 0) for t in trades]) if trades else 0
    
    results = {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "num_trades": len(trades),
        "win_rate_pct": win_rate,
        "expected_value": avg_pnl,
    }
    
    return results

def save_results(results, indicator_name, output_dir="results"):
    """Save backtest results to CSV"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = f"{output_dir}/backtest_results.csv"
    
    # Create or append to CSV
    row = {
        "indicator_name": indicator_name,
        "python_filename": f"{indicator_name.replace(' ', '_').lower()}.py",
        "roi": results["total_return_pct"],
        "drawdown": results["max_drawdown_pct"],
        "sharpe": results["sharpe_ratio"],
        "sortino": results["sortino_ratio"],
        "expected_value": results["expected_value"],
        "num_trades": results["num_trades"],
        "timestamp": datetime.now().isoformat(),
    }
    
    # Read existing or create new
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return row

def main():
    """Test the backtester"""
    print("Fetching BTC data...")
    df = fetch_btc_data()
    
    if df is not None:
        # Generate dummy signals (alternating every 24 hours)
        signals = pd.Series(0, index=df.index)
        signals.iloc[::48] = 1   # Buy every 48 hours
        signals.iloc[24::48] = -1  # Sell 24 hours later
        
        print("Running backtest...")
        results = run_backtest(df, signals)
        
        print("\n=== Backtest Results ===")
        print(f"ROI: {results['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"Trades: {results['num_trades']}")
        
        save_results(results, "test_indicator")

if __name__ == "__main__":
    main()
