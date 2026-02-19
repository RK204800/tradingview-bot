#!/usr/bin/env python3
"""
Backtester for TradingView Strategies
Enhanced with multiple indicators, stop losses, and position sizing
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

# Default parameters
DEFAULT_CAPITAL = 1_000_000  # $1M since BTC > $100k
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
        
        print(f"Fetched {len(df)} candles for {symbol} ({interval})")
        return df[["open", "high", "low", "close", "volume"]]
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


# ==================== STRATEGIES ====================

def calculate_indicators(df):
    """Calculate all technical indicators"""
    result = df.copy()
    
    # Moving Averages
    result['sma_20'] = df['close'].rolling(20).mean()
    result['sma_50'] = df['close'].rolling(50).mean()
    result['sma_200'] = df['close'].rolling(200).mean()
    result['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    result['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # RSI (14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    result['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    result['macd'] = result['ema_12'] - result['ema_26']
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    # Bollinger Bands
    result['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    result['bb_upper'] = result['bb_middle'] + (bb_std * 2)
    result['bb_lower'] = result['bb_middle'] - (bb_std * 2)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    result['atr_14'] = true_range.rolling(14).mean()
    
    # Volume indicators
    result['volume_sma_20'] = df['volume'].rolling(20).mean()
    result['volume_ratio'] = df['volume'] / result['volume_sma_20']
    
    return result


def strategy_sma_crossover(df):
    """SMA Crossover Strategy - Buy when fast SMA crosses above slow SMA"""
    signals = pd.Series(0, index=df.index)
    
    # Buy: fast SMA crosses above slow SMA
    buy_signal = (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
    # Sell: fast SMA crosses below slow SMA
    sell_signal = (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_rsi(df, oversold=30, overbought=70):
    """RSI Strategy - Buy oversold, sell overbought"""
    signals = pd.Series(0, index=df.index)
    
    # Buy when RSI crosses above oversold level
    buy_signal = (df['rsi_14'] > oversold) & (df['rsi_14'].shift(1) <= oversold)
    # Sell when RSI crosses below overbought level
    sell_signal = (df['rsi_14'] < overbought) & (df['rsi_14'].shift(1) >= overbought)
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_macd(df):
    """MACD Strategy - Buy when MACD crosses above signal, sell when crosses below"""
    signals = pd.Series(0, index=df.index)
    
    # Buy: MACD crosses above signal line
    buy_signal = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    # Sell: MACD crosses below signal line
    sell_signal = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_bollinger_bands(df):
    """Bollinger Bands Strategy - Buy at lower band, sell at upper band"""
    signals = pd.Series(0, index=df.index)
    
    # Buy when price touches lower band
    buy_signal = (df['close'] <= df['bb_lower']) & (df['close'].shift(1) > df['bb_lower'].shift(1))
    # Sell when price touches upper band
    sell_signal = (df['close'] >= df['bb_upper']) & (df['close'].shift(1) < df['bb_upper'].shift(1))
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_bollinger_bounce(df):
    """Bollinger Bands Bounce - Buy at lower band with confirmation, sell at middle"""
    signals = pd.Series(0, index=df.index)
    
    # Buy when price bounces off lower band (close > lower band after touching)
    buy_signal = (df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1))
    # Sell when price reaches middle band
    sell_signal = (df['close'] >= df['bb_middle']) & (df['close'].shift(1) < df['bb_middle'].shift(1))
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_combo_ma_rsi(df):
    """Combined MA + RSI Strategy - Trend following with RSI filter"""
    signals = pd.Series(0, index=df.index)
    
    # Uptrend: price above 200 SMA
    uptrend = df['close'] > df['sma_200']
    
    # Buy: RSI oversold + price above 200 SMA
    buy_signal = uptrend & (df['rsi_14'] < 35) & (df['rsi_14'].shift(1) >= 35)
    # Sell: RSI overbought
    sell_signal = (df['rsi_14'] > 65) & (df['rsi_14'].shift(1) <= 65)
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_macd_rsi(df):
    """MACD + RSI Combo Strategy"""
    signals = pd.Series(0, index=df.index)
    
    # Buy: MACD bullish cross + RSI not overbought
    macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    buy_signal = macd_bullish & (df['rsi_14'] < 60)
    
    # Sell: MACD bearish cross + RSI not oversold
    macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    sell_signal = macd_bearish & (df['rsi_14'] > 40)
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


def strategy_triple_ma(df):
    """Triple Moving Average - Buy when all MAs align upward"""
    signals = pd.Series(0, index=df.index)
    
    # All MAs in ascending order
    ma_aligned_up = (df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])
    ma_were_aligned_down = (df['sma_20'].shift(1) <= df['sma_50'].shift(1)) | (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
    
    # Buy when alignment forms
    buy_signal = ma_aligned_up & ma_were_aligned_down
    
    # Sell when 20 SMA crosses below 50 SMA
    sell_signal = (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals


STRATEGIES = {
    'sma_crossover': strategy_sma_crossover,
    'rsi': strategy_rsi,
    'macd': strategy_macd,
    'bollinger_bands': strategy_bollinger_bands,
    'bollinger_bounce': strategy_bollinger_bounce,
    'combo_ma_rsi': strategy_combo_ma_rsi,
    'macd_rsi': strategy_macd_rsi,
    'triple_ma': strategy_triple_ma,
}


# ==================== BACKTEST ENGINE ====================

def run_backtest(df, signals, initial_capital=DEFAULT_CAPITAL, commission=COMMISSION,
                 stop_loss=None, take_profit=None, position_size_pct=1.0):
    """
    Enhanced backtest with stop loss, take profit, and position sizing
    
    Args:
        df: OHLCV DataFrame
        signals: Series of 1 (buy), -1 (sell), 0 (hold)
        initial_capital: Starting capital
        commission: Trading commission as decimal
        stop_loss: Stop loss percentage (e.g., 0.02 for 2%)
        take_profit: Take profit percentage (e.g., 0.05 for 5%)
        position_size_pct: Percentage of capital to use (0.0 to 1.0)
    
    Returns:
        Dictionary with backtest results
    """
    capital = initial_capital
    position = 0  # Number of BTC held
    position_entry = 0
    trades = []
    equity_curve = []
    stop_triggered = False
    tp_triggered = False
    
    for i in range(len(df)):
        price = df["close"].iloc[i]
        high_price = df["high"].iloc[i]  # For stop loss check
        low_price = df["low"].iloc[i]   # For stop loss check
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Check stop loss (using intrabar high/low for accuracy)
        if position > 0 and stop_loss:
            stop_price = position_entry * (1 - stop_loss)
            if low_price <= stop_price:
                # Stop loss triggered
                capital = position * stop_price * (1 - commission)
                trades.append({
                    "exit_time": df.index[i],
                    "exit_price": stop_price,
                    "pnl": (stop_price - position_entry) * position,
                    "roi": ((stop_price - position_entry) / position_entry) * 100,
                    "type": "STOP_LOSS"
                })
                position = 0
                stop_triggered = True
                continue
        
        # Check take profit
        if position > 0 and take_profit:
            tp_price = position_entry * (1 + take_profit)
            if high_price >= tp_price:
                # Take profit triggered
                capital = position * tp_price * (1 - commission)
                trades.append({
                    "exit_time": df.index[i],
                    "exit_price": tp_price,
                    "pnl": (tp_price - position_entry) * position,
                    "roi": ((tp_price - position_entry) / position_entry) * 100,
                    "type": "TAKE_PROFIT"
                })
                position = 0
                tp_triggered = True
                continue
        
        # Buy signal
        if signal == 1 and position == 0:
            # Use position_size_pct of capital
            trade_capital = capital * position_size_pct
            position = (trade_capital * (1 - commission)) / price
            position_entry = price
            capital = capital - trade_capital
            trades.append({
                "entry_time": df.index[i],
                "entry_price": price,
                "type": "BUY"
            })
        
        # Sell signal
        elif signal == -1 and position > 0:
            capital = capital + (position * price * (1 - commission))
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
        "stop_loss_triggered": stop_triggered,
        "take_profit_triggered": tp_triggered,
    }
    
    return results


def run_strategy(strategy_name, df, initial_capital=DEFAULT_CAPITAL, 
                 stop_loss=None, take_profit=None, position_size_pct=1.0):
    """Run a specific strategy with indicators calculated"""
    # Calculate indicators
    df_with_indicators = calculate_indicators(df)
    
    # Get strategy function
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    
    strategy_func = STRATEGIES[strategy_name]
    signals = strategy_func(df_with_indicators)
    
    # Run backtest
    results = run_backtest(
        df_with_indicators, signals, 
        initial_capital=initial_capital,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size_pct=position_size_pct
    )
    results['strategy'] = strategy_name
    
    return results


def save_results(results, indicator_name, output_dir="results"):
    """Save backtest results to CSV"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = f"{output_dir}/backtest_results.csv"
    
    row = {
        "indicator_name": indicator_name,
        "strategy": results.get('strategy', 'unknown'),
        "roi": results["total_return_pct"],
        "drawdown": results["max_drawdown_pct"],
        "sharpe": results["sharpe_ratio"],
        "sortino": results["sortino_ratio"],
        "expected_value": results["expected_value"],
        "num_trades": results["num_trades"],
        "win_rate": results["win_rate_pct"],
        "stop_loss_triggers": results.get('stop_loss_triggered', False),
        "take_profit_triggers": results.get('take_profit_triggered', False),
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


def run_all_strategies(df, initial_capital=DEFAULT_CAPITAL):
    """Run all strategies and return comparison"""
    print("\n" + "="*60)
    print("RUNNING ALL STRATEGIES")
    print("="*60)
    
    all_results = []
    
    for strategy_name in STRATEGIES.keys():
        try:
            print(f"\nTesting: {strategy_name}")
            results = run_strategy(strategy_name, df, initial_capital)
            
            print(f"  ROI: {results['total_return_pct']:+.2f}%")
            print(f"  Drawdown: {results['max_drawdown_pct']:.2f}%")
            print(f"  Sharpe: {results['sharpe_ratio']:.2f}")
            print(f"  Trades: {results['num_trades']}")
            
            all_results.append(results)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return all_results


def run_with_risk_management(df, strategy_name, initial_capital=DEFAULT_CAPITAL):
    """Run strategy with various risk management settings"""
    print("\n" + "="*60)
    print(f"TESTING {strategy_name} WITH RISK MANAGEMENT")
    print("="*60)
    
    configs = [
        {"stop_loss": None, "take_profit": None, "position_size_pct": 1.0, "name": "No risk mgmt"},
        {"stop_loss": 0.02, "take_profit": None, "position_size_pct": 1.0, "name": "2% SL only"},
        {"stop_loss": 0.03, "take_profit": None, "position_size_pct": 1.0, "name": "3% SL only"},
        {"stop_loss": 0.02, "take_profit": 0.04, "position_size_pct": 1.0, "name": "2% SL + 4% TP"},
        {"stop_loss": 0.03, "take_profit": 0.06, "position_size_pct": 1.0, "name": "3% SL + 6% TP"},
        {"stop_loss": 0.02, "take_profit": 0.04, "position_size_pct": 0.5, "name": "2% SL + 50% size"},
        {"stop_loss": 0.03, "take_profit": 0.06, "position_size_pct": 0.5, "name": "3% SL + 50% size"},
    ]
    
    all_results = []
    for config in configs:
        results = run_strategy(
            strategy_name, df, initial_capital,
            stop_loss=config["stop_loss"],
            take_profit=config["take_profit"],
            position_size_pct=config["position_size_pct"]
        )
        results['config'] = config['name']
        print(f"\n{config['name']}:")
        print(f"  ROI: {results['total_return_pct']:+.2f}% | DD: {results['max_drawdown_pct']:.2f}% | Sharpe: {results['sharpe_ratio']:.2f}")
        all_results.append(results)
    
    return all_results


def test_timeframes(symbol="BTCUSDT", initial_capital=DEFAULT_CAPITAL):
    """Test strategies across different timeframes"""
    print("\n" + "="*60)
    print("TESTING ACROSS TIMEFRAMES")
    print("="*60)
    
    intervals = ["15m", "1h", "4h", "1d"]
    best_strategies = {}
    
    for interval in intervals:
        print(f"\n--- {interval} ---")
        df = fetch_btc_data(symbol=symbol, interval=interval, days=180)
        
        if df is not None:
            results = run_all_strategies(df, initial_capital)
            
            # Find best strategy for this timeframe
            if results:
                best = max(results, key=lambda x: x['total_return_pct'])
                best_strategies[interval] = best
                print(f"Best: {best['strategy']} with {best['total_return_pct']:+.2f}%")
    
    return best_strategies


def main():
    """Test the enhanced backtester"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Backtester")
    parser.add_argument("--strategy", type=str, help="Specific strategy to test")
    parser.add_argument("--timeframes", action="store_true", help="Test across timeframes")
    parser.add_argument("--risk-test", action="store_true", help="Test risk management configs")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="Initial capital")
    args = parser.parse_args()
    
    if args.timeframes:
        test_timeframes(initial_capital=args.capital)
    else:
        print("Fetching BTC data...")
        df = fetch_btc_data(days=365)
        
        if df is not None:
            if args.risk_test and args.strategy:
                run_with_risk_management(df, args.strategy, args.capital)
            elif args.strategy:
                results = run_strategy(args.strategy, df, args.capital)
                print(f"\n=== {args.strategy} Results ===")
                print(f"ROI: {results['total_return_pct']:+.2f}%")
                print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
                print(f"Sharpe: {results['sharpe_ratio']:.2f}")
                print(f"Trades: {results['num_trades']}")
            else:
                run_all_strategies(df, args.capital)


if __name__ == "__main__":
    main()
