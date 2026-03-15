#!/usr/bin/env python3
"""
HTF Reversal Strategy Backtester - Improved
Focus on 1:2 RR, proper SL location, fixed TP
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategies"))
from strategies.htf_reversal_divergence import HTFReversalDivergence

BINANCE_API = "https://api.binance.com/api/v3"


def fetch_btc_data_1m(days=30):
    """Fetch 1m BTC data from Binance"""
    print(f"Fetching {days} days of 1m BTC data...")
    
    all_data = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    chunk_hours = 16
    current_start = start_time
    
    while current_start < end_time:
        current_end = current_start + timedelta(hours=chunk_hours)
        if current_end > end_time:
            current_end = end_time
        
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(current_end.timestamp() * 1000),
            "limit": 1000
        }
        
        try:
            response = requests.get(f"{BINANCE_API}/klines", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            df_chunk = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df_chunk['open_time'] = pd.to_datetime(df_chunk['open_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_chunk[col] = pd.to_numeric(df_chunk[col])
            
            df_chunk = df_chunk[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            df_chunk.set_index('open_time', inplace=True)
            
            all_data.append(df_chunk)
            current_start = current_end
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if all_data:
        df = pd.concat(all_data)
        print(f"Total: {len(df)} candles")
        return df
    return None


def calculate_atr(df, period=14):
    """Calculate ATR for dynamic SL"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def find_swing_low(df, lookback=20):
    """Find recent swing low for SL placement"""
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        low = df['low'].iloc[i]
        if low < df['low'].iloc[i-lookback:i].min() and low < df['low'].iloc[i+1:i+lookback+1].min():
            swing_lows.append((i, low))
    return swing_lows


def find_swing_high(df, lookback=20):
    """Find recent swing high for SL placement"""
    swing_highs = []
    for i in range(lookback, len(df) - lookback):
        high = df['high'].iloc[i]
        if high > df['high'].iloc[i-lookback:i].max() and high > df['high'].iloc[i+1:i+lookback+1].max():
            swing_highs.append((i, high))
    return swing_highs


def run_backtest_improved(df, signals, 
                          initial_capital=100000,
                          commission=0.001,
                          rr_ratio=2.0,           # 1:2 Risk:Reward
                          atr_multiplier=1.5,     # SL = ATR * multiplier
                          min_rr_threshold=1.5,   # Only take trades with >= 1.5 RR
                          position_size_pct=0.95):
    """
    Improved backtest with:
    - 1:2 RR ratio
    - ATR-based SL placement
    - Fixed TP (no trailing)
    - No breakeven moves
    - Minimum RR filter
    """
    capital = initial_capital
    position = 0
    position_entry = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    
    # Calculate ATR
    df['atr'] = calculate_atr(df, 14)
    
    # Track total PnL instead of capital
    total_pnl = 0
    
    print(f"\n=== BACKTEST CONFIG (IMPROVED) ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"RR Ratio: 1:{rr_ratio}")
    print(f"SL: ATR * {atr_multiplier}")
    print(f"Min RR Filter: {min_rr_threshold}")
    print(f"Position Size: {position_size_pct*100}% per trade")
    print(f"Commission: {commission*100}%")
    
    # Find swing points for reference
    swing_lows = find_swing_low(df, 20)
    swing_highs = find_swing_high(df, 20)
    
    for i in range(100, len(df)):
        price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        atr = df['atr'].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Check if SL or TP hit
        if position > 0:
            sl_hit = low_price <= stop_loss
            tp_hit = high_price >= take_profit
            
            if sl_hit:
                # Stop loss triggered
                exit_price = stop_loss
                pnl = (exit_price - position_entry) * position
                # Add PnL to capital
                capital += pnl - (position * position_entry * commission)  # Simplified
                trades.append({
                    'entry_time': df.index[i-1],
                    'exit_time': df.index[i],
                    'entry_price': position_entry,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': ((exit_price - position_entry) / position_entry) * 100,
                    'type': 'STOP_LOSS',
                    'rr': -1
                })
                position = 0
                position_entry = 0
                stop_loss = 0
                take_profit = 0
            
            elif tp_hit:
                # Take profit triggered
                exit_price = take_profit
                pnl = (exit_price - position_entry) * position
                capital += pnl - (position * position_entry * commission)
                trades.append({
                    'entry_time': df.index[i-1],
                    'exit_time': df.index[i],
                    'entry_price': position_entry,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': ((exit_price - position_entry) / position_entry) * 100,
                    'type': 'TAKE_PROFIT',
                    'rr': rr_ratio
                })
                position = 0
                position_entry = 0
                stop_loss = 0
                take_profit = 0
        
        # Entry signals (only if no position)
        if signal == 1 and position == 0 and not pd.isna(atr):
            # Long signal - SL below recent swing low or ATR-based
            sl_distance = atr * atr_multiplier
            entry_sl = price - sl_distance
            entry_tp = price + (sl_distance * rr_ratio)
            
            # Calculate actual RR
            actual_rr = (entry_tp - price) / sl_distance
            
            # Only enter if RR meets threshold
            if actual_rr >= min_rr_threshold:
                # Use percentage of capital, not deduct full amount
                position_value = capital * position_size_pct
                position = position_value / price  # BTC quantity
                position_entry = price
                stop_loss = entry_sl
                take_profit = entry_tp
                # Capital stays fully available (we track position value separately)
                # No capital deduction - it's a paper trade calculation
                
                trades.append({
                    'entry_time': df.index[i],
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': atr,
                    'type': 'ENTRY_LONG',
                    'expected_rr': actual_rr,
                    'position_value': position_value
                })
    
    # Close any open position at end
    if position > 0:
        final_price = df['close'].iloc[-1]
        pnl = (final_price - position_entry) * position
        capital = position * final_price * (1 - commission)
        trades.append({
            'exit_time': df.index[-1],
            'exit_price': final_price,
            'pnl': pnl,
            'pnl_pct': ((final_price - position_entry) / position_entry) * 100,
            'type': 'CLOSE_POSITION'
        })
    
    final_capital = capital + (position * df['close'].iloc[-1] if position > 0 else 0)
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': ((final_capital - initial_capital) / initial_capital) * 100,
        'trades': trades
    }


def print_results(results):
    trades = results['trades']
    entries = [t for t in trades if t.get('type') == 'ENTRY_LONG']
    exits = [t for t in trades if t.get('type') in ['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_POSITION']]
    
    print(f"\n=== BACKTEST RESULTS (IMPROVED) ===")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:+.2f}%")
    print(f"\nTotal Trades: {len(entries)} entries, {len(exits)} exits")
    
    closed_trades = [t for t in trades if t.get('type') in ['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_POSITION']]
    if closed_trades:
        wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        print(f"\nWins: {len(wins)}")
        print(f"Losses: {len(losses)}")
        
        if len(closed_trades) > 0:
            win_rate = len(wins) / len(closed_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        if wins:
            avg_win = np.mean([t['pnl'] for t in wins])
            print(f"Average Win: ${avg_win:,.2f}")
        
        if losses:
            avg_loss = np.mean([t['pnl'] for t in losses])
            print(f"Average Loss: ${avg_loss:,.2f}")
        
        if wins and losses:
            rr_achieved = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            print(f"Achieved RR: {rr_achieved:.2f}:1")
        
        # Exit type breakdown
        sl_triggers = len([t for t in closed_trades if t['type'] == 'STOP_LOSS'])
        tp_triggers = len([t for t in closed_trades if t['type'] == 'TAKE_PROFIT'])
        print(f"\nStop Loss Triggers: {sl_triggers}")
        print(f"Take Profit Triggers: {tp_triggers}")


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    print(f"Running HTF Reversal Strategy (IMPROVED)")
    print(f"Period: Last {days} days on 1m data")
    print("=" * 50)
    
    df = fetch_btc_data_1m(days=days)
    
    if df is None or len(df) == 0:
        print("Failed to fetch data")
        return
    
    print(f"\nRunning strategy on {len(df)} candles...")
    
    strategy = HTFReversalDivergence(htf_timeframe="15", rsi_length=14, pivot_lookback=3)
    signals = strategy.generate_signals(df)['signal']
    
    print(f"Signals generated: {(signals != 0).sum()}")
    
    # Run with improved settings
    results = run_backtest_improved(
        df, signals,
        initial_capital=100000,
        rr_ratio=2.0,           # 1:2 RR
        atr_multiplier=1.5,     # SL = 1.5 * ATR
        min_rr_threshold=1.5,   # Min 1.5 RR to enter
        position_size_pct=0.95
    )
    
    print_results(results)


if __name__ == "__main__":
    main()