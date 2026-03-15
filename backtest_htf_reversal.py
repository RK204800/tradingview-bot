#!/usr/bin/env python3
"""
HTF Reversal + RSI Divergence Strategy Backtest
Replicates the TradingView Pine Script strategy
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def load_data(symbol):
    df = pd.read_parquet(f"/root/.openclaw/workspace/data/{symbol.lower()}_1h.parquet")
    df = df.sort_index()
    df = df[df.index >= df.index.max() - timedelta(days=365)]
    print(f"Loaded {symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df

def calculate_indicators(df):
    """Calculate HTF Reversal + RSI Divergence indicators"""
    
    # RSI on 1H
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # HTF (4h) reversal patterns - simulate using 4h candles within 1h data
    # Group by 4-hour periods
    df['htf_idx'] = df.index.floor('4h')
    
    # For each 4h bar, get OHLC
    htf_ohlc = df.groupby('htf_idx').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate HTF reversal patterns
    htf_ohlc['body'] = abs(htf_ohlc['close'] - htf_ohlc['open'])
    htf_ohlc['range'] = htf_ohlc['high'] - htf_ohlc['low']
    htf_ohlc['upper_wick'] = htf_ohlc['high'] - htf_ohlc[['close', 'open']].max(axis=1)
    htf_ohlc['lower_wick'] = htf_ohlc[['close', 'open']].min(axis=1) - htf_ohlc['low']
    
    # Hammer: small body, big lower wick
    htf_ohlc['is_hammer'] = (htf_ohlc['body'] < htf_ohlc['range'] * 0.33) & \
                             (htf_ohlc['lower_wick'] > htf_ohlc['body'] * 2) & \
                             (htf_ohlc['upper_wick'] < htf_ohlc['body'])
    
    # Bullish engulfing
    htf_ohlc['prev_bear'] = htf_ohlc['close'].shift(1) < htf_ohlc['open'].shift(1)
    htf_ohlc['bull_engulf'] = htf_ohlc['prev_bear'] & \
                               (htf_ohlc['close'] > htf_ohlc['open']) & \
                               (htf_ohlc['open'].shift(1) < htf_ohlc['close'].shift(1)) & \
                               (htf_ohlc['close'] > htf_ohlc['open'].shift(1))
    
    # Shooting star: small body, big upper wick
    htf_ohlc['is_shooting'] = (htf_ohlc['body'] < htf_ohlc['range'] * 0.33) & \
                              (htf_ohlc['upper_wick'] > htf_ohlc['body'] * 2) & \
                              (htf_ohlc['lower_wick'] < htf_ohlc['body'])
    
    # Bearish engulfing
    htf_ohlc['prev_bull'] = htf_ohlc['close'].shift(1) > htf_ohlc['open'].shift(1)
    htf_ohlc['bear_engulf'] = htf_ohlc['prev_bull'] & \
                               (htf_ohlc['close'] < htf_ohlc['open']) & \
                               (htf_ohlc['open'].shift(1) > htf_ohlc['close'].shift(1)) & \
                               (htf_ohlc['close'] < htf_ohlc['open'].shift(1))
    
    # HTF signals
    htf_ohlc['htf_bull'] = htf_ohlc['is_hammer'] | htf_ohlc['bull_engulf']
    htf_ohlc['htf_bear'] = htf_ohlc['is_shooting'] | htf_ohlc['bear_engulf']
    
    # Map back to 1h dataframe
    df['htf_bull'] = df['htf_idx'].map(htf_ohlc['htf_bull'])
    df['htf_bear'] = df['htf_idx'].map(htf_ohlc['htf_bear'])
    
    # RSI Divergence
    df['rsi_pivot_low'] = df['rsi'].rolling(3, center=True).min()
    df['price_pivot_low'] = df['low'].rolling(3, center=True).min()
    
    # Bullish RSI divergence: price makes lower low, RSI makes higher low
    df['rsi_bull_div'] = (df['low'] < df['low'].shift(3)) & \
                          (df['rsi'] > df['rsi'].shift(3))
    
    df['rsi_pivot_high'] = df['rsi'].rolling(3, center=True).max()
    df['price_pivot_high'] = df['high'].rolling(3, center=True).max()
    
    df['rsi_bear_div'] = (df['high'] > df['high'].shift(3)) & \
                           (df['rsi'] < df['rsi'].shift(3))
    
    # Final signals: HTF reversal + RSI divergence
    df['bull_signal'] = df['htf_bull'] & df['rsi_bull_div']
    df['bear_signal'] = df['htf_bear'] & df['rsi_bear_div']
    
    # ATR for stops
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    return df

def run_backtest(df, rr=2.0, atr_mult=1.5, min_rr=1.5):
    capital = 100000
    position = None
    trades = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        
        # Check exit
        if position:
            if row['low'] <= position['stop'] or row['high'] >= position['target']:
                if row['low'] <= position['stop']:
                    pnl = -position['risk']
                else:
                    pnl = position['risk'] * rr
                
                capital += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i],
                    'direction': position['direction'],
                    'pnl': pnl
                })
                position = None
        
        # Check entry
        if not position:
            atr = row['atr']
            stop_dist = atr * atr_mult
            actual_rr = (stop_dist * rr) / stop_dist
            
            if row['bull_signal'] and actual_rr >= min_rr:
                position = {
                    'direction': 'long',
                    'entry': row['close'],
                    'stop': row['close'] - stop_dist,
                    'target': row['close'] + (stop_dist * rr),
                    'risk': stop_dist,
                    'entry_time': df.index[i]
                }
            elif row['bear_signal'] and actual_rr >= min_rr:
                position = {
                    'direction': 'short',
                    'entry': row['close'],
                    'stop': row['close'] + stop_dist,
                    'target': row['close'] - (stop_dist * rr),
                    'risk': stop_dist,
                    'entry_time': df.index[i]
                }
    
    return capital, trades

def calculate_metrics(trades, initial_capital, final_capital):
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    
    total = len(trades)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    
    pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
    
    return {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': pf,
        'total_return': final_capital - initial_capital,
        'return_pct': (final_capital - initial_capital) / initial_capital * 100
    }

if __name__ == "__main__":
    print("=" * 60)
    print("HTF REVERSAL + RSI DIVERGENCE BACKTEST")
    print("=" * 60)
    
    for symbol in ['ES', 'NQ']:
        print(f"\n--- {symbol} ---")
        
        df = load_data(symbol)
        df = calculate_indicators(df)
        
        print(f"HTF Bull signals: {df['htf_bull'].sum()}")
        print(f"HTF Bear signals: {df['htf_bear'].sum()}")
        print(f"RSI Bull Div: {df['rsi_bull_div'].sum()}")
        print(f"RSI Bear Div: {df['rsi_bear_div'].sum()}")
        print(f"Combined LONG: {df['bull_signal'].sum()}")
        print(f"Combined SHORT: {df['bear_signal'].sum()}")
        
        initial_cap = 100000
        final_cap, trades = run_backtest(df, rr=2.0, atr_mult=1.5, min_rr=1.5)
        
        metrics = calculate_metrics(trades, initial_cap, final_cap)
        
        print(f"\nRESULTS:")
        print(f"  Trades: {metrics.get('total_trades', 0)}")
        print(f"  Wins: {metrics.get('wins', 0)} | Losses: {metrics.get('losses', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Return: ${metrics.get('total_return', 0):,.0f} ({metrics.get('return_pct', 0):.1f}%)")
