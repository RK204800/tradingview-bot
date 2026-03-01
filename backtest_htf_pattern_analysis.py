#!/usr/bin/env python3
"""
HTF Pattern Analysis - Identify which HTF pattern leads to profitable trades
Tracks trades by: Hammer, Bull Engulfing, Shooting Star, Bear Engulfing
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data(symbol):
    df = pd.read_parquet(f"/root/.openclaw/workspace/data/{symbol.lower()}_1h.parquet")
    df = df.sort_index()
    df = df[df.index >= df.index.max() - timedelta(days=365)]
    print(f"Loaded {symbol}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
    return df

def calculate_indicators(df):
    """Calculate HTF reversal patterns and RSI divergence"""
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # RSI Divergence - pivots
    df['rsi_swing_low'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) < df['rsi'].shift(2))
    df['rsi_swing_high'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) > df['rsi'].shift(2))
    
    # Price pivots
    df['price_swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['price_swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    
    # Bullish RSI Divergence
    df['bull_div'] = df['price_swing_low'] & df['rsi_swing_low'] & (df['rsi'] > df['rsi'].shift(2))
    
    # Bearish RSI Divergence
    df['bear_div'] = df['price_swing_high'] & df['rsi_swing_high'] & (df['rsi'] < df['rsi'].shift(2))
    
    # === HTF REVERSAL PATTERNS (on 1H for simulation) ===
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    
    # Hammer (bullish reversal) - small body, big lower wick
    df['is_hammer'] = (df['body'] < df['range'] * 0.33) & (df['lower_wick'] > df['body'] * 2) & (df['upper_wick'] < df['body'])
    
    # Bullish Engulfing
    df['bull_engulf'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & \
                          (df['open'].shift(1) < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
    
    # Shooting Star (bearish reversal)
    df['is_shooting'] = (df['body'] < df['range'] * 0.33) & (df['upper_wick'] > df['body'] * 2) & (df['lower_wick'] < df['body'])
    
    # Bearish Engulfing
    df['bear_engulf'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & \
                          (df['open'].shift(1) > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    
    # HTF Bull/Bear signals
    df['htf_bull'] = df['is_hammer'] | df['bull_engulf']
    df['htf_bear'] = df['is_shooting'] | df['bear_engulf']
    
    # Combined signals (like Pine Script)
    df['bull_signal'] = df['bull_div'] & df['htf_bull']
    df['bear_signal'] = df['bear_div'] & df['htf_bear']
    
    # ATR for stops
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    return df

def run_backtest_with_htf_tracking(df, rr=2.0, atr_mult=1.5, min_rr=1.5):
    """Backtest that tracks which HTF pattern triggered each trade"""
    
    capital = 100000
    position = None
    trades = []
    
    # Track by pattern
    pattern_trades = {
        'hammer': [],
        'bull_engulf': [],
        'shooting_star': [],
        'bear_engulf': [],
    }
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        
        # Exit position
        if position:
            if row['low'] <= position['stop'] or row['high'] >= position['target']:
                # Calculate P&L
                if row['low'] <= position['stop']:
                    pnl = -position['risk']
                else:
                    pnl = position['risk'] * rr
                
                capital += pnl
                
                # Record trade with HTF pattern
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i],
                    'pnl': pnl,
                    'direction': position['direction'],
                    'htf_pattern': position['htf_pattern']
                }
                trades.append(trade)
                
                # Add to pattern-specific list
                pattern = position['htf_pattern']
                if pattern in pattern_trades:
                    pattern_trades[pattern].append(pnl)
                
                position = None
        
        # Entry
        if not position:
            atr = row['atr']
            stop_dist = atr * atr_mult
            actual_rr = (stop_dist * rr) / stop_dist
            
            # Check which HTF pattern triggered
            if row['bull_signal'] and actual_rr >= min_rr:
                # Determine which specific pattern
                if row['is_hammer']:
                    htf_pattern = 'hammer'
                elif row['bull_engulf']:
                    htf_pattern = 'bull_engulf'
                else:
                    htf_pattern = 'other'
                
                position = {
                    'direction': 'long',
                    'entry': row['close'],
                    'stop': row['close'] - stop_dist,
                    'target': row['close'] + (stop_dist * rr),
                    'risk': stop_dist,
                    'entry_time': df.index[i],
                    'htf_pattern': htf_pattern
                }
                
            elif row['bear_signal'] and actual_rr >= min_rr:
                # Determine which specific pattern
                if row['is_shooting']:
                    htf_pattern = 'shooting_star'
                elif row['bear_engulf']:
                    htf_pattern = 'bear_engulf'
                else:
                    htf_pattern = 'other'
                
                position = {
                    'direction': 'short',
                    'entry': row['close'],
                    'stop': row['close'] + stop_dist,
                    'target': row['close'] - (stop_dist * rr),
                    'risk': stop_dist,
                    'entry_time': df.index[i],
                    'htf_pattern': htf_pattern
                }
    
    return capital, trades, pattern_trades

def analyze_patterns(trades, pattern_trades):
    """Analyze performance by HTF pattern"""
    
    print("\n" + "=" * 70)
    print("HTF PATTERN ANALYSIS - WHICH PATTERN IS MOST PROFITABLE?")
    print("=" * 70)
    
    pattern_names = {
        'hammer': 'HAMMER',
        'bull_engulf': 'BULL ENGULF',
        'shooting_star': 'SHOOTING STAR',
        'bear_engulf': 'BEAR ENGULF'
    }
    
    results = []
    
    for pattern, pnl_list in pattern_trades.items():
        if len(pnl_list) == 0:
            continue
        
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        
        total = len(pnl_list)
        win_rate = len(wins) / total * 100
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        total_pnl = sum(pnl_list)
        
        results.append({
            'pattern': pattern_names.get(pattern, pattern),
            'trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl
        })
    
    # Sort by profit factor
    results.sort(key=lambda x: (x['profit_factor'], x['total_pnl']), reverse=True)
    
    # Print results
    print(f"\n{'Pattern':<18} | {'Trades':>6} | {'Wins':>5} | {'Loss':>5} | {'WR%':>6} | {'PF':>6} | {'Total P&L':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['pattern']:<18} | {r['trades']:>6} | {r['wins']:>5} | {r['losses']:>5} | {r['win_rate']:>6.1f} | {r['profit_factor']:>6.2f} | ${r['total_pnl']:>10,.0f}")
    
    # Best pattern
    if results:
        best = results[0]
        print(f"\n🏆 BEST PATTERN: {best['pattern']} (PF: {best['profit_factor']:.2f}, Return: ${best['total_pnl']:,.0f})")
    
    return results

def main():
    print("=" * 70)
    print("HTF PATTERN ANALYSIS - WHICH HTF PATTERN LEADS TO PROFIT?")
    print("=" * 70)
    
    for symbol in ['ES', 'NQ']:
        print(f"\n\n=== {symbol} ===")
        
        df = load_data(symbol)
        df = calculate_indicators(df)
        
        print(f"\nPattern signals:")
        print(f"  Hammer: {df['is_hammer'].sum()}")
        print(f"  Bull Engulf: {df['bull_engulf'].sum()}")
        print(f"  Shooting Star: {df['is_shooting'].sum()}")
        print(f"  Bear Engulf: {df['bear_engulf'].sum()}")
        
        # Run backtest with pattern tracking
        initial_capital = 100000
        final_capital, trades, pattern_trades = run_backtest_with_htf_tracking(df)
        
        # Analyze by pattern
        results = analyze_patterns(trades, pattern_trades)
        
        # Overall results
        all_pnl = [t['pnl'] for t in trades]
        if all_pnl:
            wins = [p for p in all_pnl if p > 0]
            losses = [p for p in all_pnl if p < 0]
            overall_pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 0
            
            print(f"\n📊 OVERALL {symbol}:")
            print(f"  Total Trades: {len(trades)}")
            print(f"  Win Rate: {len(wins)/len(all_pnl)*100:.1f}%")
            print(f"  Profit Factor: {overall_pf:.2f}")
            print(f"  Return: ${final_capital - initial_capital:,.0f}")

if __name__ == "__main__":
    main()
