#!/usr/bin/env python3
"""
Strategy Variation Tester - Find most profitable configuration
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
    return df

def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema9'] = df['close'].ewm(span=9).mean()
    
    # MACD
    df['macd'] = df['ema20'] - df['ema50']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_dn'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # ATR
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_percent'] = df['atr'] / df['close'] * 100
    
    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_bounce'] = (df['close'] <= df['bb_lower']) | (df['close'] >= df['bb_upper'])
    
    # Trend
    df['trend_up'] = df['ema20'] > df['ema50']
    df['trend_dn'] = df['ema20'] < df['ema50']
    
    # RSI zones
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_overbought'] = df['rsi'] > 70
    df['rsi_bull'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) < 35)
    df['rsi_bear'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) > 65)
    
    # Order blocks (simplified)
    df['is_bear'] = df['close'] < df['open']
    df['is_bull'] = df['close'] > df['open']
    df['bull_ob'] = df['is_bear'].shift(1) & df['is_bull']
    df['bear_ob'] = df['is_bull'].shift(1) & df['is_bear']
    
    # FVG
    df['fvg_bull'] = df['low'].shift(1) > df['high'].shift(2)
    df['fvg_bear'] = df['high'].shift(1) < df['low'].shift(2)
    
    # Session (NY session: 14:00-23:00 UTC = 9AM-6PM EST)
    df['hour'] = df.index.hour
    df['ny_session'] = (df['hour'] >= 14) & (df['hour'] <= 23)
    
    # Asian session (0:00-7:00 UTC)
    df['asian_session'] = (df['hour'] >= 0) & (df['hour'] <= 7)
    
    return df

def run_strategy(df, long_cond, short_cond, name, rr=2.0, atr_mult=1.5):
    df = df.copy()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    capital = 100000
    position = None
    trades = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        
        # Exit
        if position:
            if row['low'] <= position['stop'] or row['high'] >= position['target']:
                pnl = -position['risk'] if row['low'] <= position['stop'] else position['risk'] * rr
                capital += pnl
                trades.append({'pnl': pnl, 'dir': position['dir']})
                position = None
        
        # Entry
        if not position:
            stop_dist = row['atr'] * atr_mult
            if long_cond.iloc[i] and (stop_dist * rr) / stop_dist >= 1.5:
                position = {'dir': 'long', 'entry': row['close'], 'stop': row['close'] - stop_dist, 
                           'target': row['close'] + stop_dist * rr, 'risk': stop_dist}
            elif short_cond.iloc[i] and (stop_dist * rr) / stop_dist >= 1.5:
                position = {'dir': 'short', 'entry': row['close'], 'stop': row['close'] + stop_dist,
                           'target': row['close'] - stop_dist * rr, 'risk': stop_dist}
    
    if not trades:
        return {'name': name, 'trades': 0, 'wr': 0, 'pf': 0, 'return': 0}
    
    df_trades = pd.DataFrame(trades)
    wins = len(df_trades[df_trades['pnl'] > 0])
    losses = len(df_trades[df_trades['pnl'] < 0])
    total = len(trades)
    
    pf = abs(df_trades[df_trades['pnl'] > 0]['pnl'].sum() / df_trades[df_trades['pnl'] < 0]['pnl'].sum()) if losses > 0 else 0
    
    return {
        'name': name,
        'trades': total,
        'wr': wins / total * 100 if total > 0 else 0,
        'pf': pf,
        'return': (capital - 100000) / 100000 * 100
    }

def main():
    print("=" * 70)
    print("STRATEGY VARIATION TESTER - Last 12 Months")
    print("=" * 70)
    
    results = []
    
    for symbol in ['ES', 'NQ']:
        print(f"\n--- {symbol} ---")
        df = load_data(symbol)
        df = calculate_indicators(df)
        
        variations = [
            # (name, long_condition, short_condition)
            ("RSI Oversold Bounce + Trend", 
             (df['trend_up']) & (df['rsi_bull']),
             (df['trend_dn']) & (df['rsi_bear'])),
            
            ("MACD Crossover + Trend",
             (df['trend_up']) & (df['macd_cross_up']),
             (df['trend_dn']) & (df['macd_cross_dn'])),
            
            ("BB Bounce + Trend",
             (df['trend_up']) & (df['bb_bounce']) & (df['close'] < df['bb_lower']),
             (df['trend_dn']) & (df['bb_bounce']) & (df['close'] > df['bb_upper'])),
            
            ("RSI + FVG",
             (df['rsi_bull']) & (df['fvg_bull']),
             (df['rsi_bear']) & (df['fvg_bear'])),
            
            ("MACD + RSI",
             (df['macd_cross_up']) & (df['rsi_oversold']),
             (df['macd_cross_dn']) & (df['rsi_overbought'])),
            
            ("Trend Only + BB",
             (df['trend_up']) & (df['close'] < df['bb_lower']),
             (df['trend_dn']) & (df['close'] > df['bb_upper'])),
            
            ("RSI Extreme + FVG",
             (df['rsi_oversold']) & (df['fvg_bull']),
             (df['rsi_overbought']) & (df['fvg_bear'])),
             
            ("NY Session Only + RSI",
             (df['ny_session']) & (df['rsi_bull']) & (df['trend_up']),
             (df['ny_session']) & (df['rsi_bear']) & (df['trend_dn'])),
             
            ("Asian Session + RSI",
             (df['asian_session']) & (df['rsi_bull']) & (df['trend_up']),
             (df['asian_session']) & (df['rsi_bear']) & (df['trend_dn'])),
             
            ("EMA Cross + Trend",
             (df['ema9'] > df['ema20']) & (df['trend_up']),
             (df['ema9'] < df['ema20']) & (df['trend_dn'])),
             
            ("All Combined: Trend+RSI+MACD",
             (df['trend_up']) & (df['rsi_bull']) & (df['macd_cross_up']),
             (df['trend_dn']) & (df['rsi_bear']) & (df['macd_cross_dn'])),
             
            ("FVG Only (no trend)",
             (df['fvg_bull']),
             (df['fvg_bear'])),
             
            ("OB + RSI",
             (df['bull_ob']) & (df['rsi_bull']),
             (df['bear_ob']) & (df['rsi_bear'])),
        ]
        
        for name, long_cond, short_cond in variations:
            result = run_strategy(df, long_cond, short_cond, f"{symbol} - {name}")
            results.append(result)
            print(f"  {name[:40]:<40} | Trades: {result['trades']:>3} | WR: {result['wr']:>5.1f}% | PF: {result['pf']:>5.2f} | Return: {result['return']:>6.1f}%")
    
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)
    
    # Sort by profit factor, then by return
    results.sort(key=lambda x: (x['pf'], x['return']), reverse=True)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:>2}. {r['name']:<50} | PF: {r['pf']:>5.2f} | Return: {r['return']:>6.1f}% | Trades: {r['trades']}")

if __name__ == "__main__":
    main()
