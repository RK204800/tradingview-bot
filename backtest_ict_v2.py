#!/usr/bin/env python3
"""
HTF Reversal ICT Strategy - Simplified V2
Combines: HTF Reversal Patterns + RSI Divergence + ICT Elements
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def load_data(symbol='ES'):
    path = f"/root/.openclaw/workspace/data/{symbol.lower()}_1h.parquet"
    df = pd.read_parquet(path)
    df = df.sort_index()
    cutoff = df.index.max() - timedelta(days=365)
    df = df[df.index >= cutoff]
    print(f"Loaded {symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_backtest(df, params):
    # Parameters
    htf_tf = params.get('htf_tf', '4h')
    rsi_period = params.get('rsi_period', 14)
    atr_mult = params.get('atr_mult', 1.5)
    rr_ratio = params.get('rr_ratio', 2.0)
    min_rr = params.get('min_rr', 1.5)
    
    # RSI on 1H
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    
    # Simplified HTF: Higher timeframe momentum
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['htf_bull'] = df['ema20'] > df['ema50']  # Uptrend
    df['htf_bear'] = df['ema20'] < df['ema50']  # Downtrend
    
    # RSI oversold/overbought signals
    df['rsi_oversold'] = df['rsi'] < 35
    df['rsi_overbought'] = df['rsi'] > 65
    
    # RSI bounce (coming out of oversold)
    df['rsi_bull'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) < 35)
    df['rsi_bear'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) > 65)
    
    # Simple ICT: Order Block (last bear candle before bullish move)
    df['is_bear'] = df['close'] < df['open']
    df['is_bull'] = df['close'] > df['open']
    df['bull_ob'] = df['is_bear'].shift(1) & df['is_bull']
    
    # FVG detection (simplified)
    df['fvg_bull'] = (df['low'].shift(1) > df['high'].shift(2))
    df['fvg_bear'] = (df['high'].shift(1) < df['low'].shift(2))
    
    # ATR for stops
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # COMBINED SIGNALS (simplified)
    # LONG: Uptrend + RSI bouncing from oversold + (OB or FVG)
    df['long_signal'] = df['htf_bull'] & df['rsi_bull'] & (df['bull_ob'] | df['fvg_bull'])
    
    # SHORT: Downtrend + RSI falling from overbought + FVG
    df['short_signal'] = df['htf_bear'] & df['rsi_bear'] & df['fvg_bear']
    
    # Backtest
    capital = 100000
    position = None
    trades = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Exit
        if position:
            if row['low'] <= position['stop'] or row['high'] >= position['target']:
                if row['low'] <= position['stop']:
                    pnl = -position['risk']
                else:
                    pnl = position['risk'] * rr_ratio
                capital += pnl
                trades.append({'pnl': pnl, 'dir': position['dir'], 'entry': position['entry'], 'exit': row['close'] if row['high'] >= position['target'] else position['stop']})
                position = None
        
        # Entry
        if not position:
            atr = row['atr']
            stop_dist = atr * atr_mult
            actual_rr = (stop_dist * rr_ratio) / stop_dist
            
            if row['long_signal'] and actual_rr >= min_rr:
                position = {'dir': 'long', 'entry': row['close'], 'stop': row['close'] - stop_dist, 'target': row['close'] + stop_dist * rr_ratio, 'risk': stop_dist}
            elif row['short_signal'] and actual_rr >= min_rr:
                position = {'dir': 'short', 'entry': row['close'], 'stop': row['close'] + stop_dist, 'target': row['close'] - stop_dist * rr_ratio, 'risk': stop_dist}
    
    return capital, trades

def print_results(trades, initial, final, symbol):
    if not trades:
        print(f"{symbol}: NO TRADES")
        return
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    total = len(trades)
    wr = len(wins) / total * 100 if total > 0 else 0
    pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
    
    print(f"\n{'='*50}")
    print(f"{symbol} RESULTS (12 months)")
    print(f"{'='*50}")
    print(f"Trades: {total} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Return: ${final - initial:,.0f} ({(final - initial) / initial * 100:.1f}%)")
    print(f"Final Capital: ${final:,.0f}")

if __name__ == "__main__":
    params = {'rsi_period': 14, 'atr_mult': 1.5, 'rr_ratio': 2.0, 'min_rr': 1.5}
    
    print("=" * 60)
    print("HTF REVERSAL + ICT STRATEGY - V2 BACKTEST")
    print("=" * 60)
    
    for symbol in ['ES', 'NQ']:
        df = load_data(symbol)
        initial = 100000
        final, trades = run_backtest(df, params)
        print_results(trades, initial, final, symbol)
