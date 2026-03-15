#!/usr/bin/env python3
"""HTF Reversal Backtest v3 - Fixed position sizing"""
import sys
sys.path.insert(0, "strategies")
from strategies.htf_reversal_divergence import HTFReversalDivergence
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BINANCE = "https://api.binance.com/api/v3"

def get_data(days):
    print(f"Fetching {days}d 1m data...")
    data = []
    end = datetime.now()
    start = end - timedelta(days=days)
    cur = start
    while cur < end:
        params = {"symbol":"BTCUSDT","interval":"1m","startTime":int(cur.timestamp()*1000),"endTime":int(min(cur+timedelta(hours=16),end).timestamp()*1000),"limit":1000}
        r = requests.get(f"{BINANCE}/klines", params=params, timeout=30).json()
        if not r: break
        df = pd.DataFrame(r, columns=['t','o','h','l','c','v','ct','q','n','tb','tq','i'])
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        for col, new in [('o','open'),('h','high'),('l','low'),('c','close'),('v','volume')]:
            df[new] = pd.to_numeric(df[col])
        data.append(df[['t','open','high','low','close','volume']].set_index('t'))
        cur += timedelta(hours=16)
    return pd.concat(data) if data else None

def calc_atr(df, p=14):
    h = df['high']-df['low']
    hc = abs(df['high']-df['close'].shift())
    lc = abs(df['low']-df['close'].shift())
    return pd.concat([h,hc,lc],axis=1).max(axis=1).rolling(p).mean()

def run(df, sig, rr=2.0, atr_mult=1.5, min_rr=1.5, risk_pct=0.02):
    cap = 100000.0
    initial_cap = 100000.0
    df['atr'] = calc_atr(df)
    pos = None
    trades = []
    trade_num = 0
    
    for i in range(100, len(df)):
        p = df['close'].iloc[i]; h = df['high'].iloc[i]; l = df['low'].iloc[i]; a = df['atr'].iloc[i]
        s = sig.iloc[i] if i < len(sig) else 0
        
        # Check exit
        if pos and (l <= pos['sl'] or h >= pos['tp']):
            xp = pos['tp'] if h >= pos['tp'] else pos['sl']
            # PnL = position_size * price_change
            pnl = pos['size'] * (xp - pos['en'])
            cap += pnl
            trade_num += 1
            trades.append({'type':'TP' if h >= pos['tp'] else 'SL','pnl':pnl, 'risk_rwrded': (xp - pos['en'])/pos['risk_dist']})
            pos = None
        
        # Entry
        if s == 1 and not pos and not pd.isna(a):
            sd = a * atr_mult  # SL distance
            sl = p - sd
            tp = p + sd * rr
            actual_rr = (tp - p) / sd
            
            if actual_rr >= min_rr:
                # FIXED position sizing: risk fixed % of INITIAL capital
                risk_amt = initial_cap * risk_pct
                # Position size = risk amount / SL distance (in BTC)
                btc_size = risk_amt / sd
                pos = {'en':p,'sl':sl,'tp':tp,'size':btc_size,'risk_dist':sd}
    
    final_cap = cap
    ret_pct = ((final_cap - initial_cap) / initial_cap) * 100
    
    print(f"=== RESULTS ({days}d) ===")
    print(f"Initial: ${initial_cap:,.0f} | Final: ${final_cap:,.0f}")
    print(f"Return: {ret_pct:+.1f}%")
    print(f"Trades: {len(trades)}")
    if trades:
        w = [t for t in trades if t['pnl']>0]
        l = [t for t in trades if t['pnl']<=0]
        print(f"Win Rate: {len(w)/len(trades)*100:.1f}% ({len(w)}W/{len(l)}L)")
        print(f"Avg Win: ${np.mean([t['pnl'] for t in w]):,.0f}")
        print(f"Avg Loss: ${np.mean([t['pnl'] for t in l]):,.0f}")
        print(f"Avg RR Won: {np.mean([t['risk_rwrded'] for t in w]):.1f}")
        print(f"Avg RR Lost: {np.mean([t['risk_rwrded'] for t in l]):.1f}")

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv)>1 else 30
    df = get_data(days)
    if df is None: exit()
    strat = HTFReversalDivergence(htf_timeframe="15",rsi_length=14,pivot_lookback=3)
    sig = strat.generate_signals(df)['signal']
    run(df, sig)
