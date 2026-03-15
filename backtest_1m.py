#!/usr/bin/env python3
"""
HTF Reversal Strategy Backtester
Run on 1m data with risk management
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategies"))

from strategies.htf_reversal_divergence import HTFReversalDivergence

BINANCE_API = "https://api.binance.com/api/v3"


def fetch_btc_data_1m(days=30):
    """Fetch 1m BTC data from Binance in chunks"""
    print(f"Fetching {days} days of 1m BTC data...")
    
    all_data = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Binance limits 1000 candles per request, so we need to chunk
    # 1000 minutes = ~16.6 hours
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
            
            # Convert to DataFrame
            df_chunk = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Clean up
            df_chunk['open_time'] = pd.to_datetime(df_chunk['open_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_chunk[col] = pd.to_numeric(df_chunk[col])
            
            df_chunk = df_chunk[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            df_chunk.set_index('open_time', inplace=True)
            
            all_data.append(df_chunk)
            print(f"  Fetched {len(df_chunk)} candles ({current_start} to {current_end})")
            
            current_start = current_end
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if all_data:
        df = pd.concat(all_data)
        print(f"Total: {len(df)} candles")
        return df
    return None


def run_backtest(df, signals, 
                 initial_capital=100000,
                 commission=0.001,
                 stop_loss_pct=0.02,    # 2% stop loss
                 take_profit_pct=0.05,  # 5% take profit
                 position_size_pct=0.95):  # 95% of capital per trade
    
    """Run backtest with stop loss and take profit"""
    capital = initial_capital
    position = 0
    position_entry = 0
    trades = []
    equity_curve = [initial_capital]
    
    print(f"\n=== BACKTEST CONFIG ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Stop Loss: {stop_loss_pct*100}%")
    print(f"Take Profit: {take_profit_pct*100}%")
    print(f"Position Size: {position_size_pct*100}%")
    print(f"Commission: {commission*100}%")
    
    for i in range(100, len(df)):  # Need 100 bars for indicators
        price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Check stop loss
        if position > 0:
            stop_price = position_entry * (1 - stop_loss_pct)
            if low_price <= stop_price:
                # Stop loss triggered
                exit_price = stop_price
                pnl = (exit_price - position_entry) * position
                capital = position * exit_price * (1 - commission)
                trades.append({
                    'entry_time': df.index[i-1],
                    'exit_time': df.index[i],
                    'entry_price': position_entry,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_entry * position)) * 100,
                    'type': 'STOP_LOSS'
                })
                position = 0
                position_entry = 0
            
            # Check take profit
            else:
                tp_price = position_entry * (1 + take_profit_pct)
                if high_price >= tp_price:
                    # Take profit triggered
                    exit_price = tp_price
                    pnl = (exit_price - position_entry) * position
                    capital = position * exit_price * (1 - commission)
                    trades.append({
                        'entry_time': df.index[i-1],
                        'exit_time': df.index[i],
                        'entry_price': position_entry,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (position_entry * position)) * 100,
                        'type': 'TAKE_PROFIT'
                    })
                    position = 0
                    position_entry = 0
        
        # Entry signals
        if signal == 1 and position == 0:  # Buy signal
            position_size = (capital * position_size_pct) / price
            position = position_size
            position_entry = price
            capital = capital * (1 - position_size_pct)  # Reserve capital
            trades.append({
                'entry_time': df.index[i],
                'entry_price': price,
                'type': 'ENTRY_LONG'
            })
        
        elif signal == -1 and position == 0:  # Sell signal
            # Short (optional - for now we just skip)
            pass
        
        # Track equity
        if position > 0:
            current_value = position * price
            equity = capital + current_value
        else:
            equity = capital
        equity_curve.append(equity)
    
    # Close any open position
    if position > 0:
        final_price = df['close'].iloc[-1]
        pnl = (final_price - position_entry) * position
        capital = position * final_price * (1 - commission)
        trades.append({
            'exit_time': df.index[-1],
            'exit_price': final_price,
            'pnl': pnl,
            'pnl_pct': (pnl / (position_entry * position)) * 100,
            'type': 'CLOSE_POSITION'
        })
    
    final_capital = capital + (position * df['close'].iloc[-1] if position > 0 else 0)
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': ((final_capital - initial_capital) / initial_capital) * 100,
        'trades': trades,
        'equity_curve': equity_curve
    }


def print_results(results):
    """Print backtest results"""
    trades = results['trades']
    entries = [t for t in trades if t.get('type') == 'ENTRY_LONG']
    exits = [t for t in trades if t.get('type') in ['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_POSITION']]
    
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:+.2f}%")
    print(f"\nTotal Trades: {len(entries)} entries, {len(exits)} exits")
    
    # Trade analysis
    closed_trades = [t for t in trades if t.get('type') in ['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_POSITION']]
    if closed_trades:
        wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        print(f"\nWins: {len(wins)}")
        print(f"Losses: {len(losses)}")
        print(f"Win Rate: {len(wins)/len(closed_trades)*100:.1f}%")
        
        if wins:
            avg_win = np.mean([t['pnl'] for t in wins])
            print(f"Average Win: ${avg_win:,.2f}")
        
        if losses:
            avg_loss = np.mean([t['pnl'] for t in losses])
            print(f"Average Loss: ${avg_loss:,.2f}")
        
        # Exit type breakdown
        sl_triggers = len([t for t in closed_trades if t['type'] == 'STOP_LOSS'])
        tp_triggers = len([t for t in closed_trades if t['type'] == 'TAKE_PROFIT'])
        print(f"\nStop Loss Triggers: {sl_triggers}")
        print(f"Take Profit Triggers: {tp_triggers}")


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    print(f"Running HTF Reversal Strategy Backtest")
    print(f"Period: Last {days} days on 1m data")
    print(f"=" * 50)
    
    # Fetch data
    df = fetch_btc_data_1m(days=days)
    
    if df is None or len(df) == 0:
        print("Failed to fetch data")
        return
    
    print(f"\nRunning strategy on {len(df)} candles...")
    
    # Run strategy
    strategy = HTFReversalDivergence(htf_timeframe="15", rsi_length=14, pivot_lookback=3)
    
    # Generate signals on full dataset
    signals = strategy.generate_signals(df)['signal']
    
    print(f"Signals generated: {(signals != 0).sum()}")
    
    # Run backtest with risk management
    results = run_backtest(
        df, signals,
        initial_capital=100000,
        stop_loss_pct=0.02,    # 2%
        take_profit_pct=0.05,  # 5%
        position_size_pct=0.95
    )
    
    print_results(results)


if __name__ == "__main__":
    main()
