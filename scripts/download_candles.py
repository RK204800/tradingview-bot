#!/usr/bin/env python3
"""
Candle Downloader for TradingView Bot
Run via cron: daily at 5:55 PM EST (9:55 AM Sydney next day)
Usage: python scripts/download_candles.py --timeframe 1m --days 1
"""

import os
import sys
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BINANCE_API = "https://api.binance.com/api/v3"

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fetch_candles(symbol="BTCUSDT", interval="1m", days=1):
    """Fetch candles from Binance"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": 1000
    }
    
    try:
        response = requests.get(f"{BINANCE_API}/klines", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Clean up
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Select relevant columns
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
        
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download candles from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    # Default output: data/candles/YYYY-MM-DD.csv
    if args.output is None:
        output_dir = PROJECT_ROOT / "data" / "candles"
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        args.output = output_dir / f"{args.timeframe}_{date_str}.csv"
    
    print(f"Downloading {args.timeframe} candles for {args.symbol} (last {args.days} day(s))...")
    
    df = fetch_candles(args.symbol, args.timeframe, args.days)
    
    if df is not None:
        df.to_csv(args.output, index=False)
        print(f"✅ Saved {len(df)} candles to {args.output}")
        return 0
    else:
        print(f"❌ Failed to download candles")
        return 1


if __name__ == "__main__":
    sys.exit(main())
