"""
HTF Reversal & 1m RSI Divergence Strategy
Converted from TradingView Pine Script
https://www.tradingview.com/script/... (LuxAlgo)

Bullish Signal: HTF reversal (hammer/bullish engulfing) + 1m RSI bullish divergence
Bearish Signal: HTF reversal (shooting star/bearish engulfing) + 1m RSI bearish divergence
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class HTFReversalDivergence:
    """
    Strategy implementing HTF Reversal + RSI Divergence signals
    """
    
    def __init__(self, 
                 htf_timeframe: str = "15",
                 rsi_length: int = 14,
                 pivot_lookback: int = 3):
        """
        Initialize strategy
        
        Args:
            htf_timeframe: Higher timeframe for reversal detection (15, 60, etc.)
            rsi_length: RSI period
            pivot_lookback: Bars before/after for pivot identification
        """
        self.htf_timeframe = htf_timeframe
        self.rsi_length = rsi_length
        self.pivot_lookback = pivot_lookback
        
        # State for RSI divergence tracking
        self.last_pivot_low_rsi = None
        self.last_pivot_low_price = None
        self.last_pivot_low_idx = None
        self.last_pivot_high_rsi = None
        self.last_pivot_high_price = None
        self.last_pivot_high_idx = None
    
    def calculate_rsi(self, closes: pd.Series, length: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        length = length or self.rsi_length
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=length, min_periods=length).mean()
        avg_loss = loss.rolling(window=length, min_periods=length).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_pivot_low(self, series: pd.Series, lookback: int = None) -> pd.Series:
        """Find pivot lows"""
        lookback = lookback or self.pivot_lookback
        # A pivot low is where the value is lower than all values in the lookback window
        pivot = pd.Series(False, index=series.index)
        
        for i in range(lookback, len(series) - lookback):
            if i >= lookback and i < len(series) - lookback:
                left = series.iloc[i-lookback:i]
                right = series.iloc[i+1:i+lookback+1]
                current = series.iloc[i]
                
                if current < left.min() and current < right.min():
                    pivot.iloc[i] = True
        
        return pivot
    
    def find_pivot_high(self, series: pd.Series, lookback: int = None) -> pd.Series:
        """Find pivot highs"""
        lookback = lookback or self.pivot_lookback
        pivot = pd.Series(False, index=series.index)
        
        for i in range(lookback, len(series) - lookback):
            if i >= lookback and i < len(series) - lookback:
                left = series.iloc[i-lookback:i]
                right = series.iloc[i+1:i+lookback+1]
                current = series.iloc[i]
                
                if current > left.max() and current > right.max():
                    pivot.iloc[i] = True
        
        return pivot
    
    def detect_rsi_divergence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect RSI bullish and bearish divergence
        
        Returns:
            Tuple of (bull_div, bear_div) series
        """
        rsi = self.calculate_rsi(df['close'])
        
        pivots_low = self.find_pivot_low(df['low'])
        pivots_high = self.find_pivot_high(df['high'])
        
        bull_div = pd.Series(False, index=df.index)
        bear_div = pd.Series(False, index=df.index)
        
        # Track last pivot points for divergence detection
        last_low_rsi = None
        last_low_price = None
        last_high_rsi = None
        last_high_price = None
        
        for i in range(self.pivot_lookback, len(df)):
            # Check for bullish divergence (price lower low, RSI higher low)
            if pivots_low.iloc[i]:
                current_rsi = rsi.iloc[i]
                current_price = df['low'].iloc[i]
                
                if last_low_rsi is not None:
                    if current_rsi > last_low_rsi and current_price < last_low_price:
                        bull_div.iloc[i] = True
                
                last_low_rsi = current_rsi
                last_low_price = current_price
            
            # Check for bearish divergence (price higher high, RSI lower high)
            if pivots_high.iloc[i]:
                current_rsi = rsi.iloc[i]
                current_price = df['high'].iloc[i]
                
                if last_high_rsi is not None:
                    if current_rsi < last_high_rsi and current_price > last_high_price:
                        bear_div.iloc[i] = True
                
                last_high_rsi = current_rsi
                last_high_price = current_price
        
        return bull_div, bear_div
    
    def detect_htf_reversal(self, htf_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect HTF reversal patterns from higher timeframe data
        
        Args:
            htf_df: DataFrame with OHLC data from higher timeframe
            
        Returns:
            Tuple of (bullish_signals, bearish_signals) series
        """
        htf_bull = pd.Series(False, index=htf_df.index)
        htf_bear = pd.Series(False, index=htf_df.index)
        
        for i in range(1, len(htf_df)):
            o = htf_df['open'].iloc[i]
            c = htf_df['close'].iloc[i]
            h = htf_df['high'].iloc[i]
            l = htf_df['low'].iloc[i]
            
            o1 = htf_df['open'].iloc[i-1]
            c1 = htf_df['close'].iloc[i-1]
            
            body = abs(c - o)
            total_range = h - l
            upper_wick = h - max(c, o)
            lower_wick = min(c, o) - l
            
            if total_range > 0:
                # Bullish patterns
                # Hammer: small body, long lower wick
                is_hammer = body < total_range * 0.33 and lower_wick > body * 2 and upper_wick < body
                
                # Bullish engulfing
                is_bull_engulf = c > o1 and o < c1 and c > o
                
                if is_hammer or is_bull_engulf:
                    htf_bull.iloc[i] = True
                
                # Bearish patterns
                # Shooting star: small body, long upper wick
                is_shooting_star = body < total_range * 0.33 and upper_wick > body * 2 and lower_wick < body
                
                # Bearish engulfing
                is_bear_engulf = c < o1 and o > c1 and c < o
                
                if is_shooting_star or is_bear_engulf:
                    htf_bear.iloc[i] = True
        
        return htf_bull, htf_bear
    
    def generate_signals(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            df: 1m timeframe DataFrame with columns: open, high, low, close, volume
            htf_df: Higher timeframe DataFrame (optional, will fetch if None)
            
        Returns:
            DataFrame with columns: signal (1=long, -1=short, 0=none), 
            htf_bull, htf_bear, rsi_div_bull, rsi_div_bear
        """
        # Get RSI divergence
        rsi = self.calculate_rsi(df['close'])
        pivots_low = self.find_pivot_low(df['low'])
        pivots_high = self.find_pivot_high(df['high'])
        
        bull_div, bear_div = self.detect_rsi_divergence(df)
        
        # Get HTF reversal (use provided or default to same dataframe for simplicity)
        if htf_df is None:
            htf_df = df  # Fallback to same TF
        
        htf_bull, htf_bear = self.detect_htf_reversal(htf_df)
        
        # Final signals: both HTF reversal AND RSI divergence must align
        # For now, align HTF close to 1m close (simplified)
        # In production, you'd align timestamps properly
        
        signal = pd.Series(0, index=df.index)
        
        # Bullish: HTF bull reversal + RSI bullish divergence
        # Note: In production, need to align HTF/1m timestamps
        signal[bull_div & htf_bull.reindex(bull_div.index, method='ffill').fillna(False)] = 1
        
        # Bearish: HTF bear reversal + RSI bearish divergence
        signal[bear_div & htf_bear.reindex(bear_div.index, method='ffill').fillna(False)] = -1
        
        result = pd.DataFrame({
            'signal': signal,
            'rsi': rsi,
            'bull_div': bull_div,
            'bear_div': bear_div,
            'htf_bull': htf_bull.reindex(bull_div.index, method='ffill').fillna(False),
            'htf_bear': htf_bear.reindex(bear_div.index, method='ffill').fillna(False)
        }, index=df.index)
        
        return result


# Example usage
if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data
    n = 500
    dates = [datetime.now() - timedelta(minutes=n-i) for i in range(n)]
    
    # Random walk with trend
    np.random.seed(42)
    returns = np.random.randn(n) * 0.02
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Run strategy
    strategy = HTFReversalDivergence(htf_timeframe="15", rsi_length=14, pivot_lookback=3)
    signals = strategy.generate_signals(df)
    
    print(f"Total signals: {(signals['signal'] != 0).sum()}")
    print(f"Long signals: {(signals['signal'] == 1).sum()}")
    print(f"Short signals: {(signals['signal'] == -1).sum()}")
    print("\nSignal dates:")
    long_signals = signals[signals['signal'] == 1].index
    short_signals = signals[signals['signal'] == -1].index
    print(f"Long: {long_signals[:5].tolist()}")
    print(f"Short: {short_signals[:5].tolist()}")