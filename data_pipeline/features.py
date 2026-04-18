import pandas as pd
import pandas_ta as ta
import numpy as np
from loguru import logger

class FeatureEngineer:
    def __init__(self):
        pass

    def _add_indicators_single(self, df: pd.DataFrame, nifty_roc60: pd.Series = None) -> pd.DataFrame:
        df = df.copy()
        
        # Pandas-ta drops non-numeric columns like 'symbol'. We must protect it by caching the scalar.
        protect_symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else None
        
        # Simple Moving Averages
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)

        # Macro Regime Filter: Distance from 200-SMA
        if 'SMA_200' in df.columns:
            df['Dist_200SMA'] = (df['close'] - df['SMA_200']) / df['SMA_200']
        else:
            df['Dist_200SMA'] = 0.0 

        # Relative Strength Index
        df.ta.rsi(length=14, append=True)
        
        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        
        # Average True Range for volatility modeling
        df.ta.atr(length=14, append=True)
        
        # --- NEW FEATURES ADDED ---
        
        # 1. Trend Strength: Average Directional Index (ADX)
        df.ta.adx(length=14, append=True)
        
        # 2. Multi-Timeframe Returns: 20-day and 60-day Rate of Change (ROC)
        df.ta.roc(length=20, append=True)
        df.ta.roc(length=60, append=True)
        
        # 3. Institutional Volume: Chaikin Money Flow (CMF)
        # Check if volume is all zeros or not properly configured
        try:
            df.ta.cmf(length=20, append=True)
        except Exception as e:
            df['CMF_20'] = 0.0
            
        # 4. Institutional Volume: Distance from 20-day Rolling VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        rolling_vol = df['volume'].rolling(window=20).sum()
        rolling_vwap = (typical_price * df['volume']).rolling(window=20).sum() / rolling_vol.replace(0, np.nan)
        df['Dist_VWAP'] = (df['close'] - rolling_vwap) / rolling_vwap
        
        # 5. Relative Strength (RS Momentum): Stock performance vs Nifty 50 over last 60 days
        stock_roc60 = df['close'].pct_change(periods=60) * 100
        
        if nifty_roc60 is not None:
            aligned_nifty_roc = nifty_roc60.reindex(df.index).ffill().fillna(0)
            df['RS_Momentum'] = stock_roc60 - aligned_nifty_roc
        else:
            df['RS_Momentum'] = stock_roc60

        if protect_symbol is not None:
            df['symbol'] = protect_symbol
            
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds standard and advanced technical indicators to the dataframe."""
        logger.info("Adding base and advanced technical indicators...")
        
        try:
            # Ensure index is datetime and sorted
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Pre-calculate Nifty 50 ROC 60 if '^NSEI' is present
            nifty_roc60 = None
            if 'symbol' in df.columns:
                nifty_df = df[df['symbol'] == '^NSEI'].copy()
                if not nifty_df.empty:
                    nifty_df.sort_index(inplace=True)
                    # percentage change * 100 to match standard ROC
                    nifty_roc60 = nifty_df['close'].pct_change(periods=60) * 100

            if 'symbol' in df.columns:
                logger.info("Calculating technical indicators per symbol")
                
                processed_groups = []
                for sym, group in df.groupby('symbol'):
                    processed_group = self._add_indicators_single(group, nifty_roc60)
                    processed_groups.append(processed_group)
                    
                df = pd.concat(processed_groups)
            else:
                logger.info("Calculating technical indicators on single dataset")
                df = self._add_indicators_single(df, None)

            # Count NaNs before dropping
            nans_before = df.isna().sum().sum()
            logger.debug(f"Total NaNs before drop: {nans_before}")

            # Drop missing values resulted from indicator calculations
            df.dropna(inplace=True)
            
            nans_after = df.isna().sum().sum()
            logger.debug(f"Total NaNs after drop: {nans_after} | Final shape is {df.shape}")
            
            # Explicit assertion to guarantee no NaNs leak through to PyTorch tensors
            assert nans_after == 0, "NaNs still exist in the DataFrame!"
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return df

def align_multi_ticker_data(multi_ticker_data: dict, min_coverage: float = 0.80):
    """
    Align all tickers onto a common date index and produce a 3D numpy array.
    
    Steps:
        1. Filter out tickers with insufficient date coverage.
        2. Inner-join on dates so every timestep has valid data for ALL tickers.
        3. Return a (T, N, F) array plus metadata.
    
    Args:
        multi_ticker_data: dict mapping ticker → DataFrame (numeric cols, DatetimeIndex).
        min_coverage: minimum fraction of the longest ticker's dates a ticker must have.
    
    Returns:
        data_3d:  np.ndarray of shape (T, N, F)
        tickers:  list of N ticker strings (ordered)
        dates:    pd.DatetimeIndex of T aligned dates
        columns:  list of F column names
    """
    # --- 1. Filter tickers by date coverage ---
    max_len = max(len(df) for df in multi_ticker_data.values())
    threshold = int(max_len * min_coverage)

    valid_tickers = {}
    for ticker, df in multi_ticker_data.items():
        if ticker == '^NSEI':
            continue  # Index is for feature calc only, not tradeable
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if len(numeric_df) >= threshold:
            valid_tickers[ticker] = numeric_df
        else:
            logger.warning(f"Dropping {ticker}: only {len(numeric_df)} rows "
                           f"(need {threshold})")

    if not valid_tickers:
        raise ValueError("No tickers passed the coverage filter!")

    # --- 2. Find common date index (inner join) ---
    common_dates = None
    for df in valid_tickers.values():
        idx = df.index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)

    common_dates = common_dates.sort_values()
    logger.info(f"Aligned {len(valid_tickers)} tickers on {len(common_dates)} "
                f"common dates [{common_dates[0].date()} → {common_dates[-1].date()}]")

    # --- 3. Build 3D array ---
    tickers = sorted(valid_tickers.keys())
    # Use column list from the first ticker (all should match after feature eng)
    columns = list(valid_tickers[tickers[0]].columns)

    T = len(common_dates)
    N = len(tickers)
    F = len(columns)
    data_3d = np.empty((T, N, F), dtype=np.float32)

    for j, ticker in enumerate(tickers):
        df = valid_tickers[ticker].loc[common_dates, columns]
        data_3d[:, j, :] = df.values.astype(np.float32)

    # Final NaN check
    nan_count = np.isnan(data_3d).sum()
    if nan_count > 0:
        logger.warning(f"Replacing {nan_count} residual NaNs with 0 in aligned data")
        np.nan_to_num(data_3d, copy=False, nan=0.0)

    logger.info(f"Aligned data tensor shape: ({T}, {N}, {F})  "
                f"[timesteps, tickers, features]")
    return data_3d, tickers, common_dates, columns


if __name__ == "__main__":
    import logging
    import pandas as pd
    from database import SessionLocal, HistoricalData
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing FeatureEngineer test run...")
    
    db = SessionLocal()
    try:
        # Fetch data from the database
        ticker = "RELIANCE.NS"
        records = db.query(HistoricalData).filter(HistoricalData.symbol == ticker).order_by(HistoricalData.date.asc()).all()
        logger.info(f"Fetched {len(records)} {ticker} records from db.")
        
        if records:
            # Convert to DataFrame
            data = [{
                'date': r.date,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'symbol': ticker
            } for r in records]
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            engineer = FeatureEngineer()
            df_with_features = engineer.add_technical_indicators(df)
            
            logger.info("Sample of calculated features:")
            print(df_with_features.head())
            print(df_with_features.columns)
            
    except Exception as e:
        logger.error(f"Failed to calculate features: {e}")
    finally:
        db.close()
