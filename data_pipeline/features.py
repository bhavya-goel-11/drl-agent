import pandas as pd
import pandas_ta as ta
from loguru import logger

class FeatureEngineer:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds standard technical indicators to the dataframe using pandas-ta."""
        logger.info("Adding technical indicators to dataframe")
        
        try:
            # Ensure index is datetime and sorted
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Simple Moving Averages
            df.ta.sma(length=10, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)

            # Relative Strength Index
            df.ta.rsi(length=14, append=True)
            
            # MACD
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            # Bollinger Bands
            df.ta.bbands(length=20, std=2, append=True)
            
            # Average True Range for volatility modeling
            df.ta.atr(length=14, append=True)

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

if __name__ == "__main__":
    import logging
    import pandas as pd
    from database import SessionLocal, HistoricalData
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing FeatureEngineer test run...")
    
    db = SessionLocal()
    try:
        # Fetch SPY data from the database
        records = db.query(HistoricalData).filter(HistoricalData.symbol == "SPY").order_by(HistoricalData.date.asc()).all()
        logger.info(f"Fetched {len(records)} SPY records from db.")
        
        if records:
            # Convert to DataFrame
            data = [{
                'date': r.date,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
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
