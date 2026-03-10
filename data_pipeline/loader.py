import yfinance as yf
import pandas as pd
from loguru import logger
from typing import Dict, List

class DataLoader:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetches historical data from Yahoo Finance for given tickers."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")
        dataframes = {}
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if df.empty:
                    logger.warning(f"No data found for {ticker}")
                else:
                    dataframes[ticker] = df
                    logger.debug(f"Fetched {len(df)} rows for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                
        return dataframes

    def save_to_db(self, df: pd.DataFrame, symbol: str):
        from database import SessionLocal, HistoricalData
        if df.empty:
            logger.warning(f"No data to save for {symbol}.")
            return
            
        # Clean multiindex columns if they exist (yfinance sometimes returns them for multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        db = SessionLocal()
        records = []
        for index, row in df.iterrows():
            record = HistoricalData(
                symbol=symbol,
                date=index,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            )
            records.append(record)
        
        try:
            db.add_all(records)
            db.commit()
            logger.info(f"Saved {len(records)} records for {symbol} to DB.")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save {symbol} to db: {e}")
        finally:
            db.close()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    target_tickers = ["SPY"] 
    start = "2020-01-01"
    end = "2026-01-01"
    
    logging.info(f"Initializing DataLoader for {target_tickers}...")
    
    try:
        loader = DataLoader(tickers=target_tickers, start_date=start, end_date=end)
        
        # 1. Fetch the data (which returns a dictionary of DataFrames)
        data_dict = loader.fetch_historical_data() 
        
        # 2. Check if the dictionary actually contains our ticker
        if data_dict and "SPY" in data_dict:
            df = data_dict["SPY"]
            logging.info(f"Success: Extracted {len(df)} rows for SPY from the dictionary.")
            
            # Print the first few rows to verify the columns (Open, High, Low, Close, Volume)
            print(df.head())
            
            # Next, we will uncomment the database commit method
            loader.save_to_db(df, "SPY") 
        else:
            logging.error("Failed: Dictionary is empty or 'SPY' key is missing.")
            
    except Exception as e:
        logging.error(f"Pipeline crashed: {e}")