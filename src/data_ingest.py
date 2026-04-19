from tenacity import retry, wait_exponential, stop_after_attempt
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config_loader import load_config
from src.utils.validators import data_sanity_decorator, OHLCVSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@data_sanity_decorator(OHLCVSchema, "Sector Ingest")
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_ohlcv(tickers, start_date):
    logging.info("Downloading data for %s sector tickers from %s...", len(tickers), start_date)

    df = yf.download(tickers, start=start_date, group_by="ticker", auto_adjust=True)

    records = []
    for ticker in tickers:
        if ticker in df.columns.levels[0]:
            ticker_df = df[ticker].copy()
            ticker_df["Ticker"] = ticker
            ticker_df = ticker_df.reset_index()
            ticker_df = ticker_df.dropna(subset=["Close", "Volume"])
            records.append(ticker_df)

    if not records:
        logging.error("No sector data fetched.")
        return pd.DataFrame()

    final_df = pd.concat(records, ignore_index=True)
    final_df.columns = [c.lower() for c in final_df.columns]

    logging.info("Sector dataset rows: %s", f"{len(final_df):,}")
    logging.info("Sector tickers: %s", final_df["ticker"].nunique())
    logging.info(
        "Date range: %s -> %s",
        final_df["date"].min(),
        final_df["date"].max(),
    )

    return final_df

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_options_data(tickers):
    """Fetches high-level options market signals for the given tickers."""
    logging.info("Fetching options signals for %s tickers...", len(tickers))
    
    results = []
    for ticker_symbol in tickers:
        try:
            t = yf.Ticker(ticker_symbol)
            expirations = t.options
            if not expirations:
                continue
                
            # Use the nearest expiration for stability
            target_date = expirations[0]
            chain = t.option_chain(target_date)
            
            calls = chain.calls
            puts = chain.puts
            
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            
            total_call_vol = calls['volume'].sum()
            total_put_vol = puts['volume'].sum()
            
            # Simple Put/Call ratios
            oi_ratio = total_put_oi / (total_call_oi + 1e-6)
            vol_ratio = total_put_vol / (total_call_vol + 1e-6)
            
            results.append({
                'ticker': ticker_symbol,
                'date': pd.Timestamp.now().normalize(),
                'put_call_oi_ratio': oi_ratio,
                'put_call_vol_ratio': vol_ratio,
                'nearest_expiration': target_date
            })
            
        except Exception as e:
            logging.warning("Failed to fetch options for %s: %s", ticker_symbol, e)
            
    return pd.DataFrame(results)

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sector OHLCV
    ohlcv_output = raw_data_dir / "sectors_ohlcv.parquet"
    df_ohlcv = fetch_ohlcv(
        config['universe']['sectors'], 
        start_date=config['dates']['sector_start']
    )

    if not df_ohlcv.empty:
        df_ohlcv.to_parquet(ohlcv_output, index=False)
        logging.info("Saved sector OHLCV data to %s (shape=%s)", ohlcv_output, df_ohlcv.shape)

    # 2. Options Signals
    options_output = raw_data_dir / "options_signals.parquet"
    df_options = fetch_options_data(config['universe']['sectors'])
    
    if not df_options.empty:
        df_options.to_parquet(options_output, index=False)
        logging.info("Saved options signals to %s (shape=%s)", options_output, df_options.shape)

if __name__ == "__main__":
    main()