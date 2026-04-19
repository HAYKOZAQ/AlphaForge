from tenacity import retry, wait_exponential, stop_after_attempt
import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_institutional_holders(ticker):
    """Fetches top institutional holders for a ticker via yfinance."""
    logging.info(f"Fetching institutional holders for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        df_inst = t.institutional_holders
        if df_inst is not None and not df_inst.empty:
            # We only need the Holder name and the ticker
            df_inst['ticker'] = ticker
            return df_inst[['ticker', 'Holder', 'Shares', 'Value']]
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching institutional holders for {ticker}: {e}")
        return pd.DataFrame()

from concurrent.futures import ThreadPoolExecutor

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = config['universe']['sec_focus_tickers']
    logging.info(f"Starting parallel institutional ingestion for {len(tickers)} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_institutional_holders, tickers))
            
    dfs = [df for df in results if not df.empty]
    
    if dfs:
        master_holders = pd.concat(dfs)
        output_path = raw_dir / "institutional_holders.parquet"
        master_holders.to_parquet(output_path, index=False)
        logging.info(f"Saved {len(master_holders)} holder records to {output_path}")
    else:
        logging.warning("No institutional records found.")

if __name__ == "__main__":
    main()
