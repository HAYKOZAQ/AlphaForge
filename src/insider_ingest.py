from tenacity import retry, wait_exponential, stop_after_attempt
import pandas as pd
from edgar import set_identity, Company
from pathlib import Path
import logging
import datetime
from src.utils.config_loader import load_config

# SEC Compliance Requirement
set_identity("quant-research-system@institutional-grade.org")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_insider_transactions(ticker, limit=20):
    """
    Fetches Form 4 transactions using edgartools to_dataframe().
    """
    logging.info(f"Fetching Form 4 filings for {ticker}...")
    try:
        company = Company(ticker)
        filings = company.get_filings(form="4")
        
        if len(filings) == 0:
            return pd.DataFrame()

        dfs = []
        for filing in filings[:limit]:
            try:
                f4 = filing.obj()
                df_f4 = f4.to_dataframe()
                if df_f4 is not None and not df_f4.empty:
                    # Rename columns to our internal standard
                    df_f4 = df_f4.rename(columns={
                        'Ticker': 'ticker',
                        'Date': 'date',
                        'Insider': 'owner',
                        'Code': 'code',
                        'Shares': 'shares',
                        'Price': 'price',
                        'Value': 'value',
                        'Remaining Shares': 'holdings'
                    })
                    # Add is_ceo flag (heuristically from Position)
                    df_f4['is_ceo'] = df_f4['Position'].str.contains('CEO', case=False, na=False)
                    dfs.append(df_f4)
            except Exception as f_err:
                logging.debug(f"Skipping filing for {ticker}: {f_err}")
                continue
        
        if dfs:
            return pd.concat(dfs)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to fetch filings for {ticker}: {e}")
        return pd.DataFrame()

from concurrent.futures import ThreadPoolExecutor

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = config['universe']['sec_focus_tickers']
    logging.info(f"Starting parallel insider ingestion for {len(tickers)} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_insider_transactions, tickers))
            
    dfs = [df for df in results if not df.empty]
    
    if dfs:
        master_insider = pd.concat(dfs)
        output_path = raw_dir / "insider_transactions.parquet"
        master_insider.to_parquet(output_path, index=False)
        logging.info(f"Saved {len(master_insider)} insider transactions to {output_path}")
    else:
        logging.warning("No insider transactions found.")

if __name__ == "__main__":
    main()
