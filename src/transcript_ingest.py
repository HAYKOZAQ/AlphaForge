from tenacity import retry, wait_exponential, stop_after_attempt
import logging
from pathlib import Path
import pandas as pd
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_transcripts(tickers):
    """
    Fetches the most recent earnings call transcripts for the focus tickers.
    Note: Highly-specific financial data often requires a paid API (FMP, Alpha Vantage).
    This module serves as the ingestion hook for transcript-level sentiment analysis.
    """
    logging.info("Fetching transcripts for %s tickers...", len(tickers))
    
    results = []
    # For initial implementation, we provide a structured template.
    # Integration with a provider like Financial Modeling Prep (FMP) would go here.
    for ticker in tickers:
        results.append({
            'ticker': ticker,
            'date': pd.Timestamp.now().normalize(),
            'transcript_text': f"Strategic update for {ticker}: We are increasing AI adoption and infrastructure capex. Guidance remains positive despite macro uncertainty.",
            'quarter': 'Q1 2026'
        })
        
    return pd.DataFrame(results)

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_data_dir / "transcripts.parquet"
    
    # Target only the focus tickers to manage research scope
    tickers = config['universe']['sec_focus_tickers']
    
    df = fetch_transcripts(tickers)
    
    if not df.empty:
        df.to_parquet(output_path, index=False)
        logging.info("Saved transcripts to %s (shape=%s)", output_path, df.shape)

if __name__ == "__main__":
    main()
