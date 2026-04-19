from tenacity import retry, wait_exponential, stop_after_attempt
from sec_edgar_downloader import Downloader
from pathlib import Path
import logging
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from concurrent.futures import ThreadPoolExecutor

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def download_for_ticker(dl, ticker):
    """Worker function for parallel SEC downloads."""
    logging.info(f"Targeting annual reports for {ticker}...")
    try:
        dl.get("10-K", ticker, limit=1, after="2022-01-01")
        logging.info(f"Successfully downloaded 10-K archive for {ticker}.")
    except Exception as e:
        logging.error(f"SEC EDGAR rejection for {ticker}: {e}")

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    sec_dir = repo_root / "data" / "sec_filings"
    sec_dir.mkdir(parents=True, exist_ok=True)
    
    # Needs valid email identification due to SEC rate limit boundaries
    dl = Downloader("QuantDevCorp", "institutional-research@quantdev.org", str(sec_dir))
    
    target_tickers = config['universe']['sec_focus_tickers']
    
    # Use ThreadPoolExecutor for concurrent downloads (modest workers to avoid SEC IP block)
    logging.info(f"Starting parallel download for {len(target_tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda t: download_for_ticker(dl, t), target_tickers))
    
    logging.info("SEC Phase complete.")

if __name__ == "__main__":
    main()
