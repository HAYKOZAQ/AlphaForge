import pandas as pd
import requests
import io
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sp100_tickers():
    """Fetches the latest S&P 100 constituents from Wikipedia."""
    logging.info("Fetching S&P 100 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html = requests.get(url, headers=headers, timeout=30).text
        tables = pd.read_html(io.StringIO(html))
        # The constituent table is typically the 3rd one
        df = tables[2]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        logging.error(f"Failed to fetch S&P 100 constituents: {e}")
        return []

def update_config_universe(tickers):
    """Updates the settings.yaml file with the new ticker list."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "config" / "settings.yaml"
    
    if not config_path.exists():
        logging.error(f"Config file not found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Update the focus universe
        config['universe']['sec_focus_tickers'] = tickers
        
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        
        logging.info(f"Successfully updated settings.yaml with {len(tickers)} focus tickers.")
    except Exception as e:
        logging.error(f"Error updating config: {e}")

def main():
    tickers = get_sp100_tickers()
    if tickers:
        update_config_universe(tickers)
    else:
        logging.error("Universe update aborted due to fetch failure.")

if __name__ == "__main__":
    main()
