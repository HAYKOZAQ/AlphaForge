from tenacity import retry, wait_exponential, stop_after_attempt
import io
import json
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


from src.utils.config_loader import load_config
from src.utils.validators import data_sanity_decorator, OHLCVSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_sp500_tickers():
    """Read the current S&P 500 constituent list from Wikipedia."""
    logging.info("Reading current S&P 500 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91 Safari/537.36"
        )
    }
    html = requests.get(url, headers=headers, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]

    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    metadata = df[["Symbol", "GICS Sector", "GICS Sub-Industry"]].copy()
    metadata["Symbol"] = metadata["Symbol"].str.replace(".", "-", regex=False)

    return tickers, metadata


@data_sanity_decorator(OHLCVSchema, "S&P 500 Ingest")
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_ohlcv(tickers, start_date):
    logging.info("Downloading OHLCV data for %s firms from %s...", len(tickers), start_date)

    df = yf.download(
        tickers,
        start=start_date,
        group_by="ticker",
        auto_adjust=True,
        threads=True
    )

    records = []

    unpacked_tickers = df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else []

    for ticker in unpacked_tickers:
        ticker_df = df[ticker].copy()

        if "Close" not in ticker_df.columns:
            continue

        ticker_df["Ticker"] = ticker
        ticker_df = ticker_df.reset_index()
        ticker_df = ticker_df.dropna(subset=["Close", "Volume"])

        if not ticker_df.empty:
            records.append(ticker_df)

    if not records:
        logging.error("No S&P 500 OHLCV data fetched.")
        return pd.DataFrame()

    final_df = pd.concat(records, ignore_index=True)
    final_df.columns = [c.lower() for c in final_df.columns]

    logging.info("-" * 40)
    logging.info("S&P 500 INGEST SUMMARY")
    logging.info("Downloaded rows: %s", f"{len(final_df):,}")
    logging.info("Unique tickers: %s", final_df["ticker"].nunique())
    logging.info("Date range: %s -> %s", final_df["date"].min(), final_df["date"].max())
    logging.info("-" * 40)

    return final_df


def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "data" / "raw"
    reports_dir = repo_root / "reports"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_ohlcv = raw_data_dir / "sp500_ohlcv.parquet"
    out_meta = raw_data_dir / "sp500_metadata.csv"
    out_summary = reports_dir / "sp500_ingest_summary.json"

    tickers, meta_df = get_sp500_tickers()

    meta_df.to_csv(out_meta, index=False)
    logging.info("Saved %s company metadata records.", len(tickers))

    df = fetch_ohlcv(tickers, start_date=config['dates']['sp500_start'])

    if not df.empty:
        df.to_parquet(out_ohlcv, index=False)
        logging.info("Saved S&P 500 OHLCV dataset to %s (shape=%s)", out_ohlcv, df.shape)

        summary = {
            "rows": int(len(df)),
            "tickers": int(df["ticker"].nunique()),
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
        }
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info("Saved ingest summary to %s", out_summary)
    else:
        logging.warning("Output dataframe is empty.")


if __name__ == "__main__":
    main()