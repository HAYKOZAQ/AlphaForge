import pandas as pd
import yfinance as yf
import requests
import io
import logging

logging.basicConfig(level=logging.INFO)

def get_top_100_sp500():
    logging.info("Fetching S&P 500 constituents...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    df = pd.read_html(io.StringIO(html))[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    
    logging.info(f"Fetching market caps for {len(tickers)} companies...")
    # Fetching market caps in batches to avoid timeout
    batch_size = 50
    data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            # yfinance download is faster for multiple symbols
            tickers_obj = yf.Tickers(" ".join(batch))
            for t in batch:
                info = tickers_obj.tickers[t].info
                mkt_cap = info.get('marketCap', 0)
                data.append({'ticker': t, 'market_cap': mkt_cap})
        except Exception as e:
            logging.error(f"Error batch {i}: {e}")
            
    res_df = pd.DataFrame(data).sort_values('market_cap', ascending=False)
    top_100 = res_df.head(100)['ticker'].tolist()
    return top_100, res_df.head(100)

if __name__ == "__main__":
    top_100, full_df = get_top_100_sp500()
    print("Top 10 Tickers by Market Cap:")
    print(full_df.head(10))
    print("\nTotal Top 100 count:", len(top_100))
    # Save for later
    full_df.to_csv("top_100_tickers.csv", index=False)
