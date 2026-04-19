import pandas as pd
import os

try:
    df = pd.read_parquet('data/raw/transcripts.parquet')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

    print("--- Searching for Cross-Mentions ---")
    results = []
    for t in tickers:
        ticker_content = df[df['ticker'] == t]['transcript_text']
        if ticker_content.empty:
            continue
        full_text = " ".join(ticker_content.astype(str).tolist())
        
        for other in tickers:
            if t == other:
                continue
            
            # Simple check for ticker or company name snippets
            if other.lower() in full_text.lower():
                count = full_text.lower().count(other.lower())
                results.append((t, other, count))
                print(f"{t} -> {other}: {count} mentions")

    if not results:
        print("No cross-mentions found.")
except Exception as e:
    print(f"Error: {e}")
