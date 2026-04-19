from tenacity import retry, wait_exponential, stop_after_attempt
import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
import datetime
from src.utils.validators import data_sanity_decorator, TextSignalSchema
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_sector_news(tickers):
    records = []
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            news = tkr.news
            for item in news:
                records.append({
                    'ticker': ticker,
                    'date': pd.to_datetime(item.get('providerPublishTime', 0), unit='s').date(),
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', '')
                })
        except Exception as e:
            logging.error(f"Failed to fetch news for {ticker}: {e}")
            
    return pd.DataFrame(records)

def score_themes_and_sentiment(df, config):
    if df.empty:
        return df
        
    df['title_lower'] = df['title'].str.lower()
    themes_cfg = config['nlp']['themes']
    
    # Theme scoring via keyword frequency from config
    for theme, keywords in themes_cfg.items():
        df[theme] = df['title_lower'].apply(lambda x: sum(1 for kw in keywords if kw in x))
    
    # Sentiment Scoring from config
    pos_words = config['nlp']['sentiment']['positive']
    neg_words = config['nlp']['sentiment']['negative']
    
    def calc_sentiment(text):
        pos = sum(1 for w in pos_words if w in text)
        neg = sum(1 for w in neg_words if w in text)
        return pos - neg
        
    df['sentiment'] = df['title_lower'].apply(calc_sentiment)
    
    df = df.drop(columns=['title_lower'])
    return df

@data_sanity_decorator(TextSignalSchema, "Text Signal Processing")
def aggregate_monthly_text(df, config):
    if df.empty:
        return pd.DataFrame()
    
    logging.info("Running monthly aggregation for text signals...")
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    themes_cfg = config['nlp']['themes']
    agg_dict = {theme: 'sum' for theme in themes_cfg.keys()}
    agg_dict['sentiment'] = 'mean'
    agg_dict['title'] = 'count'
    
    monthly = df.groupby(['ticker', 'month']).agg(agg_dict).reset_index()
    monthly = monthly.rename(columns={'title': 'news_volume'})
    
    monthly['date'] = monthly['month'].dt.to_timestamp(how='end').dt.normalize()
    monthly = monthly.drop(columns=['month'])
    return monthly

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_raw = raw_data_dir / "sector_news_raw.parquet"
    output_processed = processed_dir / "sector_nlp_monthly.parquet"
    
    logging.info("Fetching sector news...")
    df_news = fetch_sector_news(config['universe']['sectors'])
    
    if not df_news.empty:
        df_news.to_parquet(output_raw, index=False)
        
        logging.info("Scoring text AI features...")
        df_scored = score_themes_and_sentiment(df_news, config)
        
        logging.info("Aggregating into monthly time-series...")
        df_monthly = aggregate_monthly_text(df_scored, config)
        
        df_monthly.to_parquet(output_processed, index=False)
        logging.info(f"Saved monthly NLP features: {df_monthly.shape}")
    else:
        logging.warning("No news collected.")

if __name__ == "__main__":
    main()
