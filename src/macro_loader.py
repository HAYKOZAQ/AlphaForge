from tenacity import retry, wait_exponential, stop_after_attempt
import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MACRO_SERIES = {
    'FEDFUNDS': 'fed_funds_rate',
    'CPIAUCSL': 'cpi',
    'INDPRO': 'industrial_production',
    'UNRATE': 'unemployment_rate',
    'DCOILWTICO': 'wti_oil',
    # JOLTS Job Openings (Labor Demand) - Using valid industry IDs from FRED
    # JOLTS Job Openings (Labor Demand) - Using fully qualified Seasonally Adjusted IDs
    'JTSJOL': 'job_openings_total',
    'JTS3000JOL': 'job_openings_mfg',
    'JTU5100JOL': 'job_openings_info',
    'JTU5200JOL': 'job_openings_fin',
    'JTS6200JOL': 'job_openings_health',
    'JTS4400JOL': 'job_openings_retail',
    'JTS540099JOL': 'job_openings_prof_services'
}

from src.utils.config_loader import load_config
from src.utils.validators import data_sanity_decorator, MacroSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@data_sanity_decorator(MacroSchema, "Macro Loading")
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def fetch_macro(series_dict, start_date):
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    logging.info(f"Downloading macro data from FRED: {list(series_dict.keys())}...")
    
    master_df = None
    
    # We fetch series individually to prevent one bad ID from killing the pipe
    for fred_id, col_name in series_dict.items():
        try:
            df_single = web.DataReader(fred_id, 'fred', start_date, end_date)
            df_single.columns = [col_name]
            df_single = df_single.reset_index()
            df_single.columns = [c.lower() for c in df_single.columns] # Ensure 'date'
            
            if master_df is None:
                master_df = df_single
            else:
                master_df = pd.merge(master_df, df_single, on='date', how='outer')
                
        except Exception as e:
            logging.error(f"Error fetching series {fred_id}: {e}")

    if master_df is not None:
        master_df.sort_values('date', inplace=True)
        # 1. Resample to continuous daily frequency and ffill to bridge gaps in reporting
        master_df.set_index('date', inplace=True)
        master_df = master_df.resample('D').ffill()
        
        # 2. Resample to monthly to match the research pipeline standard
        master_df = master_df.resample('ME').last().reset_index()
        
        # Ensure all requested columns exist
        for col in series_dict.values():
            if col not in master_df.columns:
                master_df[col] = pd.NA
        
        return master_df
    else:
        # Fallback empty df with required date column
        return pd.DataFrame(columns=['date'] + list(series_dict.values()))

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_data_dir / "macro_data.parquet"
    
    df = fetch_macro(MACRO_SERIES, start_date=config['dates']['fred_start'])
    
    if not df.empty:
        df.to_parquet(output_path, index=False)
        logging.info(f"Successfully saved Macro data to {output_path} (Shape: {df.shape})")
    else:
        logging.warning("Dataframe empty. Skipping save.")


if __name__ == "__main__":
    main()
