import pandas as pd
df = pd.read_parquet('data/processed/master_dataset.parquet')
print(f"Total Rows: {len(df)}")
non_zeros = df[(df['insider_net_intensity'] != 0) & (df['insider_net_intensity'].notna())]
print(f"Non-zero Insider Rows: {len(non_zeros)}")
if len(non_zeros) > 0:
    print("Latest Date with Insider Data:", non_zeros['date'].max())
    print(non_zeros[['date', 'ticker', 'insider_net_intensity']].tail())
