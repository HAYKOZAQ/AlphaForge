import pandas as pd
df = pd.read_parquet('data/processed/insider_features.parquet')
print("Total Insider Features Rows:", len(df))
if not df.empty:
    non_zeros = df[(df['insider_net_intensity'] != 0) & (df['insider_net_intensity'].notna())]
    print("Non-zero Intensity in features:", len(non_zeros))
    print(df.head())
