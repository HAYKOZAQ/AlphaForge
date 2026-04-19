import pandas_datareader.data as web
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

keys = [
    'FEDFUNDS', 'CPIAUCSL', 'INDPRO', 'UNRATE', 'DCOILWTICO', 
    'JTSJOL', 'JTS3000JOL', 'JTS5100JOL', 'JTS5200JOL', 
    'JTS6200JOL', 'JTS4400JOL', 'JTS540099JOL'
]

print("--- Testing Individual Keys ---")
for k in keys:
    try:
        df = web.DataReader(k, 'fred', '2023-01-01', '2023-02-01')
        print(f"{k}: SUCCESS (Shape: {df.shape}, Index: {df.index.name})")
    except Exception as e:
        print(f"{k}: FAILED - {e}")

print("\n--- Testing Combined Keys ---")
try:
    df = web.DataReader(keys, 'fred', '2023-01-01', '2023-02-01')
    print(f"Combined Shape: {df.shape}")
    print(f"Combined Index Name: {df.index.name}")
    print(f"Combined Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Combined FAILED: {e}")
