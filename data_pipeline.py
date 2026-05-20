import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_PATH = 'customer_data.csv'
OUTPUT_DIR = Path('processed')
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 1 is to do the efficeint data loading and preprocessing
print("=" * 65)
print("STEP 1: EFFICIENT LOADING")
print("=" * 65)

t0 = time.perf_counter()
df_naive = pd.read_csv(DATA_PATH, low_memory=False)
naive_memory = df_naive.memory_usage(deep=True).sum() / 1_000_000
print(f"\n[Naive load]    Time: {time.perf_counter()-t0:.2f}s  |  Memory: {naive_memory:.1f} MB")

# Optimized the load by specifying dtypes and parsing dates upfront.
dtypes = {
    'customer_id': 'int32',
    'age': 'int8',                  # ages 0-127 fit in int8
    'tenure_months': 'int8',
    'monthly_charges': 'float32',
    'total_charges': 'float32',
    'contract_type': 'category',    # categories are MUCH smaller than strings
    'payment_method': 'category',
    'internet_service': 'category',
    'num_support_calls': 'int8',
    'satisfaction_score': 'float32',
    'churned': 'int8',
}

t0 = time.time()
df = pd.read_csv(
    DATA_PATH,
    dtype=dtypes,
    parse_dates=['signup_date'],   # convert dates while loading
)
optimized_memory = df.memory_usage(deep=True).sum() / 1_000_000
print(f"[Optimized load] Time: {time.time()-t0:.2f}s  |  Memory: {optimized_memory:.1f} MB")
print(f"\nMemory saved: {(1 - optimized_memory/naive_memory)*100:.0f}%")
print("(Imagine this on a 50 GB file — could mean fitting in RAM vs not!)")

# Chunked loading so that we can process files too big to fit in memory
print("\n[Chunked loading demo]")
total_rows = 0
for chunk in pd.read_csv(DATA_PATH, dtype=dtypes, chunksize=10_000):
    total_rows += len(chunk)     
print(f"Processed {total_rows:,} rows in chunks without loading all at once")


# STEP 2 is to perform EXPLORATORY DATA ANALYSIS (EDA) to understand the data and identify issues.
print("\n" + "=" * 65)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nDuplicates: {df.duplicated().sum()}")

print(f"\nNumerical summary:")
print(df.describe().round(2))

print(f"\nTarget distribution (churned):")
print(df['churned'].value_counts(normalize=True).round(3))

print(f"\nUnique values in 'contract_type':")
print(df['contract_type'].value_counts())
