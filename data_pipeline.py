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

# STEP 3 is CLEANING the data to fix issues found during EDA and prepare it for analysis/modeling.
# Order matters: dedupe → fix types → handle missing → handle outliers
print("\n" + "=" * 65)
print("STEP 3: CLEANING")
print("=" * 65)

initial_rows = len(df)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {initial_rows - len(df)} duplicates")

# Standardize string formatting
# Categories need to be converted back to string to modify, then back to category
df['contract_type'] = df['contract_type'].astype(str).str.title().astype('category')
print(f"Standardized contract_type. New unique values: {df['contract_type'].unique().tolist()}")

# Handle outliers in monthly_charges
# Use IQR method: anything beyond 1.5×IQR from quartiles is an outlier
Q1 = df['monthly_charges'].quantile(0.25)
Q3 = df['monthly_charges'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
n_outliers = (df['monthly_charges'] > upper_bound).sum()
print(f"\nOutliers in monthly_charges (>{upper_bound:.2f}): {n_outliers}")

# Strategy: CAP them rather than delete (keeps the row, removes the extreme value)
df['monthly_charges'] = df['monthly_charges'].clip(upper=upper_bound)
print(f"Capped outliers at {upper_bound:.2f}")

# Handle missing values
# Different strategies for different columns:
# - Numerical: fill with median (robust to outliers)
# - Categorical: fill with mode (most common value) or "Unknown"
print(f"\nFilling missing values...")
df['monthly_charges'] = df['monthly_charges'].fillna(df['monthly_charges'].median())
df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())

# For payment_method, add a new category "Unknown" — sometimes missingness is informative!
df['payment_method'] = df['payment_method'].cat.add_categories('Unknown').fillna('Unknown')
print(f"Missing values remaining: {df.isnull().sum().sum()}")