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

t0 = time.perf_counter()
df_naive = pd.read_csv(DATA_PATH, low_memory=False)
naive_memory = df_naive.memory_usage(deep=True).sum() / 1_000_000
print(f"\n[Naive load]    Time: {time.perf_counter()-t0:.2f}s  |  Memory: {naive_memory:.1f} MB")