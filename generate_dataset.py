import numpy as np
import pandas as pd

np.random.seed(42)
N = 50_000

# Generate realistic features
data = {
    'customer_id': range(1, N + 1),
    'age': np.random.normal(40, 15, N).clip(18, 90).astype(int),
    'tenure_months': np.random.exponential(24, N).clip(0, 120).astype(int),
    'monthly_charges': np.random.normal(70, 30, N).clip(10, 200).round(2),
    'total_charges': None,
    'contract_type': np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        N, p=[0.5, 0.3, 0.2]
    ),
    'payment_method': np.random.choice(
        ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'],
        N, p=[0.35, 0.25, 0.25, 0.15]
    ),
    'internet_service': np.random.choice(
        ['DSL', 'Fiber optic', 'No'], N, p=[0.35, 0.45, 0.20]
    ),
    'num_support_calls': np.random.poisson(2, N),
    'satisfaction_score': np.random.choice(
        [1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.25, 0.40, 0.20]
    ),
    'signup_date': pd.date_range('2018-01-01', '2024-12-31', periods=N).strftime('%Y-%m-%d'),
}

df = pd.DataFrame(data)
df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)

# Create target variable (churn) with realistic relationships
churn_prob = (
    0.05
    + 0.15 * (df['contract_type'] == 'Month-to-month')
    + 0.10 * (df['satisfaction_score'] <= 2)
    + 0.08 * (df['num_support_calls'] > 4)
    - 0.05 * (df['tenure_months'] > 36)
).clip(0, 1)
df['churned'] = (np.random.random(N) < churn_prob).astype(int)

# INTENTIONAL MESSINESS (realistic data issues)

# 1. Missing values (~3% randomly in some columns)
for col in ['monthly_charges', 'satisfaction_score', 'payment_method']:
    mask = np.random.random(N) < 0.03
    df.loc[mask, col] = np.nan

# 2. Duplicate rows (~0.5%)
n_dupes = int(N * 0.005)
dupe_indices = np.random.choice(N, n_dupes, replace=False)
df = pd.concat([df, df.iloc[dupe_indices]], ignore_index=True)

# 3. Inconsistent string formatting
df.loc[df.sample(frac=0.1).index, 'contract_type'] = (
    df.loc[df.sample(frac=0.1).index, 'contract_type'].str.lower()
)

# 4. A few outliers
outlier_idx = np.random.choice(len(df), 50, replace=False)
df.loc[outlier_idx, 'monthly_charges'] = np.random.uniform(500, 2000, 50)

# 5. Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save it
df.to_csv('customer_data.csv', index=False)
print(f"Created dataset: {len(df):,} rows × {len(df.columns)} columns")
print(f"File size: ~{len(df) * len(df.columns) * 8 / 1_000_000:.1f} MB (estimate)")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)