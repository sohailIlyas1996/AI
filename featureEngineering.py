import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Raw data
df = pd.DataFrame({
    "sale_date": ["2024-01-15", "2024-06-30", "2024-12-05", "2025-03-20"],
    "year_built": [1990, 2005, 1975, 2018],
    "sqft": [1200, 2000, 1500, 1800],
    "bedrooms": [2, 4, 3, 3],
    "bathrooms": [1, 2, 1, 2],
    "city": ["Birmingham", "London", "Leeds", "London"],
    "price": [180000, 450000, 210000, 400000],
})

# 2. Date features
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month
df["is_winter"] = df["sale_month"].isin([12, 1, 2]).astype(int)

# 3. Derived numeric features
df["house_age"] = df["sale_year"] - df["year_built"]
df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"]
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

# 4. Binning (continuous -> category)
df["age_group"] = pd.cut(df["house_age"], bins=[0, 10, 30, 100],
                         labels=["new", "mid", "old"])

# 5. Log transform (reduces skew)
df["log_price"] = np.log1p(df["price"])

# 6. One-hot encoding for categories
df = pd.get_dummies(df, columns=["city", "age_group"], dtype=int)

# 7. Scaling
num_cols = ["sqft", "house_age", "sqft_per_bedroom"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])

# 8. Drop columns you no longer need
df = df.drop(columns=["sale_date", "year_built"])

print(df.head())