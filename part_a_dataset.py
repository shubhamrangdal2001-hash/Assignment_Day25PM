"""
Part A - Dataset Understanding
Creating a small pandas DataFrame for supervised machine learning,
then separating features (X) and target (y).
"""

import pandas as pd

# Small dataset for supervised ML (house price prediction)
data = {
    'area_sqft': [1200, 1500, 800, 2000, 1100],
    'num_rooms': [3, 4, 2, 5, 3],
    'location_score': [7, 8, 5, 9, 6],
    'house_age': [10, 5, 20, 2, 15],
    'price_lakhs': [45, 62, 28, 95, 38]   # this is our target
}

df = pd.DataFrame(data)
print("Full Dataset:")
print(df)
print()

# Separating features and target
X = df.drop(columns=['price_lakhs'])   # features
y = df['price_lakhs']                  # target

print("Features (X):")
print(X)
print()
print("Target (y):")
print(y)
