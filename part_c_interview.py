"""
Part C - Interview Ready (Coding Question)
Separate features (X) and target (y) from a dataset.
"""

import pandas as pd

# Sample dataset
data = {
    'age':        [25, 30, 35, 28, 40],
    'salary':     [30000, 45000, 60000, 35000, 70000],
    'experience': [2, 5, 10, 3, 15],
    'promoted':   [0, 1, 1, 0, 1]    # target column
}

df = pd.DataFrame(data)

# Method 1: Using drop()
X = df.drop(columns=['promoted'])   # all columns except target
y = df['promoted']                  # target column only

print("Features X:")
print(X)
print()
print("Target y:")
print(y)
print()

# Method 2: Selecting columns manually
target_col = 'promoted'
feature_cols = [col for col in df.columns if col != target_col]

X2 = df[feature_cols]
y2 = df[target_col]

print("Features X (Method 2):")
print(X2)
print()
print("Target y (Method 2):")
print(y2)
