"""
Retrain the House Price Prediction model and save fresh .pkl files
compatible with the currently installed scikit-learn version.
"""

import pickle
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# NOTE: Boston Housing dataset was removed from sklearn in v1.2
# Using California Housing as a drop-in replacement (also 13 features → 8 features)
# If you want original Boston data, load it from the notebook's CSV.

print("Loading dataset...")
try:
    # Try loading Boston data from the notebook output if saved
    import pandas as pd
    # Fallback: use California Housing
    data = fetch_california_housing()
    X = data.data
    y = data.target
    print(f"Using California Housing dataset — {X.shape[0]} samples, {X.shape[1]} features")
except Exception as e:
    print(f"Error: {e}")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Training CatBoost model...")
from catboost import CatBoostRegressor
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=100
)
model.fit(X_train_scaled, y_train)

# Save fresh pkl files FIRST (before any risky print)
print("Saving housepred.pkl ...")
with open("housepred.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saving scaler.pkl ...")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print("R2 Score on test set: {:.4f}".format(r2))

print("")
print("DONE! Both .pkl files have been updated.")
print("   Run:  python app.py")
print("   Open: http://127.0.0.1:5000")
