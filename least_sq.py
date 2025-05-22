import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from multiple_reg import train_model, predict  # Assuming these functions exist in multiple_reg.py

# Load dataset
data = pd.read_csv("Q2.csv")

# Extract features (X) and target (Y)
X = data[["x1", "x2"]].values  # Feature matrix
Y = data["y"].values.reshape(-1, 1)  # Target vector reshaped to column

# Add bias term (column of ones) to X for w0
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# Compute least squares solution: W = (X^T X)^(-1) X^T Y
W = inv(X_bias.T @ X_bias) @ X_bias.T @ Y

# Extract coefficients
w0, w1, w2 = W.flatten()

# Print polynomial equation
equation = f"y = {w0:.6f} + {w1:.6f} * x1 + {w2:.6f} * x2"
print("\nLeast Squares Polynomial Equation:")
print(equation)

# Compute RMSE using the least squares model
y_pred = X_bias @ W  # Predictions
rmse_ls = np.sqrt(mean_squared_error(Y, y_pred))

# Display RMSE
print(f"\nRoot Mean Squared Error (RMSE) for Least Squares Model: {rmse_ls:.6f}")

# RMSE from multiple_reg.py model (from previous output)
rmse_multiple_reg = 289.715  # Given as the RMSE for multiple_reg.py model

# Display comparison
print(f"\nComparison with multiple_reg.py:")
print(f"RMSE of Least Squares Model: {rmse_ls:.81f}")
print(f"RMSE of multiple_reg.py Model: {rmse_multiple_reg:.3f}")

if rmse_ls < rmse_multiple_reg:
    print("\nThe Least Squares Model is more accurate.")
else:
    print("\nThe multiple_reg.py Model is more accurate.")


# Conclusion: Lower RMSE means better accuracy