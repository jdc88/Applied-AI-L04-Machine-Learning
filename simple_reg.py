import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# First dataset (Linear Regression)
X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y1 = np.array([2.5, 4.1, 5.6, 7.2, 8.8, 10.3, 11.9, 13.5, 15.0, 16.8])

# Second dataset (Polynomial Regression)
X2 = np.array([-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]).reshape(-1, 1)
Y2 = np.array([17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4])

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Function for linear regression
def linear_regression(ax, X, Y, x_pred, dataset_name):
    model = LinearRegression()
    model.fit(X, Y)
    
    # Get predictions
    Y_pred = model.predict(X)
    
    # Regression equation
    w0, w1 = model.intercept_, model.coef_[0]
    equation = f"y = {w0:.3f} + {w1:.3f}x"

    # Predict for x_pred
    y_pred_value = model.predict(np.array([[x_pred]]))[0]

    # Compute errors
    total_error = np.sum(Y - Y_pred)
    sse = np.sum((Y - Y_pred) ** 2)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)

    # Print results
    print(f"\n--- {dataset_name} (Linear Regression) ---")
    print(f"Regression Equation: {equation}")
    print(f"Predicted y for x={x_pred}: {y_pred_value:.3f}")
    print(f"Total Error: {total_error:.3f}")
    print(f"Sum of Squared Errors (SSE): {sse:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

    # Plot on the given subplot
    ax.scatter(X, Y, color='blue', label='Actual Data')
    ax.plot(X, Y_pred, color='red', label='Regression Line')
    ax.set_title(dataset_name)
    ax.legend()

# Function for polynomial regression (degree=2)
def polynomial_regression(ax, X, Y, x_pred, degree, dataset_name):
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X, Y)

    # Get predictions
    Y_pred = poly_model.predict(X)

    # Get polynomial equation coefficients
    poly_features = PolynomialFeatures(degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, Y)
    coefficients = model.coef_
    intercept = model.intercept_

    equation = f"y = {intercept:.3f} " + " + ".join([f"{coeff:.3f}x^{i}" for i, coeff in enumerate(coefficients) if i > 0])

    # Predict for x_pred
    y_pred_value = poly_model.predict(np.array([[x_pred]]))[0]

    # errors
    total_error = np.sum(Y - Y_pred)
    sse = np.sum((Y - Y_pred) ** 2)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)

    print(f"\n--- {dataset_name} (Polynomial Regression, Degree {degree}) ---")
    print(f"Regression Equation: {equation}")
    print(f"Predicted y for x={x_pred}: {y_pred_value:.3f}")
    print(f"Total Error: {total_error:.3f}")
    print(f"Sum of Squared Errors (SSE): {sse:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

    ax.scatter(X, Y, color='blue', label='Actual Data')
    ax.plot(np.sort(X, axis=0), poly_model.predict(np.sort(X, axis=0)), color='red', label='Polynomial Fit')
    ax.set_title(dataset_name)
    ax.legend()

# Run Linear Regression for the first dataset on the left subplot
linear_regression(axs[0], X1, Y1, 100, "First Dataset")

# Run Polynomial Regression for the second dataset (degree=2) on the right subplot
polynomial_regression(axs[1], X2, Y2, 0.5, degree=2, dataset_name="Second Dataset")

# Adjust layout and show both plots in one window
plt.tight_layout()
plt.show()