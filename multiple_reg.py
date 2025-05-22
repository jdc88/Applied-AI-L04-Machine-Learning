import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Efficiently load large dataset
data = pd.read_csv("Q2.csv", dtype={"y": "float32", "x1": "float32", "x2": "float32"})

# Extract features (X) and target (Y)
X = data[["x1", "x2"]].values
Y = data["y"].values

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def train_model(X_train, Y_train):
    """
    Trains the linear regression model and returns the coefficients.
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def predict(model, X_test):
    """
    Predicts the target values using the trained model.
    """
    return model.predict(X_test)

# Train the regression model
model = train_model(X_train, Y_train)

# Display the coefficients
print("\nCoefficients:")
print(f"Intercept (w0) = {model.intercept_:.3f}")
for i, coeff in enumerate(model.coef_, start=1):
    print(f"w{i} = {coeff:.6f}")

# Generate regression equation
equation = f"y = {model.intercept_:.3f} + {model.coef_[0]:.6f} * x1 + {model.coef_[1]:.6f} * x2"
print("\nRegression Equation:")
print(equation)

# Compute RMSE to evaluate model accuracy
y_test_pred = predict(model, X_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.3f}")

# Select a random sample from the test set
random_index = np.random.randint(0, X_test.shape[0])  # Pick a random index
x_sample = X_test[random_index].reshape(1, -1)  # Extract corresponding feature values
x1_value, x2_value = x_sample.flatten()  # Unpack x1 and x2 values

# Predict using the randomly selected test values
y_pred = model.predict(x_sample)[0]
print("\nPrediction for randomly selected input values:")
print(f"For x1 = {x1_value:.3f}, x2 = {x2_value:.3f}, ŷ = {y_pred:.3f}")
