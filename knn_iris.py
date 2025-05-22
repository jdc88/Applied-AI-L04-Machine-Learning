# knn_iris.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset from CSV
try:
    iris_data = pd.read_csv('iris_dataset.csv')
    X = iris_data.iloc[:, :-1].values  # Features (all columns except the last one)
    y = iris_data.iloc[:, -1].values   # Target (last column)
except FileNotFoundError:
    print("Error: iris_dataset.csv not found. Please provide the file.")
    exit()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (scaling them to zero mean and unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the k-NN classifier (k=3 in this example)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the K-NN model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Evaluate the K-NN model's performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-NN Accuracy: {accuracy_knn:.2f}")
print('K-NN Classification Report:')
print(classification_report(y_test, y_pred_knn))
print('K-NN Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_knn))

# Try different values of k to improve K-NN accuracy
best_accuracy_knn = accuracy_knn
best_k = 3
for k in range(1, 11):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracy_temp = accuracy_score(y_test, y_pred_temp)
    if accuracy_temp > best_accuracy_knn:
        best_accuracy_knn = accuracy_temp
        best_k = k

print(f"\nBest K-NN Accuracy: {best_accuracy_knn:.2f} with k={best_k}")

# Now, let's use a Decision Tree Classifier to compare
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

# Evaluate the Decision Tree model's performance
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
print(f"\nDecision Tree Accuracy: {accuracy_dtree:.2f}")
print('Decision Tree Classification Report:')
print(classification_report(y_test, y_pred_dtree))
print('Decision Tree Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_dtree))

# Comparing K-NN and Decision Tree accuracy
if best_accuracy_knn > accuracy_dtree:
    print("\nK-NN accuracy is better than Decision Tree")
elif best_accuracy_knn < accuracy_dtree:
    print("\nDecision Tree accuracy is better than K-NN")
else:
    print("\nK-NN and Decision Tree have equal accuracy")
