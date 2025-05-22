import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Digits dataset from sklearn.datasets
digits = datasets.load_digits()
X = digits.data  # Feature matrix (flattened 8x8 images)
y = digits.target  # Target labels (digits 0-9)

# Display the first image from the dataset
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Digit: {digits.target[0]}")
plt.show()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (scaling them to zero mean and unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the K-NN classifier (k=3 in this example)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the K-NN model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Evaluate the K-NN model's performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-NN Accuracy: {accuracy_knn * 100:.2f}%")
print('K-NN Classification Report:')
print(classification_report(y_test, y_pred_knn))
print('K-NN Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_knn))

# Try different values of k to improve K-NN accuracy
best_accuracy_knn = accuracy_knn
best_k = 3
for k in range(1, 21):  # Trying k values from 1 to 20
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracy_temp = accuracy_score(y_test, y_pred_temp)
    if accuracy_temp > best_accuracy_knn:
        best_accuracy_knn = accuracy_temp
        best_k = k

print(f"\nBest K-NN Accuracy: {best_accuracy_knn * 100:.2f}% with k={best_k}")
