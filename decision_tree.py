import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the Iris dataset from CSV
try:
    iris_data = pd.read_csv('iris_dataset.csv')
    X = iris_data.iloc[:, :-1].values  # Features
    y = iris_data.iloc[:, -1].values   # Target
except FileNotFoundError:
    print("Error: iris_dataset.csv not found. Please provide the file.")
    exit()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree (optional)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=iris_data.columns[:-1], class_names=np.unique(y), filled=True)
plt.show()

# Print some information about the model
print("Decision Tree Classifier trained successfully.")