import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load the Digits dataset from sklearn.datasets
digits = datasets.load_digits()
X = digits.data  # Feature matrix (flattened 8x8 images)
y = digits.target  # Target labels (digits 0-9)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (scaling them to zero mean and unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-NN Classification (for comparison)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-NN Accuracy: {accuracy_knn * 100:.2f}%")
print('K-NN Classification Report:')
print(classification_report(y_test, y_pred_knn))

# K-NN Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("K-NN Confusion Matrix:")
print(cm_knn)  # Print confusion matrix in the terminal

# MLP Classification (Multilayer Perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\nMLP Accuracy: {accuracy_mlp * 100:.2f}%")
print('MLP Classification Report:')
print(classification_report(y_test, y_pred_mlp))

# MLP Confusion Matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print("MLP Confusion Matrix:")
print(cm_mlp)  # Print confusion matrix in the terminal

# Comparing MLP with K-NN accuracy
print("\nComparison:")
print(f"K-NN Accuracy: {accuracy_knn * 100:.2f}%")
print(f"MLP Accuracy: {accuracy_mlp * 100:.2f}%")

# Hyperparameter Tuning: Adjusting MLP hyperparameters for better performance
best_accuracy_mlp = accuracy_mlp
best_mlp_params = (64, 'relu', 'constant', 0.9)  # Default initialization with all required values

# Try different configurations for hyperparameters
for hidden_layers in [(64,), (128,), (64, 64)]:
    for activation in ['relu', 'tanh']:
        for learning_rate in ['constant', 'adaptive']:
            for momentum in [0.9, 0.95]:
                mlp_temp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation,
                                         learning_rate=learning_rate, max_iter=1000, momentum=momentum, random_state=42)
                mlp_temp.fit(X_train, y_train)
                y_pred_temp = mlp_temp.predict(X_test)
                accuracy_temp = accuracy_score(y_test, y_pred_temp)
                if accuracy_temp > best_accuracy_mlp:
                    best_accuracy_mlp = accuracy_temp
                    best_mlp_params = (hidden_layers, activation, learning_rate, momentum)

print(f"\nBest MLP Accuracy: {best_accuracy_mlp * 100:.2f}%")
print(f"Best MLP Hyperparameters: hidden_layer_sizes={best_mlp_params[0]}, "
      f"activation={best_mlp_params[1]}, learning_rate={best_mlp_params[2]}, momentum={best_mlp_params[3]}")

