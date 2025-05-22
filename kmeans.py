import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # True labels (not used for clustering)

# Standardize the features (scaling them to zero mean and unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering with 3 clusters (since there are 3 species in the Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# RMSE for each cluster based on its centroids
rmse = []
for i in range(3):
    cluster_points = X_scaled[y_kmeans == i]
    cluster_center = kmeans.cluster_centers_[i]
    rmse.append(np.sqrt(np.mean(np.sum((cluster_points - cluster_center) ** 2, axis=1))))
    
print("RMSE for each cluster:", rmse)

# Apply KNN classification (as per knn_iris.py) to the Iris dataset for comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Print K-NN classification report
print("\nK-NN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Map clusters to actual classes
cluster_to_class = {}
for i in range(3):
    cluster_points = y[y_kmeans == i]
    mode_result = mode(cluster_points)  # Get the mode of the cluster points
    
    # If the result of mode is a tuple, extract the first element; otherwise, use the scalar
    most_common_class = mode_result[0][0] if isinstance(mode_result[0], np.ndarray) else mode_result[0]
    cluster_to_class[i] = most_common_class  # Assign the most common class for the cluster

# Assign the predicted class labels based on K-means clusters
y_kmeans_mapped = np.array([cluster_to_class[cluster] for cluster in y_kmeans])

# Print the classification report for cluster labels (mapping K-means clusters to true labels)
print("\nCluster Classification Report (Mapped to True Labels):")
print(classification_report(y, y_kmeans_mapped, zero_division=1))

# Evaluate the clustering by plotting the clusters
# Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', label='Cluster')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-means Clustering of Iris Dataset (PCA Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Print the Cluster Centers and the number of samples in each cluster
print(f'Cluster Centers:\n{kmeans.cluster_centers_}')
print(f'Number of samples in each cluster: {np.bincount(y_kmeans)}')
