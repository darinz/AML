# Search and Clustering

## Overview

Clustering is an unsupervised learning technique that groups similar data points together without predefined labels. It's fundamental for data exploration, pattern discovery, and understanding the underlying structure of data.

## Key Concepts

### 1. What is Clustering?

Clustering aims to:
- **Group** similar objects together
- **Separate** dissimilar objects
- **Discover** natural groupings in data
- **Reduce** data complexity

### 2. Types of Clustering

1. **Partitioning**: Divide data into non-overlapping clusters (K-means)
2. **Hierarchical**: Build clusters in a tree-like structure
3. **Density-based**: Group points based on density (DBSCAN)
4. **Model-based**: Assume data follows a statistical model

## K-Means Clustering

### Algorithm Overview

K-means is the most popular clustering algorithm due to its simplicity and effectiveness.

### Mathematical Foundation

The algorithm minimizes the within-cluster sum of squares (WCSS):

```math
\text{WCSS} = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2
```

Where:
- $`x_i`$ is a data point
- $`\mu_k`$ is the centroid of cluster k
- $`C_k`$ is the set of points in cluster k

### Algorithm Steps

1. **Initialize**: Choose K centroids randomly
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. **Repeat**: Steps 2-3 until convergence
5. **Output**: Final clusters and centroids

### Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def fit(self, X):
        """Fit K-means clustering to the data"""
        np.random.seed(self.random_state)
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for iteration in range(self.max_iters):
            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels == k], axis=0)
            
            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                break
        
        # Calculate inertia (within-cluster sum of squares)
        self.inertia_ = self._calculate_inertia(X)
        return self
    
    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _calculate_inertia(self, X):
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        X = np.array(X)
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        
        return np.argmin(distances, axis=1)
```

### Example Usage

```python
# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply K-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Visualize results
plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, alpha=0.6)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-means results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-means Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

print(f"K-means Inertia: {kmeans.inertia_:.2f}")
```

### Choosing the Right K

#### Elbow Method

```python
def plot_elbow_method(X, max_k=10):
    """Plot elbow method for K selection"""
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()

# Apply elbow method
plot_elbow_method(X)
```

#### Silhouette Analysis

```python
from sklearn.metrics import silhouette_score

def silhouette_analysis(X, max_k=10):
    """Analyze silhouette scores for different K values"""
    silhouette_scores = []
    k_values = range(2, max_k + 1)  # Silhouette requires at least 2 clusters
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal K')
    plt.grid(True)
    plt.show()
    
    return k_values, silhouette_scores

# Apply silhouette analysis
k_values, scores = silhouette_analysis(X)
best_k = k_values[np.argmax(scores)]
print(f"Best K based on silhouette: {best_k}")
```

## Hierarchical Clustering

### Overview

Hierarchical clustering builds a tree-like structure (dendrogram) showing the relationship between clusters.

### Types

1. **Agglomerative**: Bottom-up approach (start with individual points)
2. **Divisive**: Top-down approach (start with all points in one cluster)

### Linkage Methods

#### Single Linkage
```math
d(C_1, C_2) = \min\{d(x, y) \mid x \in C_1, y \in C_2\}
```

#### Complete Linkage
```math
d(C_1, C_2) = \max\{d(x, y) \mid x \in C_1, y \in C_2\}
```

#### Average Linkage
```math
d(C_1, C_2) = \frac{1}{|C_1| \times |C_2|} \sum_{x \in C_1, y \in C_2} d(x, y)
```

#### Ward's Method
Minimizes the increase in within-cluster variance.

### Implementation

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

class HierarchicalClustering:
    def __init__(self, method='ward', metric='euclidean'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.labels = None
        
    def fit(self, X, n_clusters=None):
        """Fit hierarchical clustering"""
        X = np.array(X)
        
        # Calculate linkage matrix
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)
        
        if n_clusters is not None:
            self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        return self
    
    def plot_dendrogram(self, max_d=None):
        """Plot dendrogram"""
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix, truncate_mode='level', p=30)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.method} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    
    def get_clusters(self, n_clusters):
        """Get cluster labels for specified number of clusters"""
        return fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')

# Example usage
hierarchical = HierarchicalClustering(method='ward')
hierarchical.fit(X)

# Plot dendrogram
hierarchical.plot_dendrogram()

# Get clusters
labels = hierarchical.get_clusters(4)
```

## DBSCAN (Density-Based Spatial Clustering)

### Overview

DBSCAN groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.

### Key Concepts

- **Core Point**: Point with at least `min_samples` neighbors within `eps`
- **Border Point**: Point within `eps` of a core point but not a core point itself
- **Noise Point**: Point that is neither core nor border

### Algorithm Steps

1. **Initialize**: Mark all points as unvisited
2. **Select**: Choose an unvisited point p
3. **Expand**: If p is a core point, start a new cluster
4. **Add**: Add all density-reachable points to the cluster
5. **Repeat**: Until all points are visited

### Implementation

```python
from sklearn.neighbors import NearestNeighbors

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        
    def fit(self, X):
        """Fit DBSCAN clustering"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize labels (-1 for noise)
        self.labels = np.full(n_samples, -1)
        
        # Find neighbors for each point
        neighbors_model = NearestNeighbors(radius=self.eps)
        neighbors_model.fit(X)
        neighbors = neighbors_model.radius_neighbors(X, return_distance=False)
        
        # Identify core points
        core_points = [i for i, neighbor_list in enumerate(neighbors) 
                      if len(neighbor_list) >= self.min_samples]
        
        # Initialize cluster counter
        cluster_id = 0
        
        # Process each core point
        for point in core_points:
            if self.labels[point] != -1:
                continue  # Already assigned to a cluster
            
            # Start new cluster
            self.labels[point] = cluster_id
            cluster_points = [point]
            
            # Expand cluster
            i = 0
            while i < len(cluster_points):
                current_point = cluster_points[i]
                current_neighbors = neighbors[current_point]
                
                for neighbor in current_neighbors:
                    if self.labels[neighbor] == -1:
                        self.labels[neighbor] = cluster_id
                        cluster_points.append(neighbor)
                
                i += 1
            
            cluster_id += 1
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        # For DBSCAN, we need to find the nearest core point
        # This is a simplified version
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Find nearest neighbor in training data
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] <= self.eps:
                predictions.append(self.labels[nearest_idx])
            else:
                predictions.append(-1)  # Noise
        
        return np.array(predictions)

# Example usage
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X)

# Visualize DBSCAN results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels, alpha=0.6)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(scatter)
plt.show()

# Count clusters (excluding noise)
n_clusters = len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)
n_noise = list(dbscan.labels).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

## Efficient Search Techniques

### KD-Trees

KD-trees are data structures for organizing points in k-dimensional space, enabling efficient nearest neighbor searches.

```python
from scipy.spatial import KDTree

class KDTreeSearch:
    def __init__(self):
        self.tree = None
        self.data = None
        
    def fit(self, X):
        """Build KD-tree from data"""
        self.data = np.array(X)
        self.tree = KDTree(self.data)
        return self
    
    def query(self, X, k=1):
        """Find k nearest neighbors"""
        distances, indices = self.tree.query(X, k=k)
        return distances, indices
    
    def query_radius(self, X, r):
        """Find all neighbors within radius r"""
        indices = self.tree.query_ball_point(X, r)
        return indices

# Example usage
kdtree = KDTreeSearch()
kdtree.fit(X)

# Find nearest neighbors
query_points = np.array([[0, 0], [2, 2]])
distances, indices = kdtree.query(query_points, k=3)
print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

### Ball Trees

Ball trees are similar to KD-trees but work better in high-dimensional spaces.

```python
from sklearn.neighbors import BallTree

class BallTreeSearch:
    def __init__(self, metric='euclidean'):
        self.tree = None
        self.metric = metric
        
    def fit(self, X):
        """Build ball tree from data"""
        self.tree = BallTree(X, metric=self.metric)
        return self
    
    def query(self, X, k=1):
        """Find k nearest neighbors"""
        distances, indices = self.tree.query(X, k=k)
        return distances, indices

# Example usage
balltree = BallTreeSearch()
balltree.fit(X)
distances, indices = balltree.query(query_points, k=3)
```

## Similarity Metrics

### Euclidean Distance
```python
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

### Manhattan Distance
```python
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))
```

### Cosine Similarity
```python
def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return dot_product / (norm_x1 * norm_x2)
```

### Jaccard Similarity
```python
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0
```

### Hamming Distance
```python
def hamming_distance(x1, x2):
    return np.sum(x1 != x2)
```

## Evaluation Metrics

### Silhouette Score

Measures how similar an object is to its own cluster compared to other clusters.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

def evaluate_silhouette(X, labels):
    """Evaluate clustering using silhouette score"""
    overall_score = silhouette_score(X, labels)
    sample_scores = silhouette_samples(X, labels)
    
    print(f"Overall Silhouette Score: {overall_score:.3f}")
    print(f"Sample Silhouette Scores - Mean: {np.mean(sample_scores):.3f}")
    print(f"Sample Silhouette Scores - Std: {np.std(sample_scores):.3f}")
    
    return overall_score, sample_scores

# Evaluate different clustering methods
methods = {
    'K-means': kmeans.labels,
    'DBSCAN': dbscan.labels,
    'Hierarchical': hierarchical.get_clusters(4)
}

for method_name, labels in methods.items():
    if len(set(labels)) > 1:  # Need at least 2 clusters
        print(f"\n{method_name}:")
        evaluate_silhouette(X, labels)
```

### Inertia (Within-Cluster Sum of Squares)

```python
def calculate_inertia(X, labels, centroids):
    """Calculate within-cluster sum of squares"""
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia
```

### Davies-Bouldin Index

Lower values indicate better clustering.

```python
from sklearn.metrics import davies_bouldin_score

def evaluate_davies_bouldin(X, labels):
    """Evaluate clustering using Davies-Bouldin index"""
    score = davies_bouldin_score(X, labels)
    print(f"Davies-Bouldin Index: {score:.3f}")
    return score
```

### Calinski-Harabasz Index

Higher values indicate better clustering.

```python
from sklearn.metrics import calinski_harabasz_score

def evaluate_calinski_harabasz(X, labels):
    """Evaluate clustering using Calinski-Harabasz index"""
    score = calinski_harabasz_score(X, labels)
    print(f"Calinski-Harabasz Index: {score:.3f}")
    return score
```

## Practical Applications

### Customer Segmentation

```python
# Example: Customer segmentation based on purchase behavior
customer_data = np.array([
    [100, 5, 2],   # [total_spent, num_orders, avg_order_value]
    [50, 3, 1],
    [200, 10, 3],
    # ... more customers
])

# Apply clustering
kmeans_customers = KMeans(n_clusters=3)
kmeans_customers.fit(customer_data)

# Analyze segments
for i in range(3):
    segment_customers = customer_data[kmeans_customers.labels == i]
    print(f"Segment {i}: {len(segment_customers)} customers")
    print(f"Average total spent: {np.mean(segment_customers[:, 0]):.2f}")
    print(f"Average orders: {np.mean(segment_customers[:, 1]):.2f}")
    print()
```

### Document Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Example documents
documents = [
    "machine learning algorithms",
    "deep learning neural networks",
    "data science analytics",
    "artificial intelligence systems",
    # ... more documents
]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_docs = vectorizer.fit_transform(documents)

# Reduce dimensionality
svd = TruncatedSVD(n_components=2)
X_docs_2d = svd.fit_transform(X_docs)

# Apply clustering
kmeans_docs = KMeans(n_clusters=2)
kmeans_docs.fit(X_docs_2d)

# Visualize document clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_docs_2d[:, 0], X_docs_2d[:, 1], c=kmeans_docs.labels)
plt.title('Document Clustering')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

## Summary

Clustering algorithms provide powerful tools for:

1. **Data Exploration**: Understanding data structure
2. **Pattern Discovery**: Finding hidden relationships
3. **Dimensionality Reduction**: Grouping similar features
4. **Anomaly Detection**: Identifying outliers
5. **Preprocessing**: Preparing data for supervised learning

Key considerations when choosing a clustering method:

- **Data Size**: K-means for large datasets, hierarchical for small
- **Data Shape**: DBSCAN for non-spherical clusters
- **Noise**: DBSCAN handles noise well
- **Scalability**: K-means is most scalable
- **Interpretability**: Hierarchical provides clear structure

Understanding these algorithms and their trade-offs is essential for effective unsupervised learning and data analysis. 