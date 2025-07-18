"""
Clustering Algorithms Implementation and Examples

This file contains complete implementations of various clustering algorithms
including K-means, hierarchical clustering, DBSCAN, and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class KMeans:
    """
    K-Means clustering implementation from scratch
    """
    
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

class HierarchicalClustering:
    """
    Hierarchical clustering implementation
    """
    
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

class DBSCAN:
    """
    DBSCAN clustering implementation from scratch
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.X_train = None
        
    def fit(self, X):
        """Fit DBSCAN clustering"""
        X = np.array(X)
        self.X_train = X
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

def demonstrate_kmeans():
    """
    Demonstrate K-means clustering
    """
    print("=== K-Means Clustering Demo ===")
    
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
    
    return kmeans, X, y_true

def demonstrate_elbow_method(X):
    """
    Demonstrate elbow method for choosing optimal K
    """
    print("\n=== Elbow Method Demo ===")
    
    inertias = []
    k_values = range(1, 11)
    
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
    
    # Find elbow point (simplified)
    # In practice, you'd use more sophisticated methods
    print("Elbow method suggests optimal K around 4-5 clusters")
    
    return k_values, inertias

def demonstrate_silhouette_analysis(X):
    """
    Demonstrate silhouette analysis for choosing optimal K
    """
    print("\n=== Silhouette Analysis Demo ===")
    
    silhouette_scores = []
    k_values = range(2, 11)  # Silhouette requires at least 2 clusters
    
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
    
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"Best K based on silhouette: {best_k}")
    
    return k_values, silhouette_scores, best_k

def demonstrate_hierarchical_clustering(X):
    """
    Demonstrate hierarchical clustering
    """
    print("\n=== Hierarchical Clustering Demo ===")
    
    # Apply hierarchical clustering
    hierarchical = HierarchicalClustering(method='ward')
    hierarchical.fit(X)
    
    # Plot dendrogram
    hierarchical.plot_dendrogram()
    
    # Get clusters for different numbers of clusters
    n_clusters_list = [2, 3, 4, 5]
    
    plt.figure(figsize=(15, 10))
    
    for i, n_clusters in enumerate(n_clusters_list):
        labels = hierarchical.get_clusters(n_clusters)
        
        plt.subplot(2, 2, i+1)
        plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.6)
        plt.title(f'Hierarchical Clustering (K={n_clusters})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return hierarchical

def demonstrate_dbscan():
    """
    Demonstrate DBSCAN clustering
    """
    print("\n=== DBSCAN Clustering Demo ===")
    
    # Generate different types of data
    datasets = {
        'blobs': make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42),
        'moons': make_moons(n_samples=300, noise=0.1, random_state=42),
        'circles': make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, (X, y_true)) in enumerate(datasets.items()):
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        dbscan.fit(X)
        
        # Count clusters (excluding noise)
        n_clusters = len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)
        n_noise = list(dbscan.labels).count(-1)
        
        # Plot results
        plt.subplot(3, 2, 2*i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, alpha=0.6)
        plt.title(f'{name.capitalize()} - True Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.subplot(3, 2, 2*i + 2)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels, alpha=0.6)
        plt.title(f'{name.capitalize()} - DBSCAN (Clusters: {n_clusters}, Noise: {n_noise})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        print(f"{name.capitalize()}: {n_clusters} clusters, {n_noise} noise points")
    
    plt.tight_layout()
    plt.show()
    
    return dbscan

def demonstrate_dbscan_parameter_selection():
    """
    Demonstrate DBSCAN parameter selection
    """
    print("\n=== DBSCAN Parameter Selection Demo ===")
    
    # Generate data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # Test different eps values
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_samples_values = [3, 5, 7, 10]
    
    plt.figure(figsize=(15, 12))
    
    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X)
            
            n_clusters = len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)
            n_noise = list(dbscan.labels).count(-1)
            
            plt.subplot(len(eps_values), len(min_samples_values), i * len(min_samples_values) + j + 1)
            scatter = plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels, alpha=0.6)
            plt.title(f'eps={eps}, min_samples={min_samples}\nClusters: {n_clusters}, Noise: {n_noise}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

def demonstrate_evaluation_metrics(X, y_true):
    """
    Demonstrate clustering evaluation metrics
    """
    print("\n=== Clustering Evaluation Metrics Demo ===")
    
    # Test different clustering methods
    methods = {}
    
    # K-means with different K values
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        methods[f'K-means (K={k})'] = kmeans.labels
    
    # Hierarchical clustering
    hierarchical = HierarchicalClustering(method='ward')
    hierarchical.fit(X)
    methods['Hierarchical (K=4)'] = hierarchical.get_clusters(4)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(X)
    methods['DBSCAN'] = dbscan.labels
    
    # Evaluate each method
    results = {}
    
    for method_name, labels in methods.items():
        if len(set(labels)) > 1:  # Need at least 2 clusters
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            results[method_name] = {
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz
            }
            
            print(f"\n{method_name}:")
            print(f"  Silhouette Score: {silhouette:.3f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
            print(f"  Calinski-Harabasz Index: {calinski_harabasz:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Silhouette scores
    method_names = list(results.keys())
    silhouette_scores = [results[name]['silhouette'] for name in method_names]
    axes[0].bar(method_names, silhouette_scores)
    axes[0].set_title('Silhouette Scores')
    axes[0].set_ylabel('Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Davies-Bouldin scores (lower is better)
    davies_scores = [results[name]['davies_bouldin'] for name in method_names]
    axes[1].bar(method_names, davies_scores)
    axes[1].set_title('Davies-Bouldin Index')
    axes[1].set_ylabel('Index (lower is better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Calinski-Harabasz scores
    calinski_scores = [results[name]['calinski_harabasz'] for name in method_names]
    axes[2].bar(method_names, calinski_scores)
    axes[2].set_title('Calinski-Harabasz Index')
    axes[2].set_ylabel('Index (higher is better)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_customer_segmentation():
    """
    Demonstrate customer segmentation using clustering
    """
    print("\n=== Customer Segmentation Demo ===")
    
    # Generate customer data (simulated)
    np.random.seed(42)
    n_customers = 1000
    
    # Customer features: [total_spent, num_orders, avg_order_value, days_since_last_order]
    customer_data = np.random.randn(n_customers, 4)
    
    # Make data more realistic
    customer_data[:, 0] = customer_data[:, 0] * 1000 + 500  # Total spent: $500-$1500
    customer_data[:, 1] = np.abs(customer_data[:, 1] * 5 + 10)  # Orders: 5-15
    customer_data[:, 2] = np.abs(customer_data[:, 2] * 20 + 50)  # Avg order: $30-$70
    customer_data[:, 3] = np.abs(customer_data[:, 3] * 30 + 15)  # Days: 0-45
    
    # Apply K-means clustering
    kmeans_customers = KMeans(n_clusters=4)
    kmeans_customers.fit(customer_data)
    
    # Analyze segments
    feature_names = ['Total Spent', 'Num Orders', 'Avg Order Value', 'Days Since Last Order']
    
    plt.figure(figsize=(15, 10))
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        
        for cluster_id in range(4):
            cluster_data = customer_data[kmeans_customers.labels == cluster_id, i]
            plt.hist(cluster_data, alpha=0.6, label=f'Segment {cluster_id}', bins=20)
        
        plt.title(f'{feature_names[i]} Distribution by Segment')
        plt.xlabel(feature_names[i])
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print segment characteristics
    print("\nCustomer Segment Analysis:")
    for i in range(4):
        segment_customers = customer_data[kmeans_customers.labels == i]
        print(f"\nSegment {i} ({len(segment_customers)} customers):")
        print(f"  Average total spent: ${np.mean(segment_customers[:, 0]):.2f}")
        print(f"  Average orders: {np.mean(segment_customers[:, 1]):.1f}")
        print(f"  Average order value: ${np.mean(segment_customers[:, 2]):.2f}")
        print(f"  Average days since last order: {np.mean(segment_customers[:, 3]):.1f}")
    
    return kmeans_customers, customer_data

def demonstrate_document_clustering():
    """
    Demonstrate document clustering
    """
    print("\n=== Document Clustering Demo ===")
    
    # Simulate document data (TF-IDF vectors)
    np.random.seed(42)
    n_documents = 200
    n_features = 50
    
    # Generate documents with different topics
    documents = np.random.randn(n_documents, n_features)
    
    # Create topic clusters
    topic_centers = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 5,  # Topic 1
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0] * 5,  # Topic 2
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0] * 5,  # Topic 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 5   # Topic 4
    ])
    
    # Assign documents to topics
    true_topics = np.random.choice(4, n_documents)
    for i, topic in enumerate(true_topics):
        documents[i] += topic_centers[topic] + np.random.normal(0, 0.1, n_features)
    
    # Apply clustering
    kmeans_docs = KMeans(n_clusters=4)
    kmeans_docs.fit(documents)
    
    # Reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    documents_2d = pca.fit_transform(documents)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # True topics
    plt.subplot(1, 2, 1)
    plt.scatter(documents_2d[:, 0], documents_2d[:, 1], c=true_topics, alpha=0.6)
    plt.title('True Document Topics')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Clustered topics
    plt.subplot(1, 2, 2)
    plt.scatter(documents_2d[:, 0], documents_2d[:, 1], c=kmeans_docs.labels, alpha=0.6)
    plt.title('Clustered Document Topics')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate clustering
    silhouette = silhouette_score(documents, kmeans_docs.labels)
    print(f"Document clustering silhouette score: {silhouette:.3f}")
    
    return kmeans_docs, documents, true_topics

def main():
    """
    Run all clustering demonstrations
    """
    print("Clustering Algorithms Complete Implementation and Examples")
    print("=" * 60)
    
    # Generate data for demonstrations
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # Run all demonstrations
    kmeans, X_kmeans, y_true_kmeans = demonstrate_kmeans()
    k_values, inertias = demonstrate_elbow_method(X)
    silhouette_k_values, silhouette_scores, best_k = demonstrate_silhouette_analysis(X)
    hierarchical = demonstrate_hierarchical_clustering(X)
    dbscan = demonstrate_dbscan()
    demonstrate_dbscan_parameter_selection()
    evaluation_results = demonstrate_evaluation_metrics(X, y_true)
    customer_kmeans, customer_data = demonstrate_customer_segmentation()
    doc_kmeans, documents, true_topics = demonstrate_document_clustering()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    print(f"Best K from silhouette analysis: {best_k}")
    print(f"K-means inertia: {kmeans.inertia_:.2f}")
    print(f"DBSCAN clusters found: {len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)}")
    
    # Find best method based on silhouette score
    best_method = max(evaluation_results.keys(), 
                     key=lambda x: evaluation_results[x]['silhouette'])
    print(f"Best clustering method (silhouette): {best_method}")

if __name__ == "__main__":
    main() 