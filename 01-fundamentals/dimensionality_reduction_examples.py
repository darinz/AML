"""
Dimensionality Reduction Implementation and Examples

This file contains complete implementations of various dimensionality reduction
techniques including PCA, embeddings, feature selection, and visualization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_swiss_roll, fetch_olivetti_faces
from sklearn.decomposition import PCA as SklearnPCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class PCA:
    """
    Principal Component Analysis implementation from scratch
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, X):
        """Fit PCA to the data"""
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components and explained variance
        if self.n_components is None:
            self.n_components = n_features
        
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """Transform data to lower dimensions"""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        """Fit PCA and transform data"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced):
        """Transform data back to original dimensions"""
        return np.dot(X_reduced, self.components_.T) + self.mean_
    
    def get_feature_importance(self):
        """Get feature importance based on component loadings"""
        return np.abs(self.components_).T

def demonstrate_pca():
    """
    Demonstrate PCA with synthetic data
    """
    print("=== PCA Demo ===")
    
    # Generate high-dimensional data
    X, y = make_blobs(n_samples=1000, n_features=10, centers=3, random_state=42)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data (first 2 features)
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
    plt.title('Original Data (First 2 Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # PCA reduced data
    plt.subplot(1, 3, 2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, alpha=0.6)
    plt.title('PCA Reduced Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Explained variance
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    return pca, X_reduced

def demonstrate_elbow_method(X):
    """
    Demonstrate elbow method for choosing number of components
    """
    print("\n=== Elbow Method Demo ===")
    
    # Apply PCA with different numbers of components
    max_components = min(X.shape[1], 20)
    pca = PCA()
    pca.fit(X)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:max_components])
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    print(f"Components for 95% variance: {n_components_95}")
    print(f"Components for 99% variance: {n_components_99}")
    
    return n_components_95, n_components_99

def demonstrate_scree_plot(X):
    """
    Demonstrate scree plot for eigenvalue analysis
    """
    print("\n=== Scree Plot Demo ===")
    
    pca = PCA()
    pca.fit(X)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_) + 1), 
             pca.explained_variance_, 'ro-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.show()
    
    # Kaiser criterion: keep components with eigenvalue > 1
    kaiser_components = np.sum(pca.explained_variance_ > 1)
    print(f"Components by Kaiser criterion: {kaiser_components}")

def demonstrate_kernel_pca():
    """
    Demonstrate Kernel PCA for non-linear dimensionality reduction
    """
    print("\n=== Kernel PCA Demo ===")
    
    # Generate non-linear data (swiss roll)
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    
    # Apply different kernel PCAs
    kernels = ['linear', 'rbf', 'poly']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, kernel in enumerate(kernels):
        kpca = KernelPCA(n_components=2, kernel=kernel)
        X_kpca = kpca.fit_transform(X)
        
        axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, alpha=0.6)
        axes[i].set_title(f'Kernel PCA ({kernel})')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.show()

def demonstrate_sparse_pca(X):
    """
    Demonstrate Sparse PCA for interpretable components
    """
    print("\n=== Sparse PCA Demo ===")
    
    # Regular PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Sparse PCA
    sparse_pca = SparsePCA(n_components=2, alpha=0.1)
    X_sparse = sparse_pca.fit_transform(X)
    
    # Compare sparsity
    print(f"Regular PCA components sparsity: {np.mean(pca.components_ == 0):.3f}")
    print(f"Sparse PCA components sparsity: {np.mean(sparse_pca.components_ == 0):.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.title('Regular PCA')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_sparse[:, 0], X_sparse[:, 1], alpha=0.6)
    plt.title('Sparse PCA')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.tight_layout()
    plt.show()

def demonstrate_feature_selection():
    """
    Demonstrate various feature selection methods
    """
    print("\n=== Feature Selection Demo ===")
    
    # Generate data with some irrelevant features
    np.random.seed(42)
    X, y = make_blobs(n_samples=1000, n_features=20, centers=3, random_state=42)
    
    # Add some noise features
    noise_features = np.random.randn(1000, 10)
    X_with_noise = np.hstack([X, noise_features])
    
    print(f"Original features: {X_with_noise.shape[1]}")
    
    # 1. Variance Threshold
    print("\n1. Variance Threshold Selection:")
    selector = VarianceThreshold(threshold=0.01)
    X_variance_selected = selector.fit_transform(X_with_noise)
    print(f"Features after variance threshold: {X_variance_selected.shape[1]}")
    
    # 2. Correlation-based selection
    print("\n2. Correlation-based Selection:")
    correlation_matrix = np.corrcoef(X_with_noise.T)
    high_corr_pairs = np.where(np.abs(correlation_matrix) > 0.95)
    high_corr_pairs = [(high_corr_pairs[0][i], high_corr_pairs[1][i]) 
                       for i in range(len(high_corr_pairs[0])) 
                       if high_corr_pairs[0][i] < high_corr_pairs[1][i]]
    
    features_to_remove = set()
    for pair in high_corr_pairs:
        features_to_remove.add(pair[1])
    
    feature_mask = np.ones(X_with_noise.shape[1], dtype=bool)
    feature_mask[list(features_to_remove)] = False
    X_corr_selected = X_with_noise[:, feature_mask]
    print(f"Features after correlation selection: {X_corr_selected.shape[1]}")
    
    # 3. Recursive Feature Elimination
    print("\n3. Recursive Feature Elimination:")
    rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=10)
    X_rfe_selected = rfe.fit_transform(X_with_noise, y)
    print(f"Features after RFE: {X_rfe_selected.shape[1]}")
    
    # 4. Lasso feature selection
    print("\n4. Lasso Feature Selection:")
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_with_noise, y)
    selected_features = lasso.coef_ != 0
    X_lasso_selected = X_with_noise[:, selected_features]
    print(f"Features after Lasso: {X_lasso_selected.shape[1]}")
    
    # Visualize feature selection results
    methods = ['Original', 'Variance', 'Correlation', 'RFE', 'Lasso']
    feature_counts = [X_with_noise.shape[1], X_variance_selected.shape[1], 
                     X_corr_selected.shape[1], X_rfe_selected.shape[1], 
                     X_lasso_selected.shape[1]]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, feature_counts, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Feature Selection Results')
    plt.ylabel('Number of Features')
    plt.ylim(0, max(feature_counts) + 5)
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.show()
    
    return X_variance_selected, X_corr_selected, X_rfe_selected, X_lasso_selected

def demonstrate_tsne():
    """
    Demonstrate t-SNE for visualization
    """
    print("\n=== t-SNE Demo ===")
    
    # Generate data
    X, y = make_blobs(n_samples=1000, n_features=10, centers=4, random_state=42)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # Original data (first 2 features)
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
    plt.title('Original Data (First 2 Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # t-SNE visualization
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.6)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    plt.show()
    
    return X_tsne

def demonstrate_3d_visualization(X, y):
    """
    Demonstrate 3D visualization of data
    """
    print("\n=== 3D Visualization Demo ===")
    
    # Reduce to 3D using PCA
    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit_transform(X)
    
    # 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                        c=y, alpha=0.6)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Visualization')
    
    plt.show()
    
    return X_3d

def demonstrate_image_compression():
    """
    Demonstrate image compression using PCA
    """
    print("\n=== Image Compression Demo ===")
    
    try:
        # Load face dataset
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X_faces = faces.data
        y_faces = faces.target
        
        # Apply PCA with different numbers of components
        n_components_list = [10, 50, 100, 200]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, n_components in enumerate(n_components_list):
            # Apply PCA
            pca_faces = PCA(n_components=n_components)
            X_faces_reduced = pca_faces.fit_transform(X_faces)
            X_faces_reconstructed = pca_faces.inverse_transform(X_faces_reduced)
            
            # Show original and reconstructed
            axes[0, i].imshow(X_faces[0].reshape(64, 64), cmap='gray')
            axes[0, i].set_title(f'Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(X_faces_reconstructed[0].reshape(64, 64), cmap='gray')
            axes[1, i].set_title(f'Reconstructed ({n_components} components)')
            axes[1, i].axis('off')
            
            # Calculate compression ratio
            original_size = X_faces.shape[1]
            compressed_size = n_components
            compression_ratio = (1 - compressed_size / original_size) * 100
            print(f"{n_components} components: {compression_ratio:.1f}% compression")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not load Olivetti faces dataset: {e}")
        print("This demo requires the Olivetti faces dataset to be available.")

def demonstrate_text_analysis():
    """
    Demonstrate text analysis with dimensionality reduction
    """
    print("\n=== Text Analysis Demo ===")
    
    try:
        # Load text data
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        newsgroups = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
        
        # Convert to TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_text = vectorizer.fit_transform(newsgroups.data)
        
        # Apply PCA
        pca_text = PCA(n_components=2)
        X_text_reduced = pca_text.fit_transform(X_text.toarray())
        
        # Visualize
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_text_reduced[:, 0], X_text_reduced[:, 1], 
                             c=newsgroups.target, alpha=0.6)
        plt.title('Document Clustering with PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                            title="Categories")
        plt.gca().add_artist(legend1)
        
        plt.show()
        
        return X_text_reduced, newsgroups.target
        
    except Exception as e:
        print(f"Could not load 20 newsgroups dataset: {e}")
        print("This demo requires the 20 newsgroups dataset to be available.")
        return None, None

def demonstrate_reconstruction_error():
    """
    Demonstrate reconstruction error vs number of components
    """
    print("\n=== Reconstruction Error Demo ===")
    
    # Generate data
    X, y = make_blobs(n_samples=1000, n_features=20, centers=3, random_state=42)
    
    # Test different numbers of components
    n_components_list = range(1, 21)
    reconstruction_errors = []
    
    for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        
        # Calculate reconstruction error
        error = mean_squared_error(X, X_reconstructed)
        reconstruction_errors.append(error)
    
    # Plot reconstruction error
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, reconstruction_errors, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error vs Number of Components')
    plt.grid(True)
    plt.show()
    
    # Find optimal number of components (elbow point)
    # Simple method: find point where slope changes significantly
    slopes = np.diff(reconstruction_errors)
    slope_changes = np.abs(np.diff(slopes))
    optimal_components = n_components_list[np.argmax(slope_changes) + 1]
    
    print(f"Optimal number of components (elbow method): {optimal_components}")
    
    return n_components_list, reconstruction_errors, optimal_components

def demonstrate_feature_importance():
    """
    Demonstrate feature importance in PCA
    """
    print("\n=== Feature Importance Demo ===")
    
    # Generate data with known feature importance
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    # Create features with different variances
    X = np.random.randn(n_samples, n_features)
    X[:, 0] *= 5  # High variance feature
    X[:, 1] *= 3  # Medium variance feature
    X[:, 2:] *= 1  # Low variance features
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    
    # Get feature importance
    feature_importance = pca.get_feature_importance()
    
    # Visualize feature importance
    plt.figure(figsize=(12, 4))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.bar(range(n_features), feature_importance[i])
        plt.title(f'Feature Importance - PC{i+1}')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.xticks(range(n_features))
    
    plt.tight_layout()
    plt.show()
    
    # Print feature importance summary
    print("Feature importance in first 3 principal components:")
    for i in range(3):
        top_features = np.argsort(feature_importance[i])[-3:][::-1]
        print(f"PC{i+1} top features: {top_features}")

def main():
    """
    Run all dimensionality reduction demonstrations
    """
    print("Dimensionality Reduction Complete Implementation and Examples")
    print("=" * 60)
    
    # Generate data for demonstrations
    X, y = make_blobs(n_samples=1000, n_features=20, centers=4, random_state=42)
    
    # Run all demonstrations
    pca_model, X_reduced = demonstrate_pca()
    n_95, n_99 = demonstrate_elbow_method(X)
    demonstrate_scree_plot(X)
    demonstrate_kernel_pca()
    demonstrate_sparse_pca(X)
    X_var, X_corr, X_rfe, X_lasso = demonstrate_feature_selection()
    X_tsne = demonstrate_tsne()
    X_3d = demonstrate_3d_visualization(X, y)
    demonstrate_image_compression()
    X_text, text_labels = demonstrate_text_analysis()
    n_comp_list, recon_errors, optimal_comp = demonstrate_reconstruction_error()
    demonstrate_feature_importance()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")
    print(f"Optimal components (reconstruction error): {optimal_comp}")
    print(f"PCA explained variance ratio: {pca_model.explained_variance_ratio_[:3]}")

if __name__ == "__main__":
    main() 