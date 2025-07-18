# Dimensionality Reduction

## Overview

Dimensionality reduction is a crucial technique in machine learning that transforms high-dimensional data into a lower-dimensional representation while preserving important information. It addresses the "curse of dimensionality" and helps with visualization, computational efficiency, and model performance.

## Key Concepts

### 1. The Curse of Dimensionality

As the number of dimensions increases:
- **Data sparsity**: Points become increasingly isolated
- **Distance metrics become less meaningful**
- **Computational complexity grows exponentially**
- **Overfitting risk increases**

### 2. Types of Dimensionality Reduction

1. **Linear Methods**: PCA, LDA, Factor Analysis
2. **Non-linear Methods**: t-SNE, UMAP, Isomap
3. **Feature Selection**: Filter, Wrapper, Embedded methods
4. **Embeddings**: Word embeddings, learned representations

## Principal Component Analysis (PCA)

### Mathematical Foundation

PCA finds the directions (principal components) that maximize the variance in the data.

#### Step 1: Data Centering
```math
X_{\text{centered}} = X - \mu
```

#### Step 2: Covariance Matrix
```math
\Sigma = \frac{1}{n} \times X_{\text{centered}}^T \times X_{\text{centered}}
```

#### Step 3: Eigendecomposition
```math
\Sigma = V \times \Lambda \times V^T
```

Where:
- $`V`$ contains eigenvectors (principal components)
- $`\Lambda`$ contains eigenvalues (variance explained)

#### Step 4: Projection
```math
X_{\text{reduced}} = X_{\text{centered}} \times V[:, :k]
```

### Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class PCA:
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

# Example usage
def demonstrate_pca():
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

# Run demonstration
pca_model, X_reduced = demonstrate_pca()
```

### Choosing the Number of Components

#### Cumulative Variance Method

```python
def plot_explained_variance(X, max_components=None):
    """Plot explained variance to choose number of components"""
    if max_components is None:
        max_components = X.shape[1]
    
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

# Apply to our data
n_95, n_99 = plot_explained_variance(X)
```

#### Scree Plot Method

```python
def plot_scree(X):
    """Plot scree plot for eigenvalue analysis"""
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

plot_scree(X)
```

## Advanced PCA Techniques

### Kernel PCA

For non-linear dimensionality reduction:

```python
from sklearn.decomposition import KernelPCA

def demonstrate_kernel_pca():
    # Generate non-linear data (swiss roll)
    from sklearn.datasets import make_swiss_roll
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

demonstrate_kernel_pca()
```

### Sparse PCA

For interpretable components with many zeros:

```python
from sklearn.decomposition import SparsePCA

def demonstrate_sparse_pca(X):
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

demonstrate_sparse_pca(X)
```

## Embeddings

### Word Embeddings

#### Word2Vec Implementation

```python
import gensim
from gensim.models import Word2Vec

class SimpleWord2Vec:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        
    def fit(self, sentences):
        """Train Word2Vec model"""
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1
        )
        return self
    
    def get_vector(self, word):
        """Get vector for a word"""
        return self.model.wv[word]
    
    def most_similar(self, word, topn=5):
        """Find most similar words"""
        return self.model.wv.most_similar(word, topn=topn)

# Example usage
sentences = [
    ['machine', 'learning', 'algorithms'],
    ['deep', 'learning', 'neural', 'networks'],
    ['data', 'science', 'analytics'],
    ['artificial', 'intelligence', 'systems'],
    # ... more sentences
]

word2vec = SimpleWord2Vec(vector_size=50)
word2vec.fit(sentences)

# Get word vectors
learning_vector = word2vec.get_vector('learning')
print(f"Vector for 'learning': {learning_vector[:5]}...")  # Show first 5 dimensions

# Find similar words
similar_words = word2vec.most_similar('learning')
print(f"Words similar to 'learning': {similar_words}")
```

#### GloVe Implementation

```python
import numpy as np
from collections import defaultdict

class SimpleGloVe:
    def __init__(self, vector_size=100, window_size=5):
        self.vector_size = vector_size
        self.window_size = window_size
        self.word_vectors = {}
        self.cooccurrence_matrix = defaultdict(float)
        
    def build_cooccurrence_matrix(self, sentences):
        """Build co-occurrence matrix"""
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(max(0, i - self.window_size), 
                             min(len(sentence), i + self.window_size + 1)):
                    if i != j:
                        self.cooccurrence_matrix[(word, sentence[j])] += 1.0 / abs(i - j)
    
    def fit(self, sentences):
        """Train GloVe model (simplified version)"""
        # Build co-occurrence matrix
        self.build_cooccurrence_matrix(sentences)
        
        # Get unique words
        words = list(set([word for pair in self.cooccurrence_matrix.keys() 
                         for word in pair]))
        
        # Initialize random vectors
        for word in words:
            self.word_vectors[word] = np.random.randn(self.vector_size)
        
        # Simple gradient descent (simplified)
        learning_rate = 0.01
        epochs = 100
        
        for epoch in range(epochs):
            for (word1, word2), count in self.cooccurrence_matrix.items():
                if word1 in self.word_vectors and word2 in self.word_vectors:
                    # Simplified update rule
                    diff = self.word_vectors[word1] - self.word_vectors[word2]
                    self.word_vectors[word1] -= learning_rate * diff
                    self.word_vectors[word2] += learning_rate * diff
        
        return self
    
    def get_vector(self, word):
        """Get vector for a word"""
        return self.word_vectors.get(word, np.zeros(self.vector_size))

# Example usage
glove = SimpleGloVe(vector_size=50)
glove.fit(sentences)

# Get word vectors
learning_vector_glove = glove.get_vector('learning')
print(f"GloVe vector for 'learning': {learning_vector_glove[:5]}...")
```

### Feature Embeddings

```python
class FeatureEmbedding:
    def __init__(self, embedding_dim=32):
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        
    def fit(self, categorical_features):
        """Learn embeddings for categorical features"""
        unique_values = set()
        for feature_list in categorical_features:
            unique_values.update(feature_list)
        
        # Initialize random embeddings
        for value in unique_values:
            self.embeddings[value] = np.random.randn(self.embedding_dim)
        
        return self
    
    def transform(self, categorical_features):
        """Transform categorical features to embeddings"""
        embedded_features = []
        
        for feature_list in categorical_features:
            feature_embedding = np.zeros(self.embedding_dim)
            for feature in feature_list:
                if feature in self.embeddings:
                    feature_embedding += self.embeddings[feature]
            embedded_features.append(feature_embedding)
        
        return np.array(embedded_features)

# Example usage
categorical_data = [
    ['red', 'large', 'round'],
    ['blue', 'small', 'square'],
    ['green', 'medium', 'triangle'],
    # ... more examples
]

embedding_model = FeatureEmbedding(embedding_dim=16)
embedding_model.fit(categorical_data)

# Transform to embeddings
embedded_data = embedding_model.transform(categorical_data)
print(f"Embedded data shape: {embedded_data.shape}")
```

## Feature Selection

### Filter Methods

#### Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selection(X, threshold=0.01):
    """Remove low-variance features"""
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    # Get selected feature indices
    selected_features = selector.get_support()
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Removed features: {X.shape[1] - X_selected.shape[1]}")
    
    return X_selected, selected_features

# Apply variance threshold
X_variance_selected, variance_mask = variance_threshold_selection(X)
```

#### Correlation-based Selection

```python
def correlation_based_selection(X, threshold=0.95):
    """Remove highly correlated features"""
    correlation_matrix = np.corrcoef(X.T)
    
    # Find highly correlated pairs
    high_corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    high_corr_pairs = [(high_corr_pairs[0][i], high_corr_pairs[1][i]) 
                       for i in range(len(high_corr_pairs[0])) 
                       if high_corr_pairs[0][i] < high_corr_pairs[1][i]]
    
    # Remove one feature from each pair
    features_to_remove = set()
    for pair in high_corr_pairs:
        features_to_remove.add(pair[1])  # Remove the second feature
    
    # Create mask for selected features
    feature_mask = np.ones(X.shape[1], dtype=bool)
    feature_mask[list(features_to_remove)] = False
    
    X_corr_selected = X[:, feature_mask]
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_corr_selected.shape[1]}")
    print(f"Removed features: {len(features_to_remove)}")
    
    return X_corr_selected, feature_mask

# Apply correlation-based selection
X_corr_selected, corr_mask = correlation_based_selection(X)
```

### Wrapper Methods

#### Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def recursive_feature_elimination(X, y, n_features=5):
    """Recursive feature elimination"""
    estimator = LogisticRegression(random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(X, y)
    
    # Get selected features
    selected_features = rfe.get_support()
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_rfe.shape[1]}")
    print(f"Feature rankings: {rfe.ranking_}")
    
    return X_rfe, selected_features

# Apply RFE (assuming we have labels)
# X_rfe_selected, rfe_mask = recursive_feature_elimination(X, y)
```

### Embedded Methods

#### Lasso Regression

```python
from sklearn.linear_model import Lasso

def lasso_feature_selection(X, y, alpha=0.01):
    """Feature selection using Lasso"""
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X, y)
    
    # Get non-zero coefficients
    selected_features = lasso.coef_ != 0
    X_lasso_selected = X[:, selected_features]
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_lasso_selected.shape[1]}")
    print(f"Lasso coefficients: {lasso.coef_}")
    
    return X_lasso_selected, selected_features

# Apply Lasso feature selection
# X_lasso_selected, lasso_mask = lasso_feature_selection(X, y)
```

## Visualization Techniques

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

```python
from sklearn.manifold import TSNE

def demonstrate_tsne(X, y):
    """Demonstrate t-SNE for visualization"""
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

# Apply t-SNE
X_tsne = demonstrate_tsne(X, y)
```

### UMAP (Uniform Manifold Approximation and Projection)

```python
try:
    import umap
    
    def demonstrate_umap(X, y):
        """Demonstrate UMAP for visualization"""
        # Apply UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        
        # Visualize
        plt.figure(figsize=(12, 5))
        
        # Original data (first 2 features)
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
        plt.title('Original Data (First 2 Features)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # UMAP visualization
        plt.subplot(1, 2, 2)
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, alpha=0.6)
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        
        plt.tight_layout()
        plt.show()
        
        return X_umap
    
    # Apply UMAP
    # X_umap = demonstrate_umap(X, y)
    
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
```

### 3D Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d(X, y):
    """3D visualization of data"""
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

# Apply 3D visualization
X_3d = visualize_3d(X, y)
```

## Practical Applications

### Image Compression

```python
def image_compression_demo():
    """Demonstrate image compression using PCA"""
    from sklearn.datasets import fetch_olivetti_faces
    
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

# Run image compression demo
# image_compression_demo()
```

### Text Document Analysis

```python
def text_analysis_demo():
    """Demonstrate text analysis with dimensionality reduction"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    
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

# Run text analysis demo
# X_text_reduced, text_labels = text_analysis_demo()
```

## Summary

Dimensionality reduction is essential for:

1. **Visualization**: Making high-dimensional data interpretable
2. **Computational Efficiency**: Reducing processing time and memory usage
3. **Feature Engineering**: Creating meaningful representations
4. **Noise Reduction**: Removing irrelevant information
5. **Model Performance**: Improving accuracy and preventing overfitting

Key considerations:

- **Linear vs Non-linear**: Choose based on data structure
- **Information Loss**: Balance compression with information preservation
- **Interpretability**: Some methods provide interpretable components
- **Scalability**: Consider computational requirements for large datasets

Understanding these techniques provides a solid foundation for effective data preprocessing and feature engineering in machine learning pipelines. 