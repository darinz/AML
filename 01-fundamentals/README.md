# Fundamentals & Basic Algorithms

This module covers the foundational concepts and basic algorithms that form the building blocks of machine learning.

## Learning Objectives

By the end of this module, you will be able to:
- Implement K-Nearest Neighbors for both classification and regression
- Understand data representation and feature engineering
- Apply clustering algorithms to group similar data points
- Reduce dimensionality using PCA and other embedding techniques
- Evaluate model performance and understand generalization

## Topics Covered

### 1. K-Nearest Neighbors (K-NN)
- **K-NN Classification**: Implementing nearest neighbor classification from scratch
- **Data Representation**: Understanding feature vectors, distance metrics, and data preprocessing
- **K-NN Regression**: Extending K-NN to continuous target variables
- **Generalization**: Cross-validation, overfitting, and model selection

### 2. Search and Clustering
- **Clustering Algorithms**: K-means, hierarchical clustering, DBSCAN
- **Search Techniques**: Efficient nearest neighbor search, KD-trees
- **Similarity Metrics**: Euclidean distance, cosine similarity, Manhattan distance
- **Evaluation Metrics**: Silhouette score, inertia, cluster validation

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Linear dimensionality reduction
- **Embeddings**: Word embeddings, feature embeddings, t-SNE
- **Feature Selection**: Filter methods, wrapper methods, embedded methods
- **Visualization**: 2D/3D projections of high-dimensional data

## Comprehensive Guides

This module includes detailed markdown guides for each topic:

- **[01-k-nearest-neighbors.md](01-k-nearest-neighbors.md)** - Complete guide to K-NN algorithms
- **[02-search-and-clustering.md](02-search-and-clustering.md)** - Comprehensive clustering techniques
- **[03-dimensionality-reduction.md](03-dimensionality-reduction.md)** - Dimensionality reduction methods

## Python Examples

Hands-on implementations and examples for each topic:

- **[knn_examples.py](knn_examples.py)** - Complete K-NN implementation with examples
- **[clustering_examples.py](clustering_examples.py)** - Clustering algorithms and evaluation
- **[dimensionality_reduction_examples.py](dimensionality_reduction_examples.py)** - PCA, embeddings, and feature selection

## Getting Started

### Prerequisites

- Basic Python programming
- Understanding of arrays and matrices
- Familiarity with basic statistics (mean, variance, correlation)

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the examples:
```bash
# K-Nearest Neighbors examples
python knn_examples.py

# Clustering examples
python clustering_examples.py

# Dimensionality reduction examples
python dimensionality_reduction_examples.py
```

## Practical Applications

- **Recommendation Systems**: Finding similar users or items
- **Image Recognition**: Basic image classification using pixel features
- **Document Clustering**: Grouping similar documents or articles
- **Data Visualization**: Exploring high-dimensional datasets
- **Customer Segmentation**: Grouping customers by behavior patterns
- **Feature Engineering**: Creating meaningful data representations

## Implementation Focus

This module emphasizes **hands-on implementation**:
- Code K-NN from scratch in Python
- Implement clustering algorithms without using sklearn
- Build PCA step-by-step using numpy
- Create interactive visualizations
- Apply evaluation metrics and model selection

## Key Concepts Covered

### Distance Metrics
- Euclidean distance
- Manhattan distance
- Cosine similarity
- Jaccard similarity

### Clustering Evaluation
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz index
- Inertia (within-cluster sum of squares)

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Kernel PCA
- t-SNE visualization
- Feature selection methods

### Model Selection
- Cross-validation
- Hyperparameter tuning
- Bias-variance trade-off
- Overfitting and underfitting

## Advanced Topics

### K-NN Extensions
- Weighted K-NN
- Adaptive K-NN
- Efficient search with KD-trees
- Ball trees for high-dimensional data

### Clustering Extensions
- Hierarchical clustering with different linkage methods
- Density-based clustering (DBSCAN)
- Model-based clustering
- Spectral clustering

### Dimensionality Reduction Extensions
- Sparse PCA
- Non-negative matrix factorization
- Autoencoders
- Manifold learning

## Evaluation and Assessment

Each topic includes:
- **Theoretical understanding**: Mathematical foundations and concepts
- **Practical implementation**: Complete code examples
- **Visualization**: Interactive plots and demonstrations
- **Real-world applications**: Practical use cases and examples

## Next Steps

After completing this module, you'll be ready for **Linear Models & Classical ML** where you'll learn about more sophisticated supervised learning algorithms.

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

## Contributing

Feel free to contribute improvements, additional examples, or corrections to the guides and code examples. 