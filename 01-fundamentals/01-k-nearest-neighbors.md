# K-Nearest Neighbors (K-NN)

## Overview

K-Nearest Neighbors (K-NN) is one of the simplest yet most effective machine learning algorithms. It's a non-parametric method that makes predictions based on the similarity of data points. The core idea is simple: "similar things are near each other."

## Key Concepts

### 1. The K-NN Principle

K-NN operates on the principle that objects that are close to each other in feature space are likely to belong to the same class or have similar values. The algorithm:

1. **Stores** all training data points
2. **Calculates** distances between a new point and all training points
3. **Finds** the K closest training points (neighbors)
4. **Predicts** based on the majority class (classification) or average value (regression)

### 2. Distance Metrics

The choice of distance metric significantly impacts K-NN performance:

#### Euclidean Distance
Most commonly used distance metric:
```math
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
```

**Python Implementation:**
```python
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

#### Manhattan Distance
Also called L1 distance or city block distance:
```math
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
```

**Python Implementation:**
```python
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))
```

#### Cosine Similarity
Useful for high-dimensional data, especially text:
```math
\text{cosine\_similarity}(x, y) = \frac{x \cdot y}{\|x\| \times \|y\|}
```

**Python Implementation:**
```python
def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return dot_product / (norm_x1 * norm_x2)
```

## K-NN Classification

### Algorithm Steps

1. **Input**: Training data (X_train, y_train), test point x, parameter K
2. **Distance Calculation**: Compute distance between x and all training points
3. **Neighbor Selection**: Find K training points with smallest distances
4. **Voting**: Predict class by majority vote among K neighbors
5. **Output**: Predicted class label

### Complete Implementation

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """Store training data (KNN doesn't actually train)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        """Calculate distance between two points"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm_x1 * norm_x2))
    
    def predict(self, X):
        """Predict class labels"""
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._calculate_distance(x, x_train)
                distances.append((dist, i))
            
            # Get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            neighbors = [self.y_train[i] for _, i in distances[:self.k]]
            
            # Majority vote
            most_common = Counter(neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
```

### Example: Iris Flower Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train KNN
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

## K-NN Regression

### Algorithm Steps

1. **Input**: Training data (X_train, y_train), test point x, parameter K
2. **Distance Calculation**: Compute distance between x and all training points
3. **Neighbor Selection**: Find K training points with smallest distances
4. **Averaging**: Predict value as average of K neighbors' target values
5. **Output**: Predicted continuous value

### Implementation

```python
class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        # Same as classifier
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._calculate_distance(x, x_train)
                distances.append((dist, i))
            
            # Get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            neighbors = [self.y_train[i] for _, i in distances[:self.k]]
            
            # Average the values
            prediction = np.mean(neighbors)
            predictions.append(prediction)
        
        return np.array(predictions)
```

## Data Representation and Feature Engineering

### Feature Vectors

Data points are represented as feature vectors in n-dimensional space:

```python
# Example: House price prediction
house_features = {
    'square_feet': 1500,
    'bedrooms': 3,
    'bathrooms': 2,
    'age': 10,
    'distance_to_city': 5.2
}

# Convert to feature vector
feature_vector = [1500, 3, 2, 10, 5.2]
```

### Feature Scaling

Different features may have different scales, affecting distance calculations:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling (z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-max scaling (to [0,1] range)
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
```

### Feature Engineering Techniques

1. **Polynomial Features**: Create interaction terms
2. **Binning**: Convert continuous features to categorical
3. **One-Hot Encoding**: Convert categorical to numerical
4. **Feature Selection**: Remove irrelevant features

## Generalization and Model Selection

### Cross-Validation

K-fold cross-validation helps assess model performance:

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(knn, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Hyperparameter Tuning

Finding optimal K value:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'k': [1, 3, 5, 7, 9, 11, 13, 15]}

# Grid search with cross-validation
grid_search = GridSearchCV(
    KNNClassifier(), param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best K: {grid_search.best_params_['k']}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")
```

### Overfitting and Underfitting

- **Overfitting**: K too small, model memorizes training data
- **Underfitting**: K too large, model oversimplifies
- **Sweet spot**: Balance between bias and variance

### Bias-Variance Trade-off

```python
import matplotlib.pyplot as plt

# Test different K values
k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, knn.predict(X_train))
    test_score = accuracy_score(y_test, knn.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Plot bias-variance trade-off
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Training Accuracy', marker='o')
plt.plot(k_values, test_scores, label='Test Accuracy', marker='s')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Bias-Variance Trade-off in K-NN')
plt.legend()
plt.grid(True)
plt.show()
```

## Practical Considerations

### Computational Complexity

- **Training**: $`O(1)`$ - just store data
- **Prediction**: $`O(n \times d)`$ where $`n`$ = training samples, $`d`$ = dimensions
- **Memory**: $`O(n \times d)`$ - stores all training data

### Optimization Techniques

1. **KD-Trees**: For efficient nearest neighbor search
2. **Ball Trees**: For high-dimensional data
3. **Locality Sensitive Hashing**: For approximate nearest neighbors

### When to Use K-NN

**Advantages:**
- Simple to understand and implement
- No training required
- Naturally handles multi-class problems
- Works well with small datasets

**Disadvantages:**
- Computationally expensive for large datasets
- Sensitive to irrelevant features
- Requires feature scaling
- Memory intensive

## Advanced Topics

### Weighted K-NN

Give more weight to closer neighbors:

```python
def weighted_knn_predict(self, x):
    distances = []
    for i, x_train in enumerate(self.X_train):
        dist = self._calculate_distance(x, x_train)
        distances.append((dist, i))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:self.k]
    
    # Weighted voting (inverse distance)
    weights = [1 / (dist + 1e-8) for dist, _ in neighbors]
    neighbor_labels = [self.y_train[i] for _, i in neighbors]
    
    # Weighted majority vote
    weighted_votes = {}
    for label, weight in zip(neighbor_labels, weights):
        weighted_votes[label] = weighted_votes.get(label, 0) + weight
    
    return max(weighted_votes, key=weighted_votes.get)
```

### Adaptive K-NN

Choose K based on local density:

```python
def adaptive_knn_predict(self, x, min_k=1, max_k=20):
    distances = []
    for i, x_train in enumerate(self.X_train):
        dist = self._calculate_distance(x, x_train)
        distances.append((dist, i))
    
    distances.sort(key=lambda x: x[0])
    
    # Find optimal K for this point
    best_k = min_k
    best_score = 0
    
    for k in range(min_k, min(max_k + 1, len(distances))):
        neighbors = [self.y_train[i] for _, i in distances[:k]]
        # Use leave-one-out cross-validation on neighbors
        # (simplified version)
        score = self._evaluate_local_k(k, neighbors)
        if score > best_score:
            best_score = score
            best_k = k
    
    # Use best K for prediction
    neighbors = [self.y_train[i] for _, i in distances[:best_k]]
    return Counter(neighbors).most_common(1)[0][0]
```

## Summary

K-Nearest Neighbors is a fundamental algorithm that demonstrates key machine learning concepts:

1. **Instance-based learning**: Learning by storing examples
2. **Distance-based similarity**: Using geometric relationships
3. **Non-parametric methods**: No assumptions about data distribution
4. **Lazy learning**: No explicit training phase

Understanding K-NN provides a solid foundation for more advanced algorithms and helps develop intuition about how machine learning works in practice. 