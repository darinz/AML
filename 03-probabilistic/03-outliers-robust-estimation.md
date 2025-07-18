# Outliers & Robust Estimation

## Overview

Outlier detection and robust estimation are essential for handling data that contains unusual observations or violations of model assumptions. This guide covers various techniques for identifying outliers and performing robust statistical estimation.

## Key Concepts

### 1. What are Outliers?

Outliers are data points that deviate significantly from the expected pattern. They can be:

- **Point Outliers**: Individual observations that are unusual
- **Contextual Outliers**: Observations that are unusual in a specific context
- **Collective Outliers**: Groups of observations that are unusual together

### 2. Types of Outlier Detection

1. **Statistical Methods**: Based on distribution assumptions
2. **Distance-based Methods**: Based on proximity to other points
3. **Density-based Methods**: Based on local density
4. **Isolation-based Methods**: Based on ease of isolation

## Statistical Outlier Detection

### Z-Score Method

The Z-score measures how many standard deviations a data point is from the mean:

```math
Z_i = \frac{x_i - \mu}{\sigma}
```

Where $`\mu`$ and $`\sigma`$ are the mean and standard deviation of the data.

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

class ZScoreOutlierDetector:
    """Z-score based outlier detection"""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """Fit the outlier detector"""
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def predict(self, X):
        """Predict outliers"""
        X = np.array(X)
        z_scores = np.abs((X - self.mean) / self.std)
        return np.any(z_scores > self.threshold, axis=1)
    
    def z_scores(self, X):
        """Compute Z-scores"""
        X = np.array(X)
        return (X - self.mean) / self.std

def demonstrate_zscore():
    """Demonstrate Z-score outlier detection"""
    print("=== Z-Score Outlier Detection Demo ===")
    
    # Generate data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.normal(0, 1, n_samples)
    
    # Add outliers
    outliers = np.random.normal(5, 1, 50)
    X = np.concatenate([normal_data, outliers])
    
    # Fit Z-score detector
    zscore_detector = ZScoreOutlierDetector(threshold=3.0)
    zscore_detector.fit(X.reshape(-1, 1))
    
    # Predict outliers
    is_outlier = zscore_detector.predict(X.reshape(-1, 1))
    z_scores = zscore_detector.z_scores(X.reshape(-1, 1)).flatten()
    
    print(f"Detected {np.sum(is_outlier)} outliers out of {len(X)} points")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.hist(X, bins=50, alpha=0.7, label='All Data')
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Z-scores
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(z_scores)), z_scores, alpha=0.6, c=is_outlier, cmap='viridis')
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Z-Scores')
    plt.xlabel('Data Point')
    plt.ylabel('Z-Score')
    plt.legend()
    
    # Outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(range(len(X))[normal_mask], X[normal_mask], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(range(len(X))[outlier_mask], X[outlier_mask], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('Outlier Detection')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return zscore_detector, X, is_outlier

# Run demonstration
zscore_detector, X_data, is_outlier = demonstrate_zscore()
```

### IQR Method

The Interquartile Range (IQR) method identifies outliers as points outside the range:

```math
\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}
```

```math
\text{Upper Bound} = Q3 + 1.5 \times \text{IQR}
```

Where IQR = Q3 - Q1.

### Implementation

```python
class IQROutlierDetector:
    """IQR-based outlier detection"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        self.q1 = None
        self.q3 = None
        self.iqr = None
        
    def fit(self, X):
        """Fit the outlier detector"""
        X = np.array(X)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        return self
    
    def predict(self, X):
        """Predict outliers"""
        X = np.array(X)
        lower_bound = self.q1 - self.factor * self.iqr
        upper_bound = self.q3 + self.factor * self.iqr
        
        return np.any((X < lower_bound) | (X > upper_bound), axis=1)
    
    def bounds(self):
        """Get outlier bounds"""
        lower_bound = self.q1 - self.factor * self.iqr
        upper_bound = self.q3 + self.factor * self.iqr
        return lower_bound, upper_bound

def demonstrate_iqr():
    """Demonstrate IQR outlier detection"""
    print("\n=== IQR Outlier Detection Demo ===")
    
    # Generate data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.normal(0, 1, n_samples)
    
    # Add outliers
    outliers = np.random.normal(5, 1, 50)
    X = np.concatenate([normal_data, outliers])
    
    # Fit IQR detector
    iqr_detector = IQROutlierDetector(factor=1.5)
    iqr_detector.fit(X.reshape(-1, 1))
    
    # Predict outliers
    is_outlier = iqr_detector.predict(X.reshape(-1, 1))
    lower_bound, upper_bound = iqr_detector.bounds()
    
    print(f"Detected {np.sum(is_outlier)} outliers out of {len(X)} points")
    print(f"Lower bound: {lower_bound[0]:.3f}")
    print(f"Upper bound: {upper_bound[0]:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Box plot
    plt.subplot(1, 3, 1)
    plt.boxplot(X)
    plt.title('Box Plot')
    plt.ylabel('Value')
    
    # Data with bounds
    plt.subplot(1, 3, 2)
    plt.hist(X, bins=50, alpha=0.7, label='All Data')
    plt.axvline(x=lower_bound[0], color='r', linestyle='--', label='Lower Bound')
    plt.axvline(x=upper_bound[0], color='r', linestyle='--', label='Upper Bound')
    plt.title('Data with IQR Bounds')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(range(len(X))[normal_mask], X[normal_mask], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(range(len(X))[outlier_mask], X[outlier_mask], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.axhline(y=lower_bound[0], color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=upper_bound[0], color='r', linestyle='--', alpha=0.5)
    plt.title('Outlier Detection')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return iqr_detector, X, is_outlier

# Run demonstration
iqr_detector, X_data, is_outlier = demonstrate_iqr()
```

## Robust Statistics

### Median and MAD

Robust alternatives to mean and standard deviation:

**Median Absolute Deviation (MAD)**:
```math
\text{MAD} = \text{median}(|x_i - \text{median}(X)|)
```

**Robust Z-score**:
```math
Z_i = \frac{x_i - \text{median}(X)}{1.4826 \times \text{MAD}}
```

### Implementation

```python
class RobustStatistics:
    """Robust statistical estimators"""
    
    def __init__(self):
        self.median = None
        self.mad = None
        self.robust_mean = None
        self.robust_std = None
        
    def fit(self, X):
        """Compute robust statistics"""
        X = np.array(X)
        
        # Median
        self.median = np.median(X, axis=0)
        
        # MAD
        deviations = np.abs(X - self.median)
        self.mad = np.median(deviations, axis=0)
        
        # Robust mean (trimmed mean)
        self.robust_mean = stats.trim_mean(X, proportiontocut=0.1, axis=0)
        
        # Robust standard deviation
        self.robust_std = 1.4826 * self.mad
        
        return self
    
    def robust_z_scores(self, X):
        """Compute robust Z-scores"""
        X = np.array(X)
        return (X - self.median) / (1.4826 * self.mad)
    
    def detect_outliers(self, X, threshold=3.0):
        """Detect outliers using robust statistics"""
        robust_z_scores = self.robust_z_scores(X)
        return np.any(np.abs(robust_z_scores) > threshold, axis=1)

def demonstrate_robust_statistics():
    """Demonstrate robust statistics"""
    print("\n=== Robust Statistics Demo ===")
    
    # Generate data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.normal(0, 1, n_samples)
    
    # Add outliers
    outliers = np.random.normal(5, 1, 50)
    X = np.concatenate([normal_data, outliers])
    
    # Compute robust statistics
    robust_stats = RobustStatistics()
    robust_stats.fit(X.reshape(-1, 1))
    
    # Compare with classical statistics
    classical_mean = np.mean(X)
    classical_std = np.std(X)
    
    print("Classical Statistics:")
    print(f"  Mean: {classical_mean:.3f}")
    print(f"  Std: {classical_std:.3f}")
    
    print("\nRobust Statistics:")
    print(f"  Median: {robust_stats.median[0]:.3f}")
    print(f"  MAD: {robust_stats.mad[0]:.3f}")
    print(f"  Robust Mean: {robust_stats.robust_mean[0]:.3f}")
    print(f"  Robust Std: {robust_stats.robust_std[0]:.3f}")
    
    # Detect outliers
    is_outlier = robust_stats.detect_outliers(X.reshape(-1, 1), threshold=3.0)
    robust_z_scores = robust_stats.robust_z_scores(X.reshape(-1, 1)).flatten()
    
    print(f"\nDetected {np.sum(is_outlier)} outliers using robust statistics")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Classical vs Robust Z-scores
    plt.subplot(1, 3, 1)
    classical_z_scores = (X - classical_mean) / classical_std
    plt.scatter(classical_z_scores, robust_z_scores, alpha=0.6)
    plt.plot([-5, 5], [-5, 5], 'r--', alpha=0.5)
    plt.xlabel('Classical Z-Score')
    plt.ylabel('Robust Z-Score')
    plt.title('Classical vs Robust Z-Scores')
    plt.grid(True)
    
    # Robust Z-scores
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(robust_z_scores)), robust_z_scores, alpha=0.6, c=is_outlier, cmap='viridis')
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Robust Z-Scores')
    plt.xlabel('Data Point')
    plt.ylabel('Robust Z-Score')
    plt.legend()
    
    # Outliers comparison
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(range(len(X))[normal_mask], X[normal_mask], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(range(len(X))[outlier_mask], X[outlier_mask], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.axhline(y=robust_stats.median[0], color='g', linestyle='-', label='Median')
    plt.title('Robust Outlier Detection')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return robust_stats, X, is_outlier

# Run demonstration
robust_stats, X_data, is_outlier = demonstrate_robust_statistics()
```

## Isolation Forest

### Mathematical Foundation

Isolation Forest is based on the principle that outliers are easier to isolate than normal points. It builds isolation trees by randomly selecting features and split values.

### Implementation

```python
class IsolationForestDetector:
    """Isolation Forest outlier detection"""
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )
        
    def fit(self, X):
        """Fit the isolation forest"""
        self.forest.fit(X)
        return self
    
    def predict(self, X):
        """Predict outliers (-1 for outliers, 1 for normal)"""
        return self.forest.predict(X)
    
    def decision_function(self, X):
        """Compute anomaly scores"""
        return self.forest.decision_function(X)

def demonstrate_isolation_forest():
    """Demonstrate Isolation Forest"""
    print("\n=== Isolation Forest Demo ===")
    
    # Generate 2D data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples)
    
    # Add outliers
    outliers = np.random.multivariate_normal([4, 4], [[0.5, 0], [0, 0.5]], 50)
    X = np.vstack([normal_data, outliers])
    
    # Fit isolation forest
    iso_forest = IsolationForestDetector(n_estimators=100, contamination=0.05)
    iso_forest.fit(X)
    
    # Predict outliers
    predictions = iso_forest.predict(X)
    scores = iso_forest.decision_function(X)
    
    is_outlier = predictions == -1
    
    print(f"Detected {np.sum(is_outlier)} outliers out of {len(X)} points")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Anomaly scores
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.title('Anomaly Scores')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('Isolation Forest Outlier Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return iso_forest, X, is_outlier

# Run demonstration
iso_forest, X_data, is_outlier = demonstrate_isolation_forest()
```

## One-Class SVM

### Mathematical Foundation

One-Class SVM learns a decision boundary that encompasses most of the data points. It maps data to a high-dimensional space and finds a hyperplane that separates the data from the origin.

### Implementation

```python
class OneClassSVMDetector:
    """One-Class SVM outlier detection"""
    
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale', random_state=42):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.svm = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state
        )
        
    def fit(self, X):
        """Fit the One-Class SVM"""
        self.svm.fit(X)
        return self
    
    def predict(self, X):
        """Predict outliers (-1 for outliers, 1 for normal)"""
        return self.svm.predict(X)
    
    def decision_function(self, X):
        """Compute anomaly scores"""
        return self.svm.decision_function(X)

def demonstrate_oneclass_svm():
    """Demonstrate One-Class SVM"""
    print("\n=== One-Class SVM Demo ===")
    
    # Generate 2D data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data (non-linear boundary)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.normal(2, 0.5, n_samples)
    normal_data = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    
    # Add outliers
    outliers = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 50)
    X = np.vstack([normal_data, outliers])
    
    # Fit One-Class SVM
    oc_svm = OneClassSVMDetector(nu=0.1, kernel='rbf', gamma='scale')
    oc_svm.fit(X)
    
    # Predict outliers
    predictions = oc_svm.predict(X)
    scores = oc_svm.decision_function(X)
    
    is_outlier = predictions == -1
    
    print(f"Detected {np.sum(is_outlier)} outliers out of {len(X)} points")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Anomaly scores
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.title('Anomaly Scores')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('One-Class SVM Outlier Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return oc_svm, X, is_outlier

# Run demonstration
oc_svm, X_data, is_outlier = demonstrate_oneclass_svm()
```

## Local Outlier Factor (LOF)

### Mathematical Foundation

LOF measures the local density deviation of a data point compared to its neighbors. A point with a significantly lower density than its neighbors is considered an outlier.

### Implementation

```python
class LocalOutlierFactorDetector:
    """Local Outlier Factor outlier detection"""
    
    def __init__(self, n_neighbors=20, contamination=0.1, metric='minkowski'):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric
        )
        
    def fit_predict(self, X):
        """Fit and predict outliers"""
        return self.lof.fit_predict(X)
    
    def decision_function(self, X):
        """Compute anomaly scores"""
        return self.lof.decision_function(X)

def demonstrate_lof():
    """Demonstrate Local Outlier Factor"""
    print("\n=== Local Outlier Factor Demo ===")
    
    # Generate 2D data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data (clusters)
    cluster1 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_samples // 2)
    cluster2 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples // 2)
    normal_data = np.vstack([cluster1, cluster2])
    
    # Add outliers (between clusters)
    outliers = np.random.multivariate_normal([0, 0], [[0.2, 0], [0, 0.2]], 50)
    X = np.vstack([normal_data, outliers])
    
    # Fit LOF
    lof_detector = LocalOutlierFactorDetector(n_neighbors=20, contamination=0.05)
    predictions = lof_detector.fit_predict(X)
    scores = lof_detector.decision_function(X)
    
    is_outlier = predictions == -1
    
    print(f"Detected {np.sum(is_outlier)} outliers out of {len(X)} points")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Anomaly scores
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.title('Anomaly Scores')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('LOF Outlier Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return lof_detector, X, is_outlier

# Run demonstration
lof_detector, X_data, is_outlier = demonstrate_lof()
```

## Mahalanobis Distance

### Mathematical Foundation

Mahalanobis distance measures the distance between a point and a distribution, accounting for correlations between variables:

```math
d(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
```

### Implementation

```python
class MahalanobisOutlierDetector:
    """Mahalanobis distance based outlier detection"""
    
    def __init__(self, threshold=None):
        self.threshold = threshold
        self.mean = None
        self.covariance = None
        
    def fit(self, X):
        """Fit the detector"""
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.covariance = np.cov(X.T)
        return self
    
    def mahalanobis_distance(self, X):
        """Compute Mahalanobis distances"""
        X = np.array(X)
        diff = X - self.mean
        
        # Compute inverse covariance matrix
        try:
            inv_cov = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            # Add regularization if matrix is singular
            inv_cov = np.linalg.inv(self.covariance + 1e-6 * np.eye(self.covariance.shape[0]))
        
        # Compute Mahalanobis distances
        distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        return distances
    
    def predict(self, X):
        """Predict outliers"""
        distances = self.mahalanobis_distance(X)
        
        if self.threshold is None:
            # Use chi-squared distribution threshold
            from scipy.stats import chi2
            self.threshold = chi2.ppf(0.95, df=X.shape[1])
        
        return distances > self.threshold

def demonstrate_mahalanobis():
    """Demonstrate Mahalanobis distance outlier detection"""
    print("\n=== Mahalanobis Distance Demo ===")
    
    # Generate 2D data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data (correlated)
    normal_data = np.random.multivariate_normal(
        [0, 0], [[1, 0.8], [0.8, 1]], n_samples
    )
    
    # Add outliers
    outliers = np.random.multivariate_normal([4, 4], [[0.5, 0], [0, 0.5]], 50)
    X = np.vstack([normal_data, outliers])
    
    # Fit Mahalanobis detector
    mahal_detector = MahalanobisOutlierDetector()
    mahal_detector.fit(X)
    
    # Predict outliers
    distances = mahal_detector.mahalanobis_distance(X)
    is_outlier = mahal_detector.predict(X)
    
    print(f"Detected {np.sum(is_outlier)} outliers out of {len(X)} points")
    print(f"Threshold: {mahal_detector.threshold:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Mahalanobis distances
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=distances, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Mahalanobis Distance')
    plt.title('Mahalanobis Distances')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~is_outlier
    outlier_mask = is_outlier
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('Mahalanobis Outlier Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return mahal_detector, X, is_outlier

# Run demonstration
mahal_detector, X_data, is_outlier = demonstrate_mahalanobis()
```

## Comparison of Methods

```python
def compare_outlier_detection_methods():
    """Compare different outlier detection methods"""
    print("\n=== Outlier Detection Methods Comparison ===")
    
    # Generate complex data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data (multiple clusters)
    cluster1 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_samples // 3)
    cluster2 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples // 3)
    cluster3 = np.random.multivariate_normal([0, 0], [[0.3, 0.2], [0.2, 0.3]], n_samples // 3)
    normal_data = np.vstack([cluster1, cluster2, cluster3])
    
    # Add outliers
    outliers = np.random.multivariate_normal([4, 4], [[0.2, 0], [0, 0.2]], 50)
    X = np.vstack([normal_data, outliers])
    
    # Define methods
    methods = {
        'Z-Score': ZScoreOutlierDetector(threshold=3.0),
        'IQR': IQROutlierDetector(factor=1.5),
        'Robust Z-Score': RobustStatistics(),
        'Isolation Forest': IsolationForestDetector(contamination=0.05),
        'One-Class SVM': OneClassSVMDetector(nu=0.1),
        'LOF': LocalOutlierFactorDetector(contamination=0.05),
        'Mahalanobis': MahalanobisOutlierDetector()
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        
        try:
            if name == 'Robust Z-Score':
                method.fit(X)
                is_outlier = method.detect_outliers(X, threshold=3.0)
            elif name == 'LOF':
                predictions = method.fit_predict(X)
                is_outlier = predictions == -1
            else:
                method.fit(X)
                is_outlier = method.predict(X)
            
            results[name] = {
                'outliers': np.sum(is_outlier),
                'percentage': np.mean(is_outlier) * 100
            }
            
            print(f"  Detected {np.sum(is_outlier)} outliers ({np.mean(is_outlier)*100:.1f}%)")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {'outliers': 0, 'percentage': 0}
    
    # Visualize comparison
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Outlier percentage comparison
    plt.subplot(1, 3, 2)
    method_names = list(results.keys())
    percentages = [results[name]['percentage'] for name in method_names]
    
    bars = plt.bar(method_names, percentages)
    plt.xlabel('Method')
    plt.ylabel('Outlier Percentage (%)')
    plt.title('Outlier Detection Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Number of outliers comparison
    plt.subplot(1, 3, 3)
    outlier_counts = [results[name]['outliers'] for name in method_names]
    
    bars = plt.bar(method_names, outlier_counts)
    plt.xlabel('Method')
    plt.ylabel('Number of Outliers')
    plt.title('Outlier Count Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run comparison
comparison_results = compare_outlier_detection_methods()
```

## Summary

Outlier detection and robust estimation provide essential tools for:

1. **Data Quality**: Identifying and handling problematic data points
2. **Model Robustness**: Building models that are insensitive to outliers
3. **Anomaly Detection**: Finding unusual patterns in data
4. **Statistical Inference**: Making reliable statistical conclusions
5. **Data Preprocessing**: Cleaning data before analysis

Key takeaways:

- **Statistical Methods**: Simple but sensitive to distribution assumptions
- **Distance-based Methods**: Effective for high-dimensional data
- **Density-based Methods**: Good for detecting local outliers
- **Isolation-based Methods**: Efficient for large datasets
- **Robust Statistics**: Provide reliable estimates in presence of outliers
- **Method Selection**: Choose based on data characteristics and requirements

Understanding these techniques provides a solid foundation for robust data analysis and machine learning. 