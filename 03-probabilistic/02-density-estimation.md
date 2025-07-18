# Density Estimation

## Overview

Density estimation is the process of estimating the probability density function of a random variable from observed data. This guide covers both parametric and non-parametric approaches to density estimation, including mixture models, histograms, and kernel density estimation.

## Key Concepts

### 1. What is Density Estimation?

Density estimation aims to estimate the probability density function $`p(x)`$ from a set of observations $`\{x_1, x_2, \ldots, x_n\}`$. The estimated density should satisfy:

```math
\int_{-\infty}^{\infty} \hat{p}(x) dx = 1
```

and

```math
\hat{p}(x) \geq 0 \quad \forall x
```

### 2. Types of Density Estimation

1. **Parametric**: Assume a specific functional form (e.g., Gaussian)
2. **Non-parametric**: No assumptions about functional form
3. **Semi-parametric**: Combine parametric and non-parametric approaches

## Mixture of Gaussians (MoG)

### Mathematical Foundation

A Mixture of Gaussians models the data as a weighted combination of multiple Gaussian distributions:

```math
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)
```

Where:
- $`\pi_k`$ are the mixing coefficients ($`\sum_{k=1}^{K} \pi_k = 1`$)
- $`\mu_k`$ are the means of the Gaussian components
- $`\sigma_k^2`$ are the variances of the Gaussian components

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
import seaborn as sns

class MixtureOfGaussians:
    """Mixture of Gaussians density estimation"""
    
    def __init__(self, n_components=3, max_iter=100, tolerance=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.means = None
        self.variances = None
        self.weights = None
        self.responsibilities = None
        
    def _initialize_parameters(self, X):
        """Initialize MoG parameters"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Random initialization of means
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices].copy()
        
        # Initialize variances as sample variance
        self.variances = np.var(X) * np.ones(self.n_components)
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
    def _e_step(self, X):
        """E-step: Compute responsibilities"""
        n_samples = X.shape[0]
        self.responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Compute probability density for each component
            self.responsibilities[:, k] = self.weights[k] * norm.pdf(
                X, loc=self.means[k], scale=np.sqrt(self.variances[k])
            )
        
        # Normalize responsibilities
        sum_resp = np.sum(self.responsibilities, axis=1, keepdims=True)
        self.responsibilities /= sum_resp
        
        return self.responsibilities
    
    def _m_step(self, X):
        """M-step: Update parameters"""
        n_samples = X.shape[0]
        
        # Update weights
        self.weights = np.mean(self.responsibilities, axis=0)
        
        # Update means
        for k in range(self.n_components):
            self.means[k] = np.average(X, weights=self.responsibilities[:, k])
        
        # Update variances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.variances[k] = np.average(diff**2, weights=self.responsibilities[:, k])
            
            # Add small regularization
            self.variances[k] = max(self.variances[k], 1e-6)
    
    def fit(self, X):
        """Fit MoG using EM algorithm"""
        X = np.array(X).flatten()
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            self._e_step(X)
            
            # M-step
            self._m_step(X)
        
        return self
    
    def pdf(self, X):
        """Compute probability density function"""
        X = np.array(X).flatten()
        density = np.zeros_like(X, dtype=float)
        
        for k in range(self.n_components):
            density += self.weights[k] * norm.pdf(
                X, loc=self.means[k], scale=np.sqrt(self.variances[k])
            )
        
        return density
    
    def sample(self, n_samples=1000):
        """Sample from the fitted MoG"""
        # Sample component assignments
        component_assignments = np.random.choice(
            self.n_components, size=n_samples, p=self.weights
        )
        
        # Sample from each component
        samples = []
        for assignment in component_assignments:
            sample = np.random.normal(
                self.means[assignment], np.sqrt(self.variances[assignment])
            )
            samples.append(sample)
        
        return np.array(samples)

def demonstrate_mog():
    """Demonstrate Mixture of Gaussians"""
    print("=== Mixture of Gaussians Demo ===")
    
    # Generate data from multiple Gaussians
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data from 3 Gaussians
    data1 = np.random.normal(-2, 1, n_samples // 3)
    data2 = np.random.normal(0, 1.5, n_samples // 3)
    data3 = np.random.normal(3, 0.8, n_samples // 3)
    
    X = np.concatenate([data1, data2, data3])
    
    # Fit MoG
    mog = MixtureOfGaussians(n_components=3, max_iter=100)
    mog.fit(X)
    
    # Generate samples from fitted model
    samples = mog.sample(n_samples=1000)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data histogram
    plt.subplot(1, 3, 1)
    plt.hist(X, bins=50, density=True, alpha=0.7, label='Original Data')
    plt.title('Original Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Fitted MoG density
    plt.subplot(1, 3, 2)
    x_range = np.linspace(X.min(), X.max(), 200)
    density = mog.pdf(x_range)
    plt.plot(x_range, density, 'r-', linewidth=2, label='MoG Density')
    plt.hist(X, bins=50, density=True, alpha=0.3, label='Original Data')
    plt.title('Fitted MoG Density')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Generated samples
    plt.subplot(1, 3, 3)
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='Generated Samples')
    plt.plot(x_range, density, 'r-', linewidth=2, label='MoG Density')
    plt.title('Generated Samples')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return mog, X, samples

# Run demonstration
mog_model, X_data, samples = demonstrate_mog()
```

## Histograms

### Mathematical Foundation

Histograms provide a simple non-parametric estimate of the density function by counting observations in bins:

```math
\hat{p}(x) = \frac{n_i}{n \cdot h}
```

Where:
- $`n_i`$ is the number of observations in bin $`i`$
- $`n`$ is the total number of observations
- $`h`$ is the bin width

### Implementation

```python
class HistogramDensityEstimator:
    """Histogram-based density estimation"""
    
    def __init__(self, bins='auto', range=None):
        self.bins = bins
        self.range = range
        self.bin_edges = None
        self.bin_counts = None
        self.bin_centers = None
        
    def fit(self, X):
        """Fit histogram density estimator"""
        X = np.array(X).flatten()
        
        # Compute histogram
        self.bin_counts, self.bin_edges = np.histogram(
            X, bins=self.bins, range=self.range, density=True
        )
        
        # Compute bin centers
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        return self
    
    def pdf(self, X):
        """Compute probability density function"""
        X = np.array(X).flatten()
        density = np.zeros_like(X, dtype=float)
        
        for i, x in enumerate(X):
            # Find which bin x belongs to
            bin_idx = np.digitize(x, self.bin_edges) - 1
            
            # Handle edge cases
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= len(self.bin_counts):
                bin_idx = len(self.bin_counts) - 1
            
            density[i] = self.bin_counts[bin_idx]
        
        return density
    
    def plot(self, X=None, ax=None):
        """Plot histogram density estimate"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.bar(self.bin_centers, self.bin_counts, 
               width=np.diff(self.bin_edges), alpha=0.7, label='Histogram')
        
        if X is not None:
            # Plot original data
            ax.hist(X, bins=self.bins, density=True, alpha=0.3, label='Original Data')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Histogram Density Estimation')
        ax.legend()
        
        return ax

def demonstrate_histogram():
    """Demonstrate histogram density estimation"""
    print("\n=== Histogram Density Estimation Demo ===")
    
    # Generate data
    np.random.seed(42)
    X = np.random.normal(0, 1, 1000)
    
    # Fit histogram estimator
    hist_estimator = HistogramDensityEstimator(bins=30)
    hist_estimator.fit(X)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Different bin sizes
    bin_sizes = [10, 30, 100]
    
    for i, bins in enumerate(bin_sizes):
        plt.subplot(1, 3, i+1)
        
        hist_est = HistogramDensityEstimator(bins=bins)
        hist_est.fit(X)
        hist_est.plot(X)
        plt.title(f'Histogram with {bins} bins')
    
    plt.tight_layout()
    plt.show()
    
    return hist_estimator, X

# Run demonstration
hist_model, X_data = demonstrate_histogram()
```

## Kernel Density Estimation (KDE)

### Mathematical Foundation

KDE estimates the density function by placing a kernel function at each data point:

```math
\hat{p}(x) = \frac{1}{n \cdot h} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
```

Where:
- $`K(\cdot)`$ is the kernel function
- $`h`$ is the bandwidth parameter
- $`n`$ is the number of observations

Common kernel functions:
- **Gaussian**: $`K(u) = \frac{1}{\sqrt{2\pi}} e^{-\frac{u^2}{2}}`$
- **Epanechnikov**: $`K(u) = \frac{3}{4}(1 - u^2)`$ for $`|u| \leq 1`$
- **Uniform**: $`K(u) = \frac{1}{2}`$ for $`|u| \leq 1`$

### Implementation

```python
class KernelDensityEstimator:
    """Kernel Density Estimation"""
    
    def __init__(self, bandwidth=None, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.X_train = None
        
    def _gaussian_kernel(self, u):
        """Gaussian kernel function"""
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    
    def _epanechnikov_kernel(self, u):
        """Epanechnikov kernel function"""
        mask = np.abs(u) <= 1
        result = np.zeros_like(u)
        result[mask] = 0.75 * (1 - u[mask]**2)
        return result
    
    def _uniform_kernel(self, u):
        """Uniform kernel function"""
        mask = np.abs(u) <= 1
        result = np.zeros_like(u)
        result[mask] = 0.5
        return result
    
    def _compute_kernel(self, u):
        """Compute kernel function"""
        if self.kernel == 'gaussian':
            return self._gaussian_kernel(u)
        elif self.kernel == 'epanechnikov':
            return self._epanechnikov_kernel(u)
        elif self.kernel == 'uniform':
            return self._uniform_kernel(u)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _silverman_bandwidth(self, X):
        """Compute Silverman's rule of thumb bandwidth"""
        n = len(X)
        sigma = np.std(X)
        return sigma * (4 / (3 * n)) ** 0.2
    
    def fit(self, X):
        """Fit KDE"""
        X = np.array(X).flatten()
        self.X_train = X
        
        # Set bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._silverman_bandwidth(X)
        
        return self
    
    def pdf(self, X):
        """Compute probability density function"""
        X = np.array(X).flatten()
        density = np.zeros_like(X, dtype=float)
        
        for i, x in enumerate(X):
            # Compute kernel contributions from all training points
            u = (x - self.X_train) / self.bandwidth
            kernel_values = self._compute_kernel(u)
            density[i] = np.mean(kernel_values) / self.bandwidth
        
        return density
    
    def sample(self, n_samples=1000):
        """Sample from the fitted KDE"""
        # Randomly select training points
        indices = np.random.choice(len(self.X_train), size=n_samples)
        selected_points = self.X_train[indices]
        
        # Add noise from kernel
        if self.kernel == 'gaussian':
            noise = np.random.normal(0, self.bandwidth, n_samples)
        elif self.kernel == 'epanechnikov':
            noise = self.bandwidth * np.random.uniform(-1, 1, n_samples)
        elif self.kernel == 'uniform':
            noise = self.bandwidth * np.random.uniform(-1, 1, n_samples)
        
        return selected_points + noise

def demonstrate_kde():
    """Demonstrate Kernel Density Estimation"""
    print("\n=== Kernel Density Estimation Demo ===")
    
    # Generate data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(-2, 1, 300),
        np.random.normal(2, 1.5, 700)
    ])
    
    # Fit KDE with different kernels
    kernels = ['gaussian', 'epanechnikov', 'uniform']
    kde_models = {}
    
    for kernel in kernels:
        kde = KernelDensityEstimator(kernel=kernel)
        kde.fit(X)
        kde_models[kernel] = kde
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    x_range = np.linspace(X.min() - 1, X.max() + 1, 200)
    
    for i, kernel in enumerate(kernels):
        plt.subplot(1, 3, i+1)
        
        # Plot original data
        plt.hist(X, bins=50, density=True, alpha=0.3, label='Original Data')
        
        # Plot KDE
        density = kde_models[kernel].pdf(x_range)
        plt.plot(x_range, density, 'r-', linewidth=2, label=f'{kernel.capitalize()} KDE')
        
        plt.title(f'{kernel.capitalize()} Kernel')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compare bandwidth selection
    plt.figure(figsize=(12, 4))
    
    bandwidths = [0.1, 0.5, 1.0, 2.0]
    
    for i, bw in enumerate(bandwidths):
        plt.subplot(1, 4, i+1)
        
        kde = KernelDensityEstimator(bandwidth=bw, kernel='gaussian')
        kde.fit(X)
        
        plt.hist(X, bins=50, density=True, alpha=0.3, label='Original Data')
        density = kde.pdf(x_range)
        plt.plot(x_range, density, 'r-', linewidth=2, label=f'KDE (h={bw})')
        
        plt.title(f'Bandwidth = {bw}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return kde_models, X

# Run demonstration
kde_models, X_data = demonstrate_kde()
```

## Multivariate Density Estimation

### Mathematical Foundation

For multivariate data, the density estimation extends to higher dimensions:

**Multivariate KDE**:
```math
\hat{p}(\mathbf{x}) = \frac{1}{n \cdot |\mathbf{H}|^{1/2}} \sum_{i=1}^{n} K\left(\mathbf{H}^{-1/2}(\mathbf{x} - \mathbf{x}_i)\right)
```

Where $`\mathbf{H}`$ is the bandwidth matrix.

### Implementation

```python
class MultivariateKDE:
    """Multivariate Kernel Density Estimation"""
    
    def __init__(self, bandwidth=None, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.X_train = None
        
    def _gaussian_kernel(self, u):
        """Multivariate Gaussian kernel"""
        return np.exp(-0.5 * np.sum(u**2, axis=1)) / (2 * np.pi)**(u.shape[1] / 2)
    
    def _silverman_bandwidth(self, X):
        """Compute Silverman's rule of thumb bandwidth matrix"""
        n, d = X.shape
        sigma = np.std(X, axis=0)
        return np.diag(sigma * (4 / ((d + 2) * n)) ** (1 / (d + 4)))
    
    def fit(self, X):
        """Fit multivariate KDE"""
        X = np.array(X)
        self.X_train = X
        
        # Set bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._silverman_bandwidth(X)
        elif np.isscalar(self.bandwidth):
            self.bandwidth = self.bandwidth * np.eye(X.shape[1])
        
        return self
    
    def pdf(self, X):
        """Compute probability density function"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        density = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            # Compute kernel contributions from all training points
            diff = self.X_train - x
            u = np.linalg.solve(self.bandwidth, diff.T).T
            kernel_values = self._gaussian_kernel(u)
            density[i] = np.mean(kernel_values) / np.sqrt(np.linalg.det(self.bandwidth))
        
        return density

def demonstrate_multivariate_kde():
    """Demonstrate multivariate KDE"""
    print("\n=== Multivariate KDE Demo ===")
    
    # Generate 2D data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data from multiple 2D Gaussians
    data1 = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], n_samples // 3)
    data2 = np.random.multivariate_normal([2, 2], [[1.5, -0.3], [-0.3, 1.5]], n_samples // 3)
    data3 = np.random.multivariate_normal([0, 0], [[0.8, 0], [0, 0.8]], n_samples // 3)
    
    X = np.vstack([data1, data2, data3])
    
    # Fit multivariate KDE
    kde = MultivariateKDE()
    kde.fit(X)
    
    # Create grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute density on grid
    density = kde.pdf(grid_points).reshape(xx.shape)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # KDE density
    plt.subplot(1, 3, 2)
    plt.contourf(xx, yy, density, levels=20, cmap='Blues')
    plt.colorbar(label='Density')
    plt.title('KDE Density Estimate')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Combined plot
    plt.subplot(1, 3, 3)
    plt.contourf(xx, yy, density, levels=20, cmap='Blues', alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c='red', s=10)
    plt.colorbar(label='Density')
    plt.title('Data + KDE Density')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return kde, X

# Run demonstration
multivariate_kde, X_data = demonstrate_multivariate_kde()
```

## Model Selection and Comparison

### Mathematical Foundation

For model selection in density estimation, we can use:

1. **Cross-validation**: Maximize log-likelihood on held-out data
2. **AIC/BIC**: Balance fit and complexity
3. **Kullback-Leibler divergence**: Measure distance to true density

### Implementation

```python
def compare_density_estimators():
    """Compare different density estimation methods"""
    print("\n=== Density Estimation Comparison ===")
    
    # Generate data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(-2, 1, 300),
        np.random.normal(2, 1.5, 700)
    ])
    
    # Split data
    X_train = X[:800]
    X_test = X[800:]
    
    # Fit different estimators
    estimators = {
        'Histogram': HistogramDensityEstimator(bins=30),
        'KDE (Gaussian)': KernelDensityEstimator(kernel='gaussian'),
        'KDE (Epanechnikov)': KernelDensityEstimator(kernel='epanechnikov'),
        'MoG (3 components)': MixtureOfGaussians(n_components=3),
        'MoG (5 components)': MixtureOfGaussians(n_components=5)
    }
    
    results = {}
    
    for name, estimator in estimators.items():
        # Fit estimator
        estimator.fit(X_train)
        
        # Compute log-likelihood on test set
        if hasattr(estimator, 'pdf'):
            density = estimator.pdf(X_test)
            log_likelihood = np.mean(np.log(density + 1e-10))
        else:
            log_likelihood = np.nan
        
        results[name] = {
            'log_likelihood': log_likelihood,
            'estimator': estimator
        }
        
        print(f"{name}: Log-likelihood = {log_likelihood:.4f}")
    
    # Visualize comparison
    plt.figure(figsize=(15, 5))
    
    # Plot all estimators
    x_range = np.linspace(X.min() - 1, X.max() + 1, 200)
    
    plt.subplot(1, 2, 1)
    plt.hist(X, bins=50, density=True, alpha=0.3, label='Original Data')
    
    for name, result in results.items():
        if hasattr(result['estimator'], 'pdf'):
            density = result['estimator'].pdf(x_range)
            plt.plot(x_range, density, linewidth=2, label=name)
    
    plt.title('Density Estimation Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot log-likelihood comparison
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    log_likelihoods = [results[name]['log_likelihood'] for name in names]
    
    bars = plt.bar(names, log_likelihoods)
    plt.xlabel('Estimator')
    plt.ylabel('Log-Likelihood')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run comparison
comparison_results = compare_density_estimators()
```

## Practical Applications

### Anomaly Detection

```python
def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection using density estimation"""
    print("\n=== Anomaly Detection Demo ===")
    
    # Generate normal and anomalous data
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.normal(0, 1, 1000)
    
    # Anomalous data
    anomalous_data = np.random.normal(4, 1, 50)
    
    # Combine data
    X = np.concatenate([normal_data, anomalous_data])
    labels = np.concatenate([np.zeros(1000), np.ones(50)])
    
    # Fit density estimator on normal data only
    kde = KernelDensityEstimator(kernel='gaussian')
    kde.fit(normal_data)
    
    # Compute density for all data
    density = kde.pdf(X)
    
    # Set threshold for anomaly detection
    threshold = np.percentile(density, 5)  # 5th percentile
    
    # Predict anomalies
    predictions = density < threshold
    
    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Anomaly Detection Results:")
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Density plot
    plt.subplot(1, 3, 1)
    x_range = np.linspace(X.min(), X.max(), 200)
    density_range = kde.pdf(x_range)
    plt.plot(x_range, density_range, 'b-', linewidth=2, label='Density Estimate')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.title('Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Data with predictions
    plt.subplot(1, 3, 2)
    normal_mask = predictions == 0
    anomaly_mask = predictions == 1
    
    plt.scatter(X[normal_mask], density[normal_mask], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[anomaly_mask], density[anomaly_mask], 
               alpha=0.6, label='Anomaly', c='red')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Anomaly Detection')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    return kde, X, labels, predictions

# Run demonstration
anomaly_kde, X_data, labels, predictions = demonstrate_anomaly_detection()
```

## Summary

Density estimation provides powerful tools for:

1. **Understanding Data Distribution**: Visualizing and understanding data structure
2. **Anomaly Detection**: Identifying unusual data points
3. **Data Generation**: Sampling from estimated distributions
4. **Model Selection**: Choosing appropriate density models
5. **Uncertainty Quantification**: Estimating confidence in predictions

Key takeaways:

- **Parametric vs Non-parametric**: Choose based on data characteristics
- **Bandwidth Selection**: Critical for KDE performance
- **Model Complexity**: Balance between fit and generalization
- **Multivariate Extension**: Handle high-dimensional data
- **Practical Applications**: Anomaly detection, data generation, uncertainty quantification

Understanding these techniques provides a solid foundation for probabilistic modeling and statistical inference. 