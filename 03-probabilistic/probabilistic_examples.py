"""
Probabilistic & Statistical Methods Examples

This file contains comprehensive examples demonstrating:
- Expectation-Maximization (EM) algorithm
- Gaussian Mixture Models (GMM)
- Hidden Markov Models (HMM)
- Latent Dirichlet Allocation (LDA)
- Density estimation techniques
- Outlier detection methods
- Robust statistical estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, multivariate_normal, chi2
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, silhouette_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# EM Algorithm and Latent Variable Models
# ============================================================================

class GaussianMixtureModel:
    """Gaussian Mixture Model implementation using EM algorithm"""
    
    def __init__(self, n_components=3, max_iter=100, tolerance=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None
        self.log_likelihood_history = []
        
    def _initialize_parameters(self, X):
        """Initialize GMM parameters"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Random initialization of means
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices].copy()
        
        # Initialize covariances as identity matrices
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
    def _e_step(self, X):
        """E-step: Compute responsibilities"""
        n_samples = X.shape[0]
        self.responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Compute probability density for each component
            self.responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
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
            self.means[k] = np.average(X, axis=0, weights=self.responsibilities[:, k])
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = self.responsibilities[:, k:k+1] * diff
            self.covariances[k] = np.dot(weighted_diff.T, diff) / np.sum(self.responsibilities[:, k])
            
            # Add small regularization to ensure positive definiteness
            self.covariances[k] += 1e-6 * np.eye(X.shape[1])
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood"""
        log_likelihood = 0
        for i in range(X.shape[0]):
            prob = 0
            for k in range(self.n_components):
                prob += self.weights[k] * multivariate_normal.pdf(
                    X[i], mean=self.means[k], cov=self.covariances[k]
                )
            log_likelihood += np.log(prob + 1e-10)
        return log_likelihood
    
    def fit(self, X):
        """Fit GMM using EM algorithm"""
        X = np.array(X)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            self._e_step(X)
            
            # M-step
            self._m_step(X)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                if abs(log_likelihood - self.log_likelihood_history[-2]) < self.tolerance:
                    break
        
        return self
    
    def predict(self, X):
        """Predict cluster assignments"""
        X = np.array(X)
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """Predict cluster probabilities"""
        X = np.array(X)
        return self._e_step(X)
    
    def sample(self, n_samples=1000):
        """Sample from the fitted GMM"""
        # Sample component assignments
        component_assignments = np.random.choice(
            self.n_components, size=n_samples, p=self.weights
        )
        
        # Sample from each component
        samples = []
        for assignment in component_assignments:
            sample = np.random.multivariate_normal(
                self.means[assignment], self.covariances[assignment]
            )
            samples.append(sample)
        
        return np.array(samples)

def demonstrate_gmm():
    """Demonstrate Gaussian Mixture Model"""
    print("=== Gaussian Mixture Model Demo ===")
    
    # Generate data from multiple Gaussians
    X, y_true = make_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42)
    
    # Fit GMM
    gmm = GaussianMixtureModel(n_components=3, max_iter=100)
    gmm.fit(X)
    
    # Make predictions
    y_pred = gmm.predict(X)
    
    # Evaluate clustering
    ari = adjusted_rand_score(y_true, y_pred)
    sil = silhouette_score(X, y_pred)
    
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Silhouette Score: {sil:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # GMM clustering
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.6)
    plt.title('GMM Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Log-likelihood convergence
    plt.subplot(1, 3, 3)
    plt.plot(gmm.log_likelihood_history)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('EM Convergence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return gmm, X, y_true, y_pred

# ============================================================================
# Density Estimation
# ============================================================================

class KernelDensityEstimator:
    """Kernel Density Estimation"""
    
    def __init__(self, bandwidth=None, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.X_train = None
        
    def _gaussian_kernel(self, u):
        """Gaussian kernel function"""
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    
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
            kernel_values = self._gaussian_kernel(u)
            density[i] = np.mean(kernel_values) / self.bandwidth
        
        return density
    
    def sample(self, n_samples=1000):
        """Sample from the fitted KDE"""
        # Randomly select training points
        indices = np.random.choice(len(self.X_train), size=n_samples)
        selected_points = self.X_train[indices]
        
        # Add noise from kernel
        noise = np.random.normal(0, self.bandwidth, n_samples)
        
        return selected_points + noise

def demonstrate_density_estimation():
    """Demonstrate density estimation techniques"""
    print("\n=== Density Estimation Demo ===")
    
    # Generate data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(-2, 1, 300),
        np.random.normal(2, 1.5, 700)
    ])
    
    # Fit KDE
    kde = KernelDensityEstimator()
    kde.fit(X)
    
    # Generate samples
    samples = kde.sample(n_samples=1000)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data histogram
    plt.subplot(1, 3, 1)
    plt.hist(X, bins=50, density=True, alpha=0.7, label='Original Data')
    plt.title('Original Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # KDE density
    plt.subplot(1, 3, 2)
    x_range = np.linspace(X.min(), X.max(), 200)
    density = kde.pdf(x_range)
    plt.plot(x_range, density, 'r-', linewidth=2, label='KDE Density')
    plt.hist(X, bins=50, density=True, alpha=0.3, label='Original Data')
    plt.title('KDE Density Estimate')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Generated samples
    plt.subplot(1, 3, 3)
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='Generated Samples')
    plt.plot(x_range, density, 'r-', linewidth=2, label='KDE Density')
    plt.title('Generated Samples')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return kde, X, samples

# ============================================================================
# Outlier Detection
# ============================================================================

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
        self.robust_mean = np.mean(X, axis=0)  # Simplified for demo
        
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

def demonstrate_outlier_detection():
    """Demonstrate outlier detection methods"""
    print("\n=== Outlier Detection Demo ===")
    
    # Generate data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.normal(0, 1, n_samples)
    
    # Add outliers
    outliers = np.random.normal(5, 1, 50)
    X = np.concatenate([normal_data, outliers])
    
    # Fit different detectors
    zscore_detector = ZScoreOutlierDetector(threshold=3.0)
    zscore_detector.fit(X.reshape(-1, 1))
    
    robust_stats = RobustStatistics()
    robust_stats.fit(X.reshape(-1, 1))
    
    # Predict outliers
    zscore_outliers = zscore_detector.predict(X.reshape(-1, 1))
    robust_outliers = robust_stats.detect_outliers(X.reshape(-1, 1), threshold=3.0)
    
    print(f"Z-score detected {np.sum(zscore_outliers)} outliers")
    print(f"Robust method detected {np.sum(robust_outliers)} outliers")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Z-score method
    plt.subplot(1, 3, 1)
    z_scores = zscore_detector.z_scores(X.reshape(-1, 1)).flatten()
    plt.scatter(range(len(z_scores)), z_scores, alpha=0.6, c=zscore_outliers, cmap='viridis')
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Z-Score Method')
    plt.xlabel('Data Point')
    plt.ylabel('Z-Score')
    plt.legend()
    
    # Robust method
    plt.subplot(1, 3, 2)
    robust_z_scores = robust_stats.robust_z_scores(X.reshape(-1, 1)).flatten()
    plt.scatter(range(len(robust_z_scores)), robust_z_scores, alpha=0.6, c=robust_outliers, cmap='viridis')
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Robust Method')
    plt.xlabel('Data Point')
    plt.ylabel('Robust Z-Score')
    plt.legend()
    
    # Data with outliers highlighted
    plt.subplot(1, 3, 3)
    normal_mask = ~zscore_outliers
    outlier_mask = zscore_outliers
    
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
    
    return zscore_detector, robust_stats, X, zscore_outliers, robust_outliers

# ============================================================================
# Advanced Outlier Detection
# ============================================================================

def demonstrate_advanced_outlier_detection():
    """Demonstrate advanced outlier detection methods"""
    print("\n=== Advanced Outlier Detection Demo ===")
    
    # Generate 2D data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples)
    
    # Add outliers
    outliers = np.random.multivariate_normal([4, 4], [[0.5, 0], [0, 0.5]], 50)
    X = np.vstack([normal_data, outliers])
    
    # Fit different detectors
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    oc_svm = OneClassSVM(nu=0.1, kernel='rbf')
    lof = LocalOutlierFactor(contamination=0.05)
    
    # Predict outliers
    iso_predictions = iso_forest.fit_predict(X)
    oc_svm.fit(X)
    oc_predictions = oc_svm.predict(X)
    lof_predictions = lof.fit_predict(X)
    
    # Convert predictions to boolean
    iso_outliers = iso_predictions == -1
    oc_outliers = oc_predictions == -1
    lof_outliers = lof_predictions == -1
    
    print(f"Isolation Forest detected {np.sum(iso_outliers)} outliers")
    print(f"One-Class SVM detected {np.sum(oc_outliers)} outliers")
    print(f"LOF detected {np.sum(lof_outliers)} outliers")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Original data
    plt.subplot(2, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Isolation Forest
    plt.subplot(2, 3, 2)
    normal_mask = ~iso_outliers
    outlier_mask = iso_outliers
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('Isolation Forest')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # One-Class SVM
    plt.subplot(2, 3, 3)
    normal_mask = ~oc_outliers
    outlier_mask = oc_outliers
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('One-Class SVM')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # LOF
    plt.subplot(2, 3, 4)
    normal_mask = ~lof_outliers
    outlier_mask = lof_outliers
    
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
               alpha=0.6, label='Normal', c='blue')
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
               alpha=0.8, label='Outliers', c='red', s=50)
    plt.title('Local Outlier Factor')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Anomaly scores comparison
    plt.subplot(2, 3, 5)
    iso_scores = iso_forest.decision_function(X)
    oc_scores = oc_svm.decision_function(X)
    lof_scores = lof.decision_function(X)
    
    plt.scatter(iso_scores, oc_scores, alpha=0.6)
    plt.xlabel('Isolation Forest Score')
    plt.ylabel('One-Class SVM Score')
    plt.title('Score Comparison')
    plt.grid(True)
    
    # Method comparison
    plt.subplot(2, 3, 6)
    methods = ['Isolation Forest', 'One-Class SVM', 'LOF']
    outlier_counts = [np.sum(iso_outliers), np.sum(oc_outliers), np.sum(lof_outliers)]
    
    bars = plt.bar(methods, outlier_counts)
    plt.xlabel('Method')
    plt.ylabel('Number of Outliers')
    plt.title('Outlier Detection Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return iso_forest, oc_svm, lof, X, iso_outliers, oc_outliers, lof_outliers

# ============================================================================
# Missing Data Imputation
# ============================================================================

class EMImputation:
    """Missing data imputation using EM algorithm"""
    
    def __init__(self, max_iter=100, tolerance=1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.mean = None
        self.covariance = None
        
    def _initialize_parameters(self, X):
        """Initialize parameters using available data"""
        # Use mean of available data for each feature
        self.mean = np.nanmean(X, axis=0)
        
        # Use covariance of available data
        available_data = X[~np.isnan(X).any(axis=1)]
        if len(available_data) > 1:
            self.covariance = np.cov(available_data.T)
        else:
            self.covariance = np.eye(X.shape[1])
    
    def _e_step(self, X):
        """E-step: Impute missing values"""
        X_imputed = X.copy()
        
        for i in range(X.shape[0]):
            missing_mask = np.isnan(X[i])
            observed_mask = ~missing_mask
            
            if np.any(missing_mask) and np.any(observed_mask):
                # Conditional mean for missing values
                mu_m = self.mean[missing_mask]
                mu_o = self.mean[observed_mask]
                x_o = X[i, observed_mask]
                
                # Extract relevant parts of covariance matrix
                sigma_oo = self.covariance[np.ix_(observed_mask, observed_mask)]
                sigma_mo = self.covariance[np.ix_(missing_mask, observed_mask)]
                
                # Compute conditional mean
                conditional_mean = mu_m + sigma_mo @ np.linalg.solve(sigma_oo, x_o - mu_o)
                X_imputed[i, missing_mask] = conditional_mean
        
        return X_imputed
    
    def _m_step(self, X_imputed):
        """M-step: Update parameters"""
        self.mean = np.mean(X_imputed, axis=0)
        self.covariance = np.cov(X_imputed.T)
    
    def fit_transform(self, X):
        """Fit the model and impute missing values"""
        X = np.array(X)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            X_imputed = self._e_step(X)
            
            # M-step
            self._m_step(X_imputed)
        
        return X_imputed

def demonstrate_missing_data_imputation():
    """Demonstrate missing data imputation"""
    print("\n=== Missing Data Imputation Demo ===")
    
    # Generate data with missing values
    np.random.seed(42)
    n_samples = 1000
    n_features = 3
    
    # Generate complete data
    X_complete = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]],
        size=n_samples
    )
    
    # Introduce missing values
    X_missing = X_complete.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.2
    X_missing[missing_mask] = np.nan
    
    print(f"Missing data percentage: {np.isnan(X_missing).sum() / X_missing.size:.1%}")
    
    # Impute missing values
    em_imputer = EMImputation(max_iter=50)
    X_imputed = em_imputer.fit_transform(X_missing)
    
    # Evaluate imputation quality
    mse = np.mean((X_complete[missing_mask] - X_imputed[missing_mask]) ** 2)
    print(f"Imputation MSE: {mse:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X_complete[:, 0], X_complete[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Data with missing values
    plt.subplot(1, 3, 2)
    available_mask = ~np.isnan(X_missing).any(axis=1)
    plt.scatter(X_missing[available_mask, 0], X_missing[available_mask, 1], alpha=0.6)
    plt.title('Data with Missing Values')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Imputed data
    plt.subplot(1, 3, 3)
    plt.scatter(X_imputed[:, 0], X_imputed[:, 1], alpha=0.6)
    plt.title('Imputed Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return em_imputer, X_complete, X_missing, X_imputed

# ============================================================================
# Main Demonstration
# ============================================================================

if __name__ == "__main__":
    # Run all demonstrations
    print("Probabilistic & Statistical Methods Examples")
    print("=" * 50)
    
    # GMM demonstration
    gmm_model, X_data, y_true, y_pred = demonstrate_gmm()
    
    # Density estimation demonstration
    kde_model, X_data, samples = demonstrate_density_estimation()
    
    # Outlier detection demonstration
    zscore_detector, robust_stats, X_data, zscore_outliers, robust_outliers = demonstrate_outlier_detection()
    
    # Advanced outlier detection demonstration
    iso_forest, oc_svm, lof, X_data, iso_outliers, oc_outliers, lof_outliers = demonstrate_advanced_outlier_detection()
    
    # Missing data imputation demonstration
    imputer, X_complete, X_missing, X_imputed = demonstrate_missing_data_imputation()
    
    print("\nAll demonstrations completed!") 