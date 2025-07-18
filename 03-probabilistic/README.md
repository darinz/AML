# Probabilistic & Statistical Methods

This module covers statistical learning approaches and unsupervised learning techniques that model uncertainty and hidden patterns in data.

## Learning Objectives

By the end of this module, you will be able to:
- Implement Expectation-Maximization (EM) algorithm for latent variable models
- Build density estimation models using various statistical techniques
- Detect and handle outliers using robust estimation methods
- Understand probabilistic modeling and uncertainty quantification
- Apply these methods to unsupervised learning problems

## Topics Covered

### 1. EM and Latent Variables
- **Expectation-Maximization**: Iterative optimization for incomplete data
- **Gaussian Mixture Models (GMM)**: Clustering with probabilistic interpretation
- **Hidden Markov Models (HMM)**: Sequential data modeling
- **Latent Dirichlet Allocation (LDA)**: Topic modeling for documents
- **Missing Data Imputation**: Handling incomplete datasets

### 2. Density Estimation
- **Mixture of Gaussians (MoG)**: Probabilistic clustering and density modeling
- **Histograms**: Non-parametric density estimation
- **Kernel Density Estimation (KDE)**: Smooth density estimation
- **Parametric vs Non-parametric**: Choosing appropriate density models
- **Multivariate Density Estimation**: High-dimensional density modeling

### 3. Outliers and Robust Estimation
- **Anomaly Detection**: Identifying unusual data points
- **Robust Statistics**: Median, MAD, robust covariance
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Learning normal data boundaries
- **Statistical Tests**: Z-score, IQR, Mahalanobis distance

## Comprehensive Guides

This module includes detailed markdown guides with mathematical foundations and implementations:

1. **01-em-latent-variables.md**: Complete guide to EM algorithm and latent variable models
   - Mathematical foundations of EM algorithm
   - Gaussian Mixture Models implementation from scratch
   - Hidden Markov Models with forward-backward algorithm
   - Latent Dirichlet Allocation for topic modeling
   - Missing data imputation techniques

2. **02-density-estimation.md**: Comprehensive density estimation methods
   - Mixture of Gaussians with EM optimization
   - Histogram-based density estimation
   - Kernel Density Estimation with multiple kernels
   - Multivariate density estimation
   - Model selection and comparison

3. **03-outliers-robust-estimation.md**: Outlier detection and robust statistics
   - Statistical outlier detection (Z-score, IQR)
   - Robust statistical estimators
   - Isolation Forest algorithm
   - One-Class SVM for anomaly detection
   - Local Outlier Factor (LOF)
   - Mahalanobis distance methods

## Python Examples

The `probabilistic_examples.py` file contains comprehensive implementations:

### EM Algorithm & Latent Variables
- **GaussianMixtureModel**: Complete GMM implementation with EM
- **EM convergence visualization**: Log-likelihood tracking
- **Cluster evaluation**: Adjusted Rand Index and Silhouette Score

### Density Estimation
- **KernelDensityEstimator**: KDE with Gaussian kernel
- **Bandwidth selection**: Silverman's rule of thumb
- **Density visualization**: Original data vs estimated density
- **Sample generation**: Sampling from fitted distributions

### Outlier Detection
- **ZScoreOutlierDetector**: Classical statistical outlier detection
- **RobustStatistics**: Median, MAD, and robust Z-scores
- **Advanced methods**: Isolation Forest, One-Class SVM, LOF
- **Method comparison**: Performance evaluation across techniques

### Missing Data Imputation
- **EMImputation**: EM-based missing data imputation
- **Conditional mean computation**: Using multivariate normal properties
- **Imputation quality**: MSE evaluation against true values

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running Examples

Run the comprehensive examples:

```bash
python probabilistic_examples.py
```

This will execute all demonstrations:
- Gaussian Mixture Model clustering
- Kernel Density Estimation
- Outlier detection with multiple methods
- Advanced anomaly detection
- Missing data imputation

## Key Learning Outcomes

### Mathematical Understanding
- **EM Algorithm**: Understand the iterative E-step and M-step process
- **Likelihood Maximization**: Learn how EM maximizes incomplete data likelihood
- **Probabilistic Models**: Master GMM, HMM, and LDA formulations
- **Density Estimation**: Understand parametric vs non-parametric approaches
- **Robust Statistics**: Learn alternatives to mean and standard deviation

### Implementation Skills
- **From Scratch**: Implement EM algorithm without using libraries
- **Numerical Stability**: Handle edge cases and convergence issues
- **Model Selection**: Choose appropriate methods for different data types
- **Evaluation Metrics**: Assess clustering and outlier detection quality
- **Visualization**: Create informative plots for model interpretation

### Practical Applications
- **Clustering**: Use GMM for soft clustering with uncertainty
- **Anomaly Detection**: Build systems to identify unusual patterns
- **Topic Modeling**: Extract themes from document collections
- **Missing Data**: Handle incomplete datasets in real-world scenarios
- **Sequential Data**: Model time series and sequence patterns

## Practical Applications

- **Fraud Detection**: Identifying anomalous transactions
- **Quality Control**: Detecting defective products in manufacturing
- **Network Security**: Intrusion detection systems
- **Medical Imaging**: Tumor detection and segmentation
- **Text Analysis**: Topic modeling and document clustering

## Implementation Focus

This module emphasizes **statistical rigor and robustness**:
- Implement EM algorithm from scratch for GMM
- Build KDE with different kernel functions
- Code robust estimation methods
- Create anomaly detection systems
- Handle edge cases and numerical stability

## Key Concepts

- **Likelihood and Maximum Likelihood**: Understanding probabilistic inference
- **Bayesian vs Frequentist**: Different approaches to statistical modeling
- **Uncertainty Quantification**: Confidence intervals and credible regions
- **Model Selection**: AIC, BIC, and cross-validation for probabilistic models

## Mathematical Prerequisites

- Probability theory (distributions, Bayes' rule)
- Linear algebra (eigenvalues, matrix operations)
- Calculus (derivatives, optimization)
- Basic statistics (mean, variance, correlation)

## Prerequisites

- Completion of Linear Models & Classical ML module
- Comfort with probability and statistics
- Understanding of optimization algorithms

## Next Steps

After completing this module, you'll be ready for **Neural Networks & Deep Learning Foundations** where you'll learn about modern deep learning approaches. 