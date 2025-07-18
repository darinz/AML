# Linear Regression & Regularization

## Overview

Linear regression is one of the most fundamental and widely used machine learning algorithms. It models the relationship between a dependent variable and one or more independent variables using a linear function. This guide covers the mathematical foundations, implementation from scratch, and various regularization techniques.

## Key Concepts

### 1. What is Linear Regression?

Linear regression assumes a linear relationship between the input features and the target variable:

```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
```

Where:
- $`y`$ is the target variable
- $`x_i`$ are the input features
- $`\beta_i`$ are the model parameters (coefficients)
- $`\beta_0`$ is the intercept (bias term)
- $`\epsilon`$ is the error term (noise)

### 2. Assumptions of Linear Regression

1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

## Mathematical Foundation

### Ordinary Least Squares (OLS)

The goal is to minimize the sum of squared residuals:

```math
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
```

Where:
- $`\text{RSS}`$ is the Residual Sum of Squares
- $`y_i`$ is the actual value
- $`\hat{y}_i`$ is the predicted value
- $`\mathbf{x}_i`$ is the feature vector for observation i
- $`\boldsymbol{\beta}`$ is the coefficient vector

### Matrix Formulation

In matrix notation:

```math
\text{RSS} = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
```

The optimal coefficients are found by setting the derivative to zero:

```math
\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0
```

Solving for $`\boldsymbol{\beta}`$:

```math
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
```

This is the **normal equation** for linear regression.

## Implementation from Scratch

### Basic Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    """
    Linear Regression implementation from scratch
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """Fit linear regression using normal equation"""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        if self.fit_intercept:
            # Add intercept term (column of ones)
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: β = (X^T X)^(-1) X^T y
        X_transpose = X.T
        X_transpose_X = X_transpose @ X
        X_transpose_y = X_transpose @ y
        
        # Solve for coefficients
        try:
            coefficients = np.linalg.solve(X_transpose_X, X_transpose_y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            coefficients = np.linalg.pinv(X_transpose_X) @ X_transpose_y
        
        if self.fit_intercept:
            self.intercept = coefficients[0][0]
            self.coefficients = coefficients[1:].flatten()
        else:
            self.intercept = 0
            self.coefficients = coefficients.flatten()
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        if self.fit_intercept:
            return X @ self.coefficients + self.intercept
        else:
            return X @ self.coefficients
    
    def get_params(self):
        """Get model parameters"""
        return {
            'coefficients': self.coefficients,
            'intercept': self.intercept
        }

# Example usage
def demonstrate_linear_regression():
    """Demonstrate basic linear regression"""
    print("=== Linear Regression Demo ===")
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Intercept: {lr.intercept:.3f}")
    print(f"Coefficient: {lr.coefficients[0]:.3f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
    plt.scatter(X_test, y_pred, alpha=0.6, label='Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.show()
    
    return lr, X_train, y_train, X_test, y_test

# Run demonstration
lr_model, X_train, y_train, X_test, y_test = demonstrate_linear_regression()
```

### Gradient Descent Implementation

```python
class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
        
    def _compute_cost(self, X, y, coefficients, intercept):
        """Compute mean squared error cost"""
        predictions = X @ coefficients + intercept
        cost = np.mean((y - predictions) ** 2)
        return cost
    
    def _compute_gradients(self, X, y, coefficients, intercept):
        """Compute gradients for coefficients and intercept"""
        predictions = X @ coefficients + intercept
        errors = y - predictions
        
        # Gradient for coefficients
        grad_coefficients = -2 * np.mean(X * errors.reshape(-1, 1), axis=0)
        
        # Gradient for intercept
        grad_intercept = -2 * np.mean(errors)
        
        return grad_coefficients, grad_intercept
    
    def fit(self, X, y):
        """Fit model using gradient descent"""
        X = np.array(X)
        y = np.array(y)
        
        # Initialize parameters
        n_features = X.shape[1]
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X, y, self.coefficients, self.intercept)
            self.cost_history.append(cost)
            
            # Compute gradients
            grad_coefficients, grad_intercept = self._compute_gradients(
                X, y, self.coefficients, self.intercept
            )
            
            # Update parameters
            self.coefficients -= self.learning_rate * grad_coefficients
            self.intercept -= self.learning_rate * grad_intercept
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.coefficients + self.intercept
    
    def plot_cost_history(self):
        """Plot cost history during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Cost History During Training')
        plt.grid(True)
        plt.show()

# Example usage
def demonstrate_gradient_descent():
    """Demonstrate gradient descent for linear regression"""
    print("\n=== Gradient Descent Demo ===")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=2, noise=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model using gradient descent
    lr_gd = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
    lr_gd.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_gd.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Intercept: {lr_gd.intercept:.3f}")
    print(f"Coefficients: {lr_gd.coefficients}")
    
    # Plot cost history
    lr_gd.plot_cost_history()
    
    return lr_gd, X_train_scaled, y_train, X_test_scaled, y_test

# Run demonstration
lr_gd_model, X_train_scaled, y_train, X_test_scaled, y_test = demonstrate_gradient_descent()
```

## Regularization Techniques

### Ridge Regression (L2 Regularization)

Ridge regression adds L2 penalty to the cost function:

```math
\text{Cost} = \text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2
```

Where $`\lambda`$ is the regularization strength.

```python
class RidgeRegression:
    """
    Ridge Regression implementation
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """Fit ridge regression"""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Ridge regression solution: β = (X^T X + λI)^(-1) X^T y
        X_transpose = X.T
        X_transpose_X = X_transpose @ X
        
        # Add regularization term
        n_features = X_transpose_X.shape[0]
        regularization_matrix = self.alpha * np.eye(n_features)
        if self.fit_intercept:
            # Don't regularize intercept
            regularization_matrix[0, 0] = 0
        
        X_transpose_X_regularized = X_transpose_X + regularization_matrix
        X_transpose_y = X_transpose @ y
        
        # Solve for coefficients
        coefficients = np.linalg.solve(X_transpose_X_regularized, X_transpose_y)
        
        if self.fit_intercept:
            self.intercept = coefficients[0][0]
            self.coefficients = coefficients[1:].flatten()
        else:
            self.intercept = 0
            self.coefficients = coefficients.flatten()
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        if self.fit_intercept:
            return X @ self.coefficients + self.intercept
        else:
            return X @ self.coefficients

def demonstrate_ridge_regression():
    """Demonstrate ridge regression"""
    print("\n=== Ridge Regression Demo ===")
    
    # Generate data with some noise
    X, y = make_regression(n_samples=100, n_features=20, noise=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different alpha values
    alpha_values = [0, 0.1, 1, 10, 100]
    results = {}
    
    for alpha in alpha_values:
        ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[alpha] = {
            'mse': mse,
            'r2': r2,
            'coefficients': ridge.coefficients.copy()
        }
        
        print(f"Alpha={alpha}: MSE={mse:.2f}, R²={r2:.3f}")
    
    # Visualize coefficient shrinkage
    plt.figure(figsize=(12, 5))
    
    # Plot MSE vs alpha
    plt.subplot(1, 2, 1)
    alphas = list(results.keys())
    mses = [results[alpha]['mse'] for alpha in alphas]
    plt.semilogx(alphas, mses, 'bo-')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Regularization Strength')
    plt.grid(True)
    
    # Plot coefficient magnitudes
    plt.subplot(1, 2, 2)
    for alpha in alphas:
        plt.plot(np.abs(results[alpha]['coefficients']), 
                label=f'α={alpha}', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('|Coefficient|')
    plt.title('Coefficient Magnitudes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
ridge_results = demonstrate_ridge_regression()
```

### Lasso Regression (L1 Regularization)

Lasso regression adds L1 penalty to the cost function:

```math
\text{Cost} = \text{RSS} + \lambda \sum_{j=1}^{p} |\beta_j|
```

This tends to produce sparse solutions (many coefficients become exactly zero).

```python
class LassoRegression:
    """
    Lasso Regression implementation using coordinate descent
    """
    
    def __init__(self, alpha=1.0, max_iterations=1000, tolerance=1e-6):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        
    def _soft_threshold(self, z, gamma):
        """Soft thresholding operator"""
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0
    
    def fit(self, X, y):
        """Fit lasso regression using coordinate descent"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Center the data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Initialize coefficients
        self.coefficients = np.zeros(n_features)
        self.intercept = y_mean
        
        # Coordinate descent
        for iteration in range(self.max_iterations):
            old_coefficients = self.coefficients.copy()
            
            for j in range(n_features):
                # Compute partial residual
                r = y_centered - X_centered @ self.coefficients + self.coefficients[j] * X_centered[:, j]
                
                # Compute correlation
                correlation = X_centered[:, j] @ r
                
                # Update coefficient using soft thresholding
                self.coefficients[j] = self._soft_threshold(correlation, self.alpha * n_samples) / (X_centered[:, j] @ X_centered[:, j])
            
            # Check convergence
            if np.max(np.abs(self.coefficients - old_coefficients)) < self.tolerance:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.coefficients + self.intercept

def demonstrate_lasso_regression():
    """Demonstrate lasso regression"""
    print("\n=== Lasso Regression Demo ===")
    
    # Generate data with some irrelevant features
    X, y = make_regression(n_samples=100, n_features=20, noise=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different alpha values
    alpha_values = [0.01, 0.1, 1, 10, 100]
    results = {}
    
    for alpha in alpha_values:
        lasso = LassoRegression(alpha=alpha)
        lasso.fit(X_train_scaled, y_train)
        y_pred = lasso.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_nonzero = np.sum(lasso.coefficients != 0)
        
        results[alpha] = {
            'mse': mse,
            'r2': r2,
            'coefficients': lasso.coefficients.copy(),
            'n_nonzero': n_nonzero
        }
        
        print(f"Alpha={alpha}: MSE={mse:.2f}, R²={r2:.3f}, Non-zero coefficients={n_nonzero}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot MSE vs alpha
    plt.subplot(1, 3, 1)
    alphas = list(results.keys())
    mses = [results[alpha]['mse'] for alpha in alphas]
    plt.semilogx(alphas, mses, 'ro-')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Regularization Strength')
    plt.grid(True)
    
    # Plot coefficient magnitudes
    plt.subplot(1, 3, 2)
    for alpha in alphas:
        plt.plot(np.abs(results[alpha]['coefficients']), 
                label=f'α={alpha}', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('|Coefficient|')
    plt.title('Coefficient Magnitudes')
    plt.legend()
    plt.grid(True)
    
    # Plot number of non-zero coefficients
    plt.subplot(1, 3, 3)
    n_nonzeros = [results[alpha]['n_nonzero'] for alpha in alphas]
    plt.semilogx(alphas, n_nonzeros, 'go-')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Sparsity vs Regularization Strength')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
lasso_results = demonstrate_lasso_regression()
```

### Elastic Net

Elastic Net combines L1 and L2 regularization:

```math
\text{Cost} = \text{RSS} + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2
```

```python
class ElasticNet:
    """
    Elastic Net implementation
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iterations=1000, tolerance=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # l1_ratio = λ1 / (λ1 + λ2)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """Fit elastic net regression"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Center the data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Initialize coefficients
        self.coefficients = np.zeros(n_features)
        self.intercept = y_mean
        
        # Elastic net parameters
        lambda1 = self.alpha * self.l1_ratio
        lambda2 = self.alpha * (1 - self.l1_ratio)
        
        # Coordinate descent
        for iteration in range(self.max_iterations):
            old_coefficients = self.coefficients.copy()
            
            for j in range(n_features):
                # Compute partial residual
                r = y_centered - X_centered @ self.coefficients + self.coefficients[j] * X_centered[:, j]
                
                # Compute correlation
                correlation = X_centered[:, j] @ r
                
                # Update coefficient with elastic net penalty
                denominator = X_centered[:, j] @ X_centered[:, j] + lambda2 * n_samples
                self.coefficients[j] = self._soft_threshold(correlation, lambda1 * n_samples) / denominator
            
            # Check convergence
            if np.max(np.abs(self.coefficients - old_coefficients)) < self.tolerance:
                break
        
        return self
    
    def _soft_threshold(self, z, gamma):
        """Soft thresholding operator"""
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.coefficients + self.intercept

def demonstrate_elastic_net():
    """Demonstrate elastic net regression"""
    print("\n=== Elastic Net Demo ===")
    
    # Generate data
    X, y = make_regression(n_samples=100, n_features=20, noise=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different l1_ratio values
    l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    for l1_ratio in l1_ratios:
        elastic_net = ElasticNet(alpha=1.0, l1_ratio=l1_ratio)
        elastic_net.fit(X_train_scaled, y_train)
        y_pred = elastic_net.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_nonzero = np.sum(elastic_net.coefficients != 0)
        
        results[l1_ratio] = {
            'mse': mse,
            'r2': r2,
            'coefficients': elastic_net.coefficients.copy(),
            'n_nonzero': n_nonzero
        }
        
        print(f"L1_ratio={l1_ratio}: MSE={mse:.2f}, R²={r2:.3f}, Non-zero coefficients={n_nonzero}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot MSE vs l1_ratio
    plt.subplot(1, 3, 1)
    l1_ratios_list = list(results.keys())
    mses = [results[l1_ratio]['mse'] for l1_ratio in l1_ratios_list]
    plt.plot(l1_ratios_list, mses, 'mo-')
    plt.xlabel('L1 Ratio')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs L1 Ratio')
    plt.grid(True)
    
    # Plot coefficient magnitudes
    plt.subplot(1, 3, 2)
    for l1_ratio in l1_ratios_list:
        plt.plot(np.abs(results[l1_ratio]['coefficients']), 
                label=f'L1_ratio={l1_ratio}', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('|Coefficient|')
    plt.title('Coefficient Magnitudes')
    plt.legend()
    plt.grid(True)
    
    # Plot number of non-zero coefficients
    plt.subplot(1, 3, 3)
    n_nonzeros = [results[l1_ratio]['n_nonzero'] for l1_ratio in l1_ratios_list]
    plt.plot(l1_ratios_list, n_nonzeros, 'co-')
    plt.xlabel('L1 Ratio')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Sparsity vs L1 Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
elastic_net_results = demonstrate_elastic_net()
```

## Feature Engineering

### Polynomial Features

```python
def create_polynomial_features(X, degree=2):
    """Create polynomial features"""
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    return X_poly, poly

def demonstrate_polynomial_regression():
    """Demonstrate polynomial regression"""
    print("\n=== Polynomial Regression Demo ===")
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X**2 - 3 * X + 1 + np.random.normal(0, 0.5, (100, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 4, 5]
    results = {}
    
    for degree in degrees:
        # Create polynomial features
        X_train_poly, poly_transformer = create_polynomial_features(X_train, degree)
        X_test_poly = poly_transformer.transform(X_test)
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_train_poly, y_train)
        y_pred = lr.predict(X_test_poly)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[degree] = {
            'mse': mse,
            'r2': r2,
            'model': lr,
            'transformer': poly_transformer
        }
        
        print(f"Degree={degree}: MSE={mse:.4f}, R²={r2:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot MSE vs degree
    plt.subplot(1, 3, 1)
    degrees_list = list(results.keys())
    mses = [results[degree]['mse'] for degree in degrees_list]
    plt.plot(degrees_list, mses, 'bo-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Polynomial Degree')
    plt.grid(True)
    
    # Plot R² vs degree
    plt.subplot(1, 3, 2)
    r2_scores = [results[degree]['r2'] for degree in degrees_list]
    plt.plot(degrees_list, r2_scores, 'ro-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('R² vs Polynomial Degree')
    plt.grid(True)
    
    # Plot best fit
    plt.subplot(1, 3, 3)
    plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
    
    # Find best degree
    best_degree = min(results.keys(), key=lambda d: results[d]['mse'])
    best_model = results[best_degree]['model']
    best_transformer = results[best_degree]['transformer']
    
    # Plot predictions
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_plot_poly = best_transformer.transform(X_plot)
    y_plot_pred = best_model.predict(X_plot_poly)
    
    plt.plot(X_plot, y_plot_pred, 'r-', label=f'Polynomial (degree={best_degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Best Polynomial Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
poly_results = demonstrate_polynomial_regression()
```

## Model Evaluation

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

def evaluate_models_cv(X, y, models, cv=5):
    """Evaluate multiple models using cross-validation"""
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mse_scores = -cv_scores  # Convert back to positive MSE
        
        results[name] = {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'mean_rmse': np.sqrt(np.mean(mse_scores)),
            'cv_scores': mse_scores
        }
        
        print(f"{name}:")
        print(f"  Mean MSE: {results[name]['mean_mse']:.4f} (+/- {results[name]['std_mse']*2:.4f})")
        print(f"  Mean RMSE: {results[name]['mean_rmse']:.4f}")
    
    return results

def demonstrate_cross_validation():
    """Demonstrate cross-validation for model selection"""
    print("\n=== Cross-Validation Demo ===")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1)': RidgeRegression(alpha=1.0),
        'Ridge (α=10)': RidgeRegression(alpha=10.0),
        'Lasso (α=0.1)': LassoRegression(alpha=0.1),
        'Lasso (α=1)': LassoRegression(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    # Evaluate models
    results = evaluate_models_cv(X_scaled, y, models, cv=5)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot mean MSE
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    mean_mses = [results[name]['mean_mse'] for name in model_names]
    std_mses = [results[name]['std_mse'] for name in model_names]
    
    bars = plt.bar(model_names, mean_mses, yerr=std_mses, capsize=5)
    plt.xlabel('Model')
    plt.ylabel('Mean MSE')
    plt.title('Cross-Validation Results')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot individual CV scores
    plt.subplot(1, 2, 2)
    for name in model_names:
        plt.plot(results[name]['cv_scores'], 'o-', label=name, alpha=0.7)
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.title('Individual CV Scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
cv_results = demonstrate_cross_validation()
```

## Practical Applications

### House Price Prediction

```python
def house_price_prediction_demo():
    """Demonstrate house price prediction"""
    print("\n=== House Price Prediction Demo ===")
    
    # Generate synthetic house data
    np.random.seed(42)
    n_houses = 1000
    
    # Features: square_feet, bedrooms, bathrooms, age, distance_to_city
    square_feet = np.random.normal(2000, 500, n_houses)
    bedrooms = np.random.poisson(3, n_houses)
    bathrooms = np.random.poisson(2, n_houses)
    age = np.random.exponential(20, n_houses)
    distance_to_city = np.random.exponential(10, n_houses)
    
    # Create feature matrix
    X = np.column_stack([square_feet, bedrooms, bathrooms, age, distance_to_city])
    
    # Generate target (house prices)
    # Price = 100 + 0.1*sqft + 20*bedrooms + 30*bathrooms - 2*age - 5*distance + noise
    y = (100 + 0.1*square_feet + 20*bedrooms + 30*bathrooms - 2*age - 5*distance_to_city + 
         np.random.normal(0, 20, n_houses))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': RidgeRegression(alpha=1.0),
        'Lasso Regression': LassoRegression(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'r2': r2,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"{name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.3f}")
        if hasattr(model, 'coefficients'):
            print(f"  Coefficients: {model.coefficients}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot actual vs predicted
    plt.subplot(1, 3, 1)
    for name in results.keys():
        plt.scatter(y_test, results[name]['predictions'], alpha=0.6, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals
    plt.subplot(1, 3, 2)
    for name in results.keys():
        residuals = y_test - results[name]['predictions']
        plt.scatter(results[name]['predictions'], residuals, alpha=0.6, label=name)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True)
    
    # Plot model comparison
    plt.subplot(1, 3, 3)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    bars = plt.bar(model_names, r2_scores)
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
house_results = house_price_prediction_demo()
```

## Summary

Linear regression and regularization techniques provide powerful tools for:

1. **Understanding Relationships**: Modeling linear relationships between variables
2. **Prediction**: Making accurate predictions on new data
3. **Feature Selection**: Identifying important features (Lasso)
4. **Overfitting Prevention**: Regularization techniques
5. **Interpretability**: Understanding feature importance

Key takeaways:

- **OLS** is the foundation but can overfit with many features
- **Ridge regression** helps with multicollinearity and overfitting
- **Lasso regression** performs feature selection by creating sparse solutions
- **Elastic Net** combines benefits of both L1 and L2 regularization
- **Cross-validation** is essential for model selection and evaluation
- **Feature engineering** can capture non-linear relationships

Understanding these techniques provides a solid foundation for more advanced machine learning algorithms and real-world applications. 