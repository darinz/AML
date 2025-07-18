"""
Linear Regression & Regularization Examples

This file contains comprehensive examples demonstrating:
- Linear regression from scratch
- Gradient descent optimization
- Ridge, Lasso, and Elastic Net regularization
- Feature engineering and polynomial regression
- Cross-validation and model selection
- Real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_boston, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

class LinearRegressionFromScratch:
    """Linear Regression implementation from scratch"""
    
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

class LinearRegressionGD:
    """Linear Regression using Gradient Descent"""
    
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

class RidgeRegression:
    """Ridge Regression implementation"""
    
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

class LassoRegression:
    """Lasso Regression implementation using coordinate descent"""
    
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

def demonstrate_basic_linear_regression():
    """Demonstrate basic linear regression"""
    print("=== Basic Linear Regression Demo ===")
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model using normal equation
    lr_normal = LinearRegressionFromScratch()
    lr_normal.fit(X_train, y_train)
    
    # Fit model using gradient descent
    lr_gd = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
    lr_gd.fit(X_train, y_train)
    
    # Make predictions
    y_pred_normal = lr_normal.predict(X_test)
    y_pred_gd = lr_gd.predict(X_test)
    
    # Evaluate
    mse_normal = mean_squared_error(y_test, y_pred_normal)
    r2_normal = r2_score(y_test, y_pred_normal)
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    r2_gd = r2_score(y_test, y_pred_gd)
    
    print(f"Normal Equation:")
    print(f"  MSE: {mse_normal:.2f}")
    print(f"  R²: {r2_normal:.3f}")
    print(f"  Intercept: {lr_normal.intercept:.3f}")
    print(f"  Coefficient: {lr_normal.coefficients[0]:.3f}")
    
    print(f"\nGradient Descent:")
    print(f"  MSE: {mse_gd:.2f}")
    print(f"  R²: {r2_gd:.3f}")
    print(f"  Intercept: {lr_gd.intercept:.3f}")
    print(f"  Coefficient: {lr_gd.coefficients[0]:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Cost history
    plt.subplot(1, 3, 1)
    lr_gd.plot_cost_history()
    
    # Predictions comparison
    plt.subplot(1, 3, 2)
    plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
    plt.scatter(X_test, y_pred_normal, alpha=0.6, label='Normal Equation')
    plt.scatter(X_test, y_pred_gd, alpha=0.6, label='Gradient Descent')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression Results')
    plt.legend()
    
    # Residuals
    plt.subplot(1, 3, 3)
    residuals_normal = y_test - y_pred_normal
    residuals_gd = y_test - y_pred_gd
    plt.scatter(y_pred_normal, residuals_normal, alpha=0.6, label='Normal Equation')
    plt.scatter(y_pred_gd, residuals_gd, alpha=0.6, label='Gradient Descent')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return lr_normal, lr_gd, X_train, y_train, X_test, y_test

def demonstrate_regularization():
    """Demonstrate regularization techniques"""
    print("\n=== Regularization Demo ===")
    
    # Generate data with some noise
    X, y = make_regression(n_samples=100, n_features=20, noise=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different regularization strengths
    alpha_values = [0, 0.1, 1, 10, 100]
    results = {}
    
    for alpha in alpha_values:
        # Ridge regression
        ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        
        # Lasso regression
        lasso = LassoRegression(alpha=alpha)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        
        # Evaluate
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        mse_lasso = mean_squared_error(y_test, y_pred_lasso)
        r2_lasso = r2_score(y_test, y_pred_lasso)
        
        results[alpha] = {
            'ridge': {'mse': mse_ridge, 'r2': r2_ridge, 'coefficients': ridge.coefficients.copy()},
            'lasso': {'mse': mse_lasso, 'r2': r2_lasso, 'coefficients': lasso.coefficients.copy()}
        }
        
        print(f"Alpha={alpha}:")
        print(f"  Ridge - MSE: {mse_ridge:.2f}, R²: {r2_ridge:.3f}")
        print(f"  Lasso - MSE: {mse_lasso:.2f}, R²: {r2_lasso:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # MSE comparison
    plt.subplot(2, 3, 1)
    alphas = list(results.keys())
    mse_ridge = [results[alpha]['ridge']['mse'] for alpha in alphas]
    mse_lasso = [results[alpha]['lasso']['mse'] for alpha in alphas]
    plt.semilogx(alphas, mse_ridge, 'bo-', label='Ridge')
    plt.semilogx(alphas, mse_lasso, 'ro-', label='Lasso')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Regularization Strength')
    plt.legend()
    plt.grid(True)
    
    # R² comparison
    plt.subplot(2, 3, 2)
    r2_ridge = [results[alpha]['ridge']['r2'] for alpha in alphas]
    r2_lasso = [results[alpha]['lasso']['r2'] for alpha in alphas]
    plt.semilogx(alphas, r2_ridge, 'bo-', label='Ridge')
    plt.semilogx(alphas, r2_lasso, 'ro-', label='Lasso')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('R² Score')
    plt.title('R² vs Regularization Strength')
    plt.legend()
    plt.grid(True)
    
    # Coefficient magnitudes - Ridge
    plt.subplot(2, 3, 3)
    for alpha in alphas:
        plt.plot(np.abs(results[alpha]['ridge']['coefficients']), 
                label=f'α={alpha}', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('|Coefficient|')
    plt.title('Ridge Coefficient Magnitudes')
    plt.legend()
    plt.grid(True)
    
    # Coefficient magnitudes - Lasso
    plt.subplot(2, 3, 4)
    for alpha in alphas:
        plt.plot(np.abs(results[alpha]['lasso']['coefficients']), 
                label=f'α={alpha}', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('|Coefficient|')
    plt.title('Lasso Coefficient Magnitudes')
    plt.legend()
    plt.grid(True)
    
    # Sparsity comparison
    plt.subplot(2, 3, 5)
    n_nonzero_ridge = [np.sum(results[alpha]['ridge']['coefficients'] != 0) for alpha in alphas]
    n_nonzero_lasso = [np.sum(results[alpha]['lasso']['coefficients'] != 0) for alpha in alphas]
    plt.semilogx(alphas, n_nonzero_ridge, 'bo-', label='Ridge')
    plt.semilogx(alphas, n_nonzero_lasso, 'ro-', label='Lasso')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Sparsity vs Regularization Strength')
    plt.legend()
    plt.grid(True)
    
    # Best model predictions
    plt.subplot(2, 3, 6)
    best_alpha = min(results.keys(), key=lambda a: results[a]['ridge']['mse'])
    best_ridge = RidgeRegression(alpha=best_alpha)
    best_ridge.fit(X_train_scaled, y_train)
    y_pred_best = best_ridge.predict(X_test_scaled)
    
    plt.scatter(y_test, y_pred_best, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Best Ridge Model (α={best_alpha})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_polynomial_regression():
    """Demonstrate polynomial regression"""
    print("\n=== Polynomial Regression Demo ===")
    
    # Generate non-linear data
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X**2 - 3 * X + 1 + np.random.normal(0, 0.5, (100, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 4, 5]
    results = {}
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Fit linear regression
        lr = LinearRegressionFromScratch()
        lr.fit(X_train_poly, y_train)
        y_pred = lr.predict(X_test_poly)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[degree] = {
            'mse': mse,
            'r2': r2,
            'model': lr,
            'transformer': poly
        }
        
        print(f"Degree={degree}: MSE={mse:.4f}, R²={r2:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # MSE vs degree
    plt.subplot(1, 3, 1)
    degrees_list = list(results.keys())
    mses = [results[degree]['mse'] for degree in degrees_list]
    plt.plot(degrees_list, mses, 'bo-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Polynomial Degree')
    plt.grid(True)
    
    # R² vs degree
    plt.subplot(1, 3, 2)
    r2_scores = [results[degree]['r2'] for degree in degrees_list]
    plt.plot(degrees_list, r2_scores, 'ro-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('R² vs Polynomial Degree')
    plt.grid(True)
    
    # Best fit
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
        'Linear Regression': LinearRegressionFromScratch(),
        'Ridge (α=1)': RidgeRegression(alpha=1.0),
        'Ridge (α=10)': RidgeRegression(alpha=10.0),
        'Lasso (α=0.1)': LassoRegression(alpha=0.1),
        'Lasso (α=1)': LassoRegression(alpha=1.0)
    }
    
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
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
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Mean MSE comparison
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
    
    # Individual CV scores
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

def demonstrate_real_world_application():
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
        'Linear Regression': LinearRegressionFromScratch(),
        'Ridge Regression': RidgeRegression(alpha=1.0),
        'Lasso Regression': LassoRegression(alpha=0.1),
        'Gradient Descent': LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'predictions': y_pred
        }
        
        print(f"{name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAE: {mae:.2f}")
        if hasattr(model, 'coefficients'):
            print(f"  Coefficients: {model.coefficients}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Actual vs predicted
    plt.subplot(1, 3, 1)
    for name in results.keys():
        plt.scatter(y_test, results[name]['predictions'], alpha=0.6, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.grid(True)
    
    # Residuals
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
    
    # Model comparison
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

if __name__ == "__main__":
    # Run all demonstrations
    print("Linear Regression & Regularization Examples")
    print("=" * 50)
    
    # Basic linear regression
    lr_normal, lr_gd, X_train, y_train, X_test, y_test = demonstrate_basic_linear_regression()
    
    # Regularization
    reg_results = demonstrate_regularization()
    
    # Polynomial regression
    poly_results = demonstrate_polynomial_regression()
    
    # Cross-validation
    cv_results = demonstrate_cross_validation()
    
    # Real-world application
    house_results = demonstrate_real_world_application()
    
    print("\nAll demonstrations completed!") 