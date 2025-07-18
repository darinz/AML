# Classification Algorithms

## Overview

Classification is a fundamental supervised learning task where the goal is to predict categorical labels for new data points. This guide covers the most important classification algorithms: Logistic Regression, Support Vector Machines (SVM), and Naïve Bayes, along with their mathematical foundations and practical implementations.

## Key Concepts

### 1. What is Classification?

Classification involves predicting discrete class labels for input data. The algorithm learns a mapping from input features to output classes based on labeled training data.

### 2. Types of Classification

1. **Binary Classification**: Two classes (e.g., spam/not spam)
2. **Multiclass Classification**: More than two classes (e.g., digit recognition)
3. **Multi-label Classification**: Multiple labels per instance

### 3. Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC Curve**: True positive rate vs false positive rate
- **AUC**: Area under the ROC curve

## Logistic Regression

### Mathematical Foundation

Logistic regression models the probability of belonging to a class using the logistic function:

```math
P(y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}} = \frac{1}{1 + e^{-\mathbf{x}^T \boldsymbol{\beta}}}
```

The logistic function (sigmoid) transforms any real number to the range [0, 1]:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

### Cost Function

For binary classification, we use the log-likelihood (cross-entropy loss):

```math
J(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(h_{\boldsymbol{\beta}}(\mathbf{x}_i)) + (1 - y_i) \log(1 - h_{\boldsymbol{\beta}}(\mathbf{x}_i))]
```

Where $`h_{\boldsymbol{\beta}}(\mathbf{x}) = \sigma(\mathbf{x}^T \boldsymbol{\beta})`$ is the predicted probability.

### Gradient Descent Update

The gradient of the cost function is:

```math
\frac{\partial J}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (h_{\boldsymbol{\beta}}(\mathbf{x}_i) - y_i) x_{i,j}
```

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class LogisticRegression:
    """
    Logistic Regression implementation from scratch
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _compute_cost(self, X, y, coefficients, intercept):
        """Compute logistic regression cost"""
        m = X.shape[0]
        z = X @ coefficients + intercept
        h = self._sigmoid(z)
        
        # Avoid log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def _compute_gradients(self, X, y, coefficients, intercept):
        """Compute gradients for coefficients and intercept"""
        m = X.shape[0]
        z = X @ coefficients + intercept
        h = self._sigmoid(z)
        
        # Gradients
        grad_coefficients = (1/m) * X.T @ (h - y)
        grad_intercept = np.mean(h - y)
        
        return grad_coefficients, grad_intercept
    
    def fit(self, X, y):
        """Fit logistic regression using gradient descent"""
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
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X)
        z = X @ self.coefficients + self.intercept
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """Plot cost history during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost History During Training')
        plt.grid(True)
        plt.show()

def demonstrate_logistic_regression():
    """Demonstrate logistic regression"""
    print("=== Logistic Regression Demo ===")
    
    # Generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    lr = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test_scaled)
    y_proba = lr.predict_proba(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot cost history
    lr.plot_cost_history()
    
    # Visualize decision boundary
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Decision boundary
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, alpha=0.6)
    plt.title('Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return lr, X_train_scaled, y_train, X_test_scaled, y_test

# Run demonstration
lr_model, X_train, y_train, X_test, y_test = demonstrate_logistic_regression()
```

### Multiclass Logistic Regression

```python
class MulticlassLogisticRegression:
    """
    Multiclass Logistic Regression using One-vs-Rest strategy
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.classifiers = []
        self.classes = None
        
    def _sigmoid(self, z):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _fit_binary_classifier(self, X, y_binary):
        """Fit a binary classifier"""
        n_features = X.shape[1]
        coefficients = np.zeros(n_features)
        intercept = 0
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X @ coefficients + intercept
            h = self._sigmoid(z)
            
            # Gradients
            grad_coefficients = (1/len(X)) * X.T @ (h - y_binary)
            grad_intercept = np.mean(h - y_binary)
            
            # Update parameters
            coefficients -= self.learning_rate * grad_coefficients
            intercept -= self.learning_rate * grad_intercept
        
        return coefficients, intercept
    
    def fit(self, X, y):
        """Fit multiclass logistic regression"""
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        self.classifiers = []
        
        # Train one classifier per class (One-vs-Rest)
        for class_label in self.classes:
            # Create binary labels for this class
            y_binary = (y == class_label).astype(int)
            
            # Fit binary classifier
            coefficients, intercept = self._fit_binary_classifier(X, y_binary)
            self.classifiers.append((coefficients, intercept))
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        probabilities = []
        
        for coefficients, intercept in self.classifiers:
            z = X @ coefficients + intercept
            prob = self._sigmoid(z)
            probabilities.append(prob)
        
        # Normalize to get proper probabilities
        probabilities = np.column_stack(probabilities)
        row_sums = probabilities.sum(axis=1)
        probabilities = probabilities / row_sums[:, np.newaxis]
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

def demonstrate_multiclass_logistic():
    """Demonstrate multiclass logistic regression"""
    print("\n=== Multiclass Logistic Regression Demo ===")
    
    # Generate multiclass data
    X, y = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    mlr = MulticlassLogisticRegression(learning_rate=0.1, max_iterations=1000)
    mlr.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = mlr.predict(X_test_scaled)
    y_proba = mlr.predict_proba(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, alpha=0.6)
    plt.title('Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return mlr, X_train_scaled, y_train, X_test_scaled, y_test

# Run demonstration
mlr_model, X_train, y_train, X_test, y_test = demonstrate_multiclass_logistic()
```

## Support Vector Machines (SVM)

### Mathematical Foundation

SVM finds the optimal hyperplane that separates classes with maximum margin. The margin is the distance between the hyperplane and the nearest data points (support vectors).

For linearly separable data, the optimization problem is:

```math
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
```

Subject to:

```math
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
```

For non-separable data, we introduce slack variables $`\xi_i`$:

```math
\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
```

Subject to:

```math
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i
\xi_i \geq 0 \quad \forall i
```

### Kernel Trick

For non-linear decision boundaries, we use kernel functions:

```math
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
```

Common kernels:
- **Linear**: $`K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j`$
- **Polynomial**: $`K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d`$
- **RBF**: $`K(\mathbf{x}_i, \mathbf{x}_j) = e^{-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2}`$

### Implementation

```python
class SVM:
    """
    Support Vector Machine implementation using SMO algorithm
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma='scale', max_iterations=1000, tolerance=1e-3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
        
    def _linear_kernel(self, x1, x2):
        """Linear kernel"""
        return np.dot(x1, x2)
    
    def _rbf_kernel(self, x1, x2):
        """RBF kernel"""
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
    
    def _polynomial_kernel(self, x1, x2, degree=3):
        """Polynomial kernel"""
        return (self.gamma * np.dot(x1, x2) + 1) ** degree
    
    def _compute_kernel(self, x1, x2):
        """Compute kernel function"""
        if self.kernel == 'linear':
            return self._linear_kernel(x1, x2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(x1, x2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(x1, x2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X):
        """Compute kernel matrix"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._compute_kernel(X[i], X[j])
        
        return K
    
    def _smo_step(self, X, y, alphas, b, i, j):
        """Single SMO optimization step"""
        if i == j:
            return False
        
        alpha_i_old = alphas[i]
        alpha_j_old = alphas[j]
        
        # Compute errors
        Ei = self._predict_single(X[i], X, y, alphas, b) - y[i]
        Ej = self._predict_single(X[j], X, y, alphas, b) - y[j]
        
        # Compute bounds
        if y[i] != y[j]:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        
        if L == H:
            return False
        
        # Compute eta
        K_ii = self._compute_kernel(X[i], X[i])
        K_jj = self._compute_kernel(X[j], X[j])
        K_ij = self._compute_kernel(X[i], X[j])
        eta = K_ii + K_jj - 2 * K_ij
        
        if eta <= 0:
            return False
        
        # Update alpha_j
        alpha_j_new = alpha_j_old + y[j] * (Ei - Ej) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)
        
        if abs(alpha_j_new - alpha_j_old) < self.tolerance:
            return False
        
        # Update alpha_i
        alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)
        
        # Update b
        b1 = b - Ei - y[i] * (alpha_i_new - alpha_i_old) * K_ii - y[j] * (alpha_j_new - alpha_j_old) * K_ij
        b2 = b - Ej - y[i] * (alpha_i_new - alpha_i_old) * K_ij - y[j] * (alpha_j_new - alpha_j_old) * K_jj
        
        if 0 < alpha_i_new < self.C:
            b_new = b1
        elif 0 < alpha_j_new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        
        # Update alphas and b
        alphas[i] = alpha_i_new
        alphas[j] = alpha_j_new
        self.b = b_new
        
        return True
    
    def _predict_single(self, x, X, y, alphas, b):
        """Predict single point"""
        prediction = b
        for i in range(len(X)):
            prediction += alphas[i] * y[i] * self._compute_kernel(X[i], x)
        return prediction
    
    def fit(self, X, y):
        """Fit SVM using SMO algorithm"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        
        # Initialize alphas and b
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # SMO algorithm
        num_changed = 0
        examine_all = True
        
        while num_changed > 0 or examine_all:
            num_changed = 0
            
            if examine_all:
                for i in range(n_samples):
                    num_changed += self._smo_step(X, y, self.alphas, self.b, i, i)
            else:
                # Only examine non-bound alphas
                non_bound_indices = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
                for i in non_bound_indices:
                    num_changed += self._smo_step(X, y, self.alphas, self.b, i, i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        
        # Store support vectors
        support_vector_indices = np.where(self.alphas > 0)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.alphas = self.alphas[support_vector_indices]
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            prediction = self.b
            for i in range(len(self.support_vectors)):
                prediction += (self.alphas[i] * self.support_vector_labels[i] * 
                             self._compute_kernel(self.support_vectors[i], x))
            predictions.append(np.sign(prediction))
        
        return np.array(predictions)

def demonstrate_svm():
    """Demonstrate SVM"""
    print("\n=== Support Vector Machine Demo ===")
    
    # Generate data
    X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit SVM
    svm = SVM(C=1.0, kernel='rbf', gamma=0.1, max_iterations=1000)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Number of support vectors: {len(svm.support_vectors)}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Training data with support vectors
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
               c='red', marker='x', s=100, linewidths=3, label='Support Vectors')
    plt.title('Training Data with Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, alpha=0.6)
    plt.title('Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return svm, X_train_scaled, y_train, X_test_scaled, y_test

# Run demonstration
svm_model, X_train, y_train, X_test, y_test = demonstrate_svm()
```

## Naïve Bayes

### Mathematical Foundation

Naïve Bayes is based on Bayes' theorem with the "naïve" assumption of conditional independence:

```math
P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) P(y)}{P(\mathbf{x})}
```

Using the naïve assumption:

```math
P(\mathbf{x} | y) = \prod_{j=1}^{n} P(x_j | y)
```

Therefore:

```math
P(y | \mathbf{x}) \propto P(y) \prod_{j=1}^{n} P(x_j | y)
```

### Gaussian Naïve Bayes

For continuous features, we assume they follow a normal distribution:

```math
P(x_j | y) = \frac{1}{\sqrt{2\pi \sigma_{y,j}^2}} \exp\left(-\frac{(x_j - \mu_{y,j})^2}{2\sigma_{y,j}^2}\right)
```

### Implementation

```python
class GaussianNaiveBayes:
    """
    Gaussian Naïve Bayes implementation
    """
    
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
        
    def fit(self, X, y):
        """Fit Gaussian Naïve Bayes"""
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        
        # Compute parameters for each class
        for i, class_label in enumerate(self.classes):
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            
            # Class prior
            self.class_priors[i] = np.mean(class_mask)
            
            # Mean and variance for each feature
            self.means[i] = np.mean(X_class, axis=0)
            self.variances[i] = np.var(X_class, axis=0)
            
            # Add small value to avoid division by zero
            self.variances[i] = np.maximum(self.variances[i], 1e-9)
        
        return self
    
    def _gaussian_pdf(self, x, mean, variance):
        """Compute Gaussian probability density function"""
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Compute log probabilities to avoid numerical underflow
        log_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            # Log prior
            log_probs[:, i] = np.log(self.class_priors[i])
            
            # Log likelihood for each feature
            for j in range(X.shape[1]):
                log_probs[:, i] += np.log(self._gaussian_pdf(X[:, j], 
                                                           self.means[i, j], 
                                                           self.variances[i, j]))
        
        # Convert to probabilities
        # Subtract max for numerical stability
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_probs)
        probabilities = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

def demonstrate_naive_bayes():
    """Demonstrate Naïve Bayes"""
    print("\n=== Naïve Bayes Demo ===")
    
    # Generate data
    X, y = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Naïve Bayes
    nb = GaussianNaiveBayes()
    nb.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = nb.predict(X_test_scaled)
    y_proba = nb.predict_proba(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, alpha=0.6)
    plt.title('Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return nb, X_train_scaled, y_train, X_test_scaled, y_test

# Run demonstration
nb_model, X_train, y_train, X_test, y_test = demonstrate_naive_bayes()
```

## Model Comparison and Evaluation

### Comprehensive Evaluation

```python
def compare_classifiers():
    """Compare different classification algorithms"""
    print("\n=== Classifier Comparison Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(learning_rate=0.1, max_iterations=1000),
        'SVM (Linear)': SVM(C=1.0, kernel='linear', max_iterations=1000),
        'SVM (RBF)': SVM(C=1.0, kernel='rbf', gamma=0.1, max_iterations=1000),
        'Naïve Bayes': GaussianNaiveBayes()
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Fit classifier
        classifier.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Compute ROC curve and AUC (for binary classification)
        if hasattr(classifier, 'predict_proba'):
            y_proba = classifier.predict_proba(X_test_scaled)
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]  # Take probability of positive class
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
        else:
            fpr, tpr, auc_score = None, None, None
        
        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score,
            'predictions': y_pred
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        if auc_score is not None:
            print(f"  AUC: {auc_score:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 3, 1)
    classifier_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in classifier_names]
    bars = plt.bar(classifier_names, accuracies)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # ROC curves
    plt.subplot(2, 3, 2)
    for name in classifier_names:
        if results[name]['auc'] is not None:
            plt.plot(results[name]['fpr'], results[name]['tpr'], 
                    label=f'{name} (AUC = {results[name]["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrices
    for i, name in enumerate(classifier_names[:4]):
        plt.subplot(2, 3, i + 3)
        cm = results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
comparison_results = compare_classifiers()
```

## Practical Applications

### Spam Detection

```python
def spam_detection_demo():
    """Demonstrate spam detection using classification"""
    print("\n=== Spam Detection Demo ===")
    
    # Generate synthetic email data
    np.random.seed(42)
    n_emails = 1000
    
    # Features: word_count, exclamation_count, spam_words_count, sender_known
    word_count = np.random.poisson(50, n_emails)
    exclamation_count = np.random.poisson(2, n_emails)
    spam_words_count = np.random.poisson(1, n_emails)
    sender_known = np.random.binomial(1, 0.7, n_emails)
    
    # Create feature matrix
    X = np.column_stack([word_count, exclamation_count, spam_words_count, sender_known])
    
    # Generate labels (spam = 1, ham = 0)
    # Higher spam_words_count and exclamation_count increase spam probability
    spam_prob = 1 / (1 + np.exp(-(-2 + 0.1 * spam_words_count + 0.2 * exclamation_count - 0.5 * sender_known)))
    y = np.random.binomial(1, spam_prob)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(learning_rate=0.1, max_iterations=1000),
        'SVM': SVM(C=1.0, kernel='rbf', gamma=0.1, max_iterations=1000),
        'Naïve Bayes': GaussianNaiveBayes()
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate precision, recall, F1-score
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Metrics comparison
    plt.subplot(1, 3, 1)
    classifier_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(classifier_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in classifier_names]
        plt.bar(x + i * width, values, width, label=metric.capitalize())
    
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('Spam Detection Performance')
    plt.xticks(x + width * 1.5, classifier_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion matrices
    for i, name in enumerate(classifier_names):
        plt.subplot(1, 3, i + 2)
        cm = results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
spam_results = spam_detection_demo()
```

## Summary

Classification algorithms provide powerful tools for:

1. **Pattern Recognition**: Identifying patterns in data
2. **Decision Making**: Making predictions for new instances
3. **Feature Understanding**: Understanding which features are important
4. **Risk Assessment**: Evaluating probabilities of different outcomes

Key takeaways:

- **Logistic Regression**: Simple, interpretable, good baseline
- **SVM**: Effective for high-dimensional data, handles non-linear boundaries
- **Naïve Bayes**: Fast, works well with small datasets, handles missing data
- **Model Selection**: Depends on data characteristics and requirements
- **Evaluation**: Use multiple metrics, not just accuracy
- **Feature Engineering**: Often more important than algorithm choice

Understanding these algorithms provides a solid foundation for more advanced machine learning techniques and real-world applications. 