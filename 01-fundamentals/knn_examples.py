"""
K-Nearest Neighbors (K-NN) Implementation and Examples

This file contains complete implementations of K-NN for both classification
and regression, along with various distance metrics and practical examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implementation from scratch
    """
    
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN Classifier
        
        Parameters:
        k (int): Number of neighbors to consider
        distance_metric (str): Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Store training data (KNN doesn't actually "train")
        
        Parameters:
        X (array): Training features
        y (array): Training labels
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm_x1 * norm_x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, x):
        """
        Find k nearest neighbors for a given point
        """
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = [self.y_train[i] for _, i in distances[:self.k]]
        return neighbors
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            # Majority vote
            most_common = Counter(neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        """
        probabilities = []
        for x in X:
            neighbors = self._get_neighbors(x)
            # Count occurrences of each class
            class_counts = Counter(neighbors)
            total = len(neighbors)
            
            # Calculate probabilities
            proba = {cls: count/total for cls, count in class_counts.items()}
            probabilities.append(proba)
        return probabilities

class KNNRegressor:
    """
    K-Nearest Neighbors Regressor implementation from scratch
    """
    
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
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
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

def demonstrate_knn_classification():
    """
    Demonstrate KNN classification with synthetic data
    """
    print("=== K-NN Classification Demo ===")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_classes=3, 
        n_clusters_per_class=1,
        n_redundant=0,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train KNN classifier
    knn = KNNClassifier(k=5, distance_metric='euclidean')
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Classification Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Test predictions
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.6)
    plt.title('Test Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return knn, X_train, y_train, X_test, y_test

def demonstrate_knn_regression():
    """
    Demonstrate KNN regression with synthetic data
    """
    print("\n=== K-NN Regression Demo ===")
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000, 
        n_features=2, 
        n_targets=1, 
        noise=0.1, 
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train KNN regressor
    knn_reg = KNNRegressor(k=5, distance_metric='euclidean')
    knn_reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_reg.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"KNN Regression MSE: {mse:.3f}")
    print(f"KNN Regression R²: {r2:.3f}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Test predictions
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.6)
    plt.title('Test Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return knn_reg, X_train, y_train, X_test, y_test

def demonstrate_distance_metrics():
    """
    Compare different distance metrics
    """
    print("\n=== Distance Metrics Comparison ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test different distance metrics
    metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}
    
    for metric in metrics:
        knn = KNNClassifier(k=5, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[metric] = accuracy
        print(f"{metric.capitalize()} distance accuracy: {accuracy:.3f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    metrics_list = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(metrics_list, accuracies, color=['blue', 'green', 'red'])
    plt.title('K-NN Accuracy by Distance Metric')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.show()
    
    return results

def demonstrate_feature_scaling():
    """
    Demonstrate the importance of feature scaling in K-NN
    """
    print("\n=== Feature Scaling Demo ===")
    
    # Create data with different scales
    np.random.seed(42)
    n_samples = 1000
    
    # Feature 1: small scale (0-1)
    feature1 = np.random.uniform(0, 1, n_samples)
    
    # Feature 2: large scale (0-1000)
    feature2 = np.random.uniform(0, 1000, n_samples)
    
    # Create target based on feature1 (feature2 is noise)
    y = (feature1 > 0.5).astype(int)
    
    X = np.column_stack([feature1, feature2])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test without scaling
    knn_unscaled = KNNClassifier(k=5)
    knn_unscaled.fit(X_train, y_train)
    y_pred_unscaled = knn_unscaled.predict(X_test)
    accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)
    
    # Test with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn_scaled = KNNClassifier(k=5)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)
    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
    
    print(f"Accuracy without scaling: {accuracy_unscaled:.3f}")
    print(f"Accuracy with scaling: {accuracy_scaled:.3f}")
    
    # Visualize the difference
    plt.figure(figsize=(12, 5))
    
    # Unscaled data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6)
    plt.title('Unscaled Data')
    plt.xlabel('Feature 1 (0-1)')
    plt.ylabel('Feature 2 (0-1000)')
    
    # Scaled data
    plt.subplot(1, 2, 2)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Scaled Data')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    
    plt.tight_layout()
    plt.show()
    
    return accuracy_unscaled, accuracy_scaled

def demonstrate_hyperparameter_tuning():
    """
    Demonstrate hyperparameter tuning for K-NN
    """
    print("\n=== Hyperparameter Tuning Demo ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
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
    
    # Find best K
    best_k = k_values[np.argmax(test_scores)]
    print(f"Best K value: {best_k}")
    print(f"Best test accuracy: {max(test_scores):.3f}")
    
    # Plot bias-variance trade-off
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_scores, label='Training Accuracy', marker='o')
    plt.plot(k_values, test_scores, label='Test Accuracy', marker='s')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('Bias-Variance Trade-off in K-NN')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_k, max(test_scores)

def demonstrate_cross_validation():
    """
    Demonstrate cross-validation for K-NN
    """
    print("\n=== Cross-Validation Demo ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Test different K values with cross-validation
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    cv_scores = []
    
    for k in k_values:
        knn = KNNClassifier(k=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"K={k}: CV Accuracy = {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Find best K
    best_k_cv = k_values[np.argmax(cv_scores)]
    print(f"\nBest K from cross-validation: {best_k_cv}")
    print(f"Best CV accuracy: {max(cv_scores):.3f}")
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'bo-')
    plt.axvline(x=best_k_cv, color='red', linestyle='--', label=f'Best K = {best_k_cv}')
    plt.xlabel('K Value')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('K-NN Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_k_cv, max(cv_scores)

def demonstrate_weighted_knn():
    """
    Demonstrate weighted K-NN where closer neighbors have more influence
    """
    print("\n=== Weighted K-NN Demo ===")
    
    class WeightedKNNClassifier(KNNClassifier):
        def predict(self, X):
            predictions = []
            for x in X:
                # Calculate distances
                distances = []
                for i, x_train in enumerate(self.X_train):
                    dist = self._calculate_distance(x, x_train)
                    distances.append((dist, i))
                
                # Sort by distance and get k nearest
                distances.sort(key=lambda x: x[0])
                k_nearest = distances[:self.k]
                
                # Weighted voting (inverse distance)
                weighted_votes = {}
                for dist, idx in k_nearest:
                    label = self.y_train[idx]
                    weight = 1 / (dist + 1e-8)  # Add small epsilon to avoid division by zero
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                
                # Predict class with highest weighted vote
                prediction = max(weighted_votes, key=weighted_votes.get)
                predictions.append(prediction)
            
            return np.array(predictions)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=500, 
        n_features=2, 
        n_classes=2, 
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Compare regular vs weighted K-NN
    regular_knn = KNNClassifier(k=5)
    weighted_knn = WeightedKNNClassifier(k=5)
    
    regular_knn.fit(X_train, y_train)
    weighted_knn.fit(X_train, y_train)
    
    regular_accuracy = accuracy_score(y_test, regular_knn.predict(X_test))
    weighted_accuracy = accuracy_score(y_test, weighted_knn.predict(X_test))
    
    print(f"Regular K-NN accuracy: {regular_accuracy:.3f}")
    print(f"Weighted K-NN accuracy: {weighted_accuracy:.3f}")
    
    # Visualize decision boundaries
    def plot_decision_boundary(knn, X, y, title):
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Predict on mesh grid
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_decision_boundary(regular_knn, X_train, y_train, 'Regular K-NN Decision Boundary')
    
    plt.subplot(1, 2, 2)
    plot_decision_boundary(weighted_knn, X_train, y_train, 'Weighted K-NN Decision Boundary')
    
    plt.tight_layout()
    plt.show()
    
    return regular_accuracy, weighted_accuracy

def main():
    """
    Run all demonstrations
    """
    print("K-Nearest Neighbors Complete Implementation and Examples")
    print("=" * 60)
    
    # Run all demonstrations
    knn_class, X_train_class, y_train_class, X_test_class, y_test_class = demonstrate_knn_classification()
    knn_reg, X_train_reg, y_train_reg, X_test_reg, y_test_reg = demonstrate_knn_regression()
    distance_results = demonstrate_distance_metrics()
    acc_unscaled, acc_scaled = demonstrate_feature_scaling()
    best_k, best_acc = demonstrate_hyperparameter_tuning()
    best_k_cv, best_cv_acc = demonstrate_cross_validation()
    reg_acc, weighted_acc = demonstrate_weighted_knn()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    print(f"Best distance metric: {max(distance_results, key=distance_results.get)}")
    print(f"Feature scaling improvement: {acc_scaled - acc_unscaled:.3f}")
    print(f"Best K value: {best_k}")
    print(f"Best cross-validation K: {best_k_cv}")
    print(f"Weighted K-NN improvement: {weighted_acc - reg_acc:.3f}")

if __name__ == "__main__":
    main() 