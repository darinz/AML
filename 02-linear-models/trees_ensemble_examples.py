"""
Decision Trees & Ensemble Methods Examples

This file contains comprehensive examples demonstrating:
- Decision trees from scratch
- Bagging (Bootstrap Aggregating)
- Random Forest
- AdaBoost
- Model comparison and evaluation
- Real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

class DecisionTreeNode:
    """Node in a decision tree"""
    
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = value is not None

class DecisionTree:
    """Decision Tree implementation"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.root = None
        
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """Calculate entropy"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _mse(self, y):
        """Calculate mean squared error"""
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.task == 'classification':
            if self.criterion == 'gini':
                return self._gini_impurity(y)
            elif self.criterion == 'entropy':
                return self._entropy(y)
        else:  # regression
            return self._mse(y)
    
    def _find_best_split(self, X, y):
        """Find the best split for the data"""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                # Calculate information gain
                left_weight = np.sum(left_mask) / n_samples
                right_weight = np.sum(right_mask) / n_samples
                gain = current_impurity - (left_weight * left_impurity + right_weight * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf(self, y):
        """Create a leaf node"""
        if self.task == 'classification':
            classes, counts = np.unique(y, return_counts=True)
            return DecisionTreeNode(value=classes[np.argmax(counts)])
        else:  # regression
            return DecisionTreeNode(value=np.mean(y))
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_gain <= 0:
            return self._create_leaf(y)
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Create child nodes
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X, y):
        """Fit the decision tree"""
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.root))
        return np.array(predictions)

def demonstrate_decision_tree():
    """Demonstrate decision tree"""
    print("=== Decision Tree Demo ===")
    
    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit decision tree
    dt = DecisionTree(max_depth=5, min_samples_split=10, criterion='gini')
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Visualize decision boundary
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Decision boundary
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.6)
    plt.title('Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return dt, X_train, y_train, X_test, y_test

class BaggingClassifier:
    """Bagging ensemble classifier"""
    
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators = []
        
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.max_samples)
        
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Fit bagging ensemble"""
        X = np.array(X)
        y = np.array(y)
        
        self.estimators = []
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Create and fit estimator
            estimator = type(self.base_estimator)()
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = []
        
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        
        # Majority vote for classification
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.T)

def demonstrate_bagging():
    """Demonstrate bagging"""
    print("\n=== Bagging Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create base estimator
    base_dt = DecisionTree(max_depth=3, min_samples_split=10)
    
    # Fit bagging ensemble
    bagging = BaggingClassifier(base_estimator=base_dt, n_estimators=10)
    bagging.fit(X_train, y_train)
    
    # Make predictions
    y_pred = bagging.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Bagging Accuracy: {accuracy:.3f}")
    
    # Compare with single tree
    single_dt = DecisionTree(max_depth=3, min_samples_split=10)
    single_dt.fit(X_train, y_train)
    single_pred = single_dt.predict(X_test)
    single_accuracy = accuracy_score(y_test, single_pred)
    print(f"Single Tree Accuracy: {single_accuracy:.3f}")
    
    return bagging, X_train, y_train, X_test, y_test

class RandomForest:
    """Random Forest implementation"""
    
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.estimators = []
        self.feature_indices = []
        
    def _get_feature_subset(self, n_features):
        """Get random feature subset"""
        if self.max_features == 'sqrt':
            n_subset = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_subset = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            n_subset = int(self.max_features * n_features)
        else:
            n_subset = min(self.max_features, n_features)
        
        return np.random.choice(n_features, size=n_subset, replace=False)
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Fit random forest"""
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        self.estimators = []
        self.feature_indices = []
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Get feature subset
            feature_subset = self._get_feature_subset(n_features)
            X_subset = X_bootstrap[:, feature_subset]
            
            # Create and fit tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, y_bootstrap)
            
            self.estimators.append(tree)
            self.feature_indices.append(feature_subset)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = []
        
        for i, estimator in enumerate(self.estimators):
            X_subset = X[:, self.feature_indices[i]]
            predictions.append(estimator.predict(X_subset))
        
        # Majority vote
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.T)
    
    def feature_importance(self, X, y):
        """Calculate feature importance"""
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        for i, estimator in enumerate(self.estimators):
            # Get feature subset
            feature_subset = self.feature_indices[i]
            X_subset = X[:, feature_subset]
            
            # Calculate importance for this tree (simplified)
            # In practice, you'd track importance during tree building
            for j, feature_idx in enumerate(feature_subset):
                importance[feature_idx] += 1
        
        return importance / np.sum(importance)

def demonstrate_random_forest():
    """Demonstrate random forest"""
    print("\n=== Random Forest Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                              n_informative=5, n_redundant=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit random forest
    rf = RandomForest(n_estimators=20, max_depth=5, max_features='sqrt')
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.3f}")
    
    # Feature importance
    importance = rf.feature_importance(X_train, y_train)
    print("\nFeature Importance:")
    for i, imp in enumerate(importance):
        print(f"  Feature {i}: {imp:.3f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rf, X_train, y_train, X_test, y_test

class AdaBoost:
    """AdaBoost implementation"""
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        
    def fit(self, X, y):
        """Fit AdaBoost ensemble"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        
        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators = []
        self.estimator_weights = []
        
        for _ in range(self.n_estimators):
            # Create weak learner (decision stump)
            estimator = DecisionTree(max_depth=1, min_samples_split=2)
            estimator.fit(X, y)
            
            # Make predictions
            predictions = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y
            weighted_error = np.sum(sample_weights * incorrect)
            
            # Skip if perfect classifier or error too high
            if weighted_error <= 0 or weighted_error >= 0.5:
                break
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect * ((predictions != y) * 2 - 1))
            sample_weights /= np.sum(sample_weights)
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions += weight * estimator.predict(X)
        
        return np.sign(predictions)

def demonstrate_adaboost():
    """Demonstrate AdaBoost"""
    print("\n=== AdaBoost Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit AdaBoost
    adaboost = AdaBoost(n_estimators=50, learning_rate=1.0)
    adaboost.fit(X_train, y_train)
    
    # Make predictions
    y_pred = adaboost.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"AdaBoost Accuracy: {accuracy:.3f}")
    print(f"Number of estimators used: {len(adaboost.estimators)}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.6)
    plt.title('AdaBoost Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return adaboost, X_train, y_train, X_test, y_test

def compare_ensemble_methods():
    """Compare different ensemble methods"""
    print("\n=== Ensemble Methods Comparison ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                              n_informative=5, n_redundant=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define ensemble methods
    base_dt = DecisionTree(max_depth=3, min_samples_split=10)
    
    ensembles = {
        'Single Tree': DecisionTree(max_depth=5, min_samples_split=10),
        'Bagging': BaggingClassifier(base_estimator=base_dt, n_estimators=10),
        'Random Forest': RandomForest(n_estimators=20, max_depth=5),
        'AdaBoost': AdaBoost(n_estimators=50)
    }
    
    results = {}
    
    for name, ensemble in ensembles.items():
        print(f"\nTraining {name}...")
        
        # Fit ensemble
        if name == 'AdaBoost':
            y_train_ada = np.where(y_train == 0, -1, 1)
            ensemble.fit(X_train, y_train_ada)
            y_pred = ensemble.predict(X_test)
            y_pred = np.where(y_pred == -1, 0, 1)
        else:
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.3f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    ensemble_names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(ensemble_names, accuracies)
    plt.xlabel('Ensemble Method')
    plt.ylabel('Accuracy')
    plt.title('Ensemble Methods Comparison')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_real_world_application():
    """Demonstrate credit card fraud detection"""
    print("\n=== Credit Card Fraud Detection Demo ===")
    
    # Generate synthetic credit card transaction data
    np.random.seed(42)
    n_transactions = 10000
    
    # Features: amount, time_of_day, merchant_category, distance_from_home, 
    # previous_transactions, card_age, transaction_frequency
    amount = np.random.exponential(100, n_transactions)
    time_of_day = np.random.uniform(0, 24, n_transactions)
    merchant_category = np.random.randint(0, 10, n_transactions)
    distance_from_home = np.random.exponential(50, n_transactions)
    previous_transactions = np.random.poisson(20, n_transactions)
    card_age = np.random.exponential(5, n_transactions)
    transaction_frequency = np.random.exponential(2, n_transactions)
    
    # Create feature matrix
    X = np.column_stack([amount, time_of_day, merchant_category, distance_from_home, 
                        previous_transactions, card_age, transaction_frequency])
    
    # Generate fraud labels (fraud = 1, legitimate = 0)
    # Higher risk factors increase fraud probability
    fraud_risk = (0.001 * amount + 0.02 * distance_from_home + 
                  0.1 * (merchant_category == 9) +  # High-risk category
                  0.05 * (time_of_day < 6) +  # Late night transactions
                  0.02 * (transaction_frequency > 5))  # High frequency
    
    fraud_prob = 1 / (1 + np.exp(-(-3 + fraud_risk)))
    y = np.random.binomial(1, fraud_prob)
    
    print(f"Fraud rate: {np.mean(y):.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different ensemble models
    models = {
        'Single Tree': DecisionTree(max_depth=5, min_samples_split=50),
        'Bagging': BaggingClassifier(base_estimator=DecisionTree(max_depth=3), n_estimators=20),
        'Random Forest': RandomForest(n_estimators=50, max_depth=5, max_features='sqrt'),
        'AdaBoost': AdaBoost(n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Fit model
        if name == 'AdaBoost':
            y_train_ada = np.where(y_train == 0, -1, 1)
            model.fit(X_train_scaled, y_train_ada)
            y_pred = model.predict(X_test_scaled)
            y_pred = np.where(y_pred == -1, 0, 1)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate precision, recall, F1-score for fraud detection
        cm = confusion_matrix(y_test, y_pred)
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
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        plt.bar(x + i * width, values, width, label=metric.capitalize())
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Fraud Detection Performance')
    plt.xticks(x + width * 1.5, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion matrices
    for i, name in enumerate(model_names):
        plt.subplot(1, 3, i + 2)
        cm = results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Run all demonstrations
    print("Decision Trees & Ensemble Methods Examples")
    print("=" * 50)
    
    # Decision tree
    dt_model, X_train, y_train, X_test, y_test = demonstrate_decision_tree()
    
    # Bagging
    bagging_model, X_train, y_train, X_test, y_test = demonstrate_bagging()
    
    # Random forest
    rf_model, X_train, y_train, X_test, y_test = demonstrate_random_forest()
    
    # AdaBoost
    adaboost_model, X_train, y_train, X_test, y_test = demonstrate_adaboost()
    
    # Ensemble comparison
    ensemble_results = compare_ensemble_methods()
    
    # Real-world application
    fraud_results = demonstrate_real_world_application()
    
    print("\nAll demonstrations completed!") 