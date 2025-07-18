"""
Neural Networks & Deep Learning Examples

This file contains comprehensive examples demonstrating:
- Stochastic Gradient Descent and optimization
- Multi-Layer Perceptrons with backpropagation
- Convolutional Neural Networks
- Deep learning optimization techniques
- Computer vision applications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Stochastic Gradient Descent & Optimization
# ============================================================================

class SimpleNeuralNetwork:
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass"""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward pass"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        """Update parameters using gradient descent"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class SGD:
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, params, grads):
        """Update parameters with momentum"""
        if not self.velocity:
            for key in params:
                self.velocity[key] = np.zeros_like(params[key])
        
        for key in params:
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]

class Adam:
    """Adam optimizer"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, params, grads):
        """Update parameters with Adam"""
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def demonstrate_sgd():
    """Demonstrate Stochastic Gradient Descent"""
    print("=== Stochastic Gradient Descent Demo ===")
    
    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1)
    
    # Training parameters
    epochs = 100
    batch_size = 32
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train_scaled))
        X_shuffled = X_train_scaled[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size].reshape(-1, 1)
            
            # Forward pass
            output = nn.forward(batch_X)
            
            # Backward pass
            dW1, db1, dW2, db2 = nn.backward(batch_X, batch_y, output)
            
            # Update parameters
            nn.update_parameters(dW1, db1, dW2, db2)
            
            # Compute loss
            batch_loss = nn.compute_loss(batch_y, output)
            epoch_loss += batch_loss
            num_batches += 1
        
        # Average loss for epoch
        avg_train_loss = epoch_loss / num_batches
        
        # Test loss
        test_output = nn.forward(X_test_scaled)
        test_loss = nn.compute_loss(y_test.reshape(-1, 1), test_output)
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Decision boundary
    plt.subplot(1, 3, 2)
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Training data
    plt.subplot(1, 3, 3)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return nn, train_losses, test_losses

# ============================================================================
# Multi-Layer Perceptrons & Backpropagation
# ============================================================================

class MLP:
    """Multi-Layer Perceptron implementation with backpropagation"""
    
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid', 
                 learning_rate=0.01, weight_init='xavier'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_parameters()
        
        # Store activations and pre-activations for backpropagation
        self.activations = []
        self.pre_activations = []
        
    def initialize_parameters(self):
        """Initialize weights and biases"""
        for i in range(self.num_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            # Weight initialization
            if self.weight_init == 'xavier':
                # Xavier/Glorot initialization
                scale = np.sqrt(2.0 / (input_size + output_size))
                W = np.random.randn(input_size, output_size) * scale
            elif self.weight_init == 'he':
                # He initialization
                scale = np.sqrt(2.0 / input_size)
                W = np.random.randn(input_size, output_size) * scale
            else:
                # Random initialization
                W = np.random.randn(input_size, output_size) * 0.01
            
            b = np.zeros((1, output_size))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def activation_function(self, x, activation_type):
        """Apply activation function"""
        if activation_type == 'relu':
            return np.maximum(0, x)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation_type == 'tanh':
            return np.tanh(x)
        elif activation_type == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def activation_derivative(self, x, activation_type):
        """Compute derivative of activation function"""
        if activation_type == 'relu':
            return np.where(x > 0, 1, 0)
        elif activation_type == 'sigmoid':
            s = self.activation_function(x, 'sigmoid')
            return s * (1 - s)
        elif activation_type == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation_type == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.pre_activations = []
        
        # Hidden layers
        for i in range(self.num_layers - 2):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.pre_activations.append(z)
            a = self.activation_function(z, self.activation)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.pre_activations.append(z)
        a = self.activation_function(z, self.output_activation)
        self.activations.append(a)
        
        return a
    
    def backward(self, X, y, output):
        """Backward pass to compute gradients"""
        m = X.shape[0]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error
        if self.output_activation == 'sigmoid':
            # Binary cross-entropy loss
            delta = output - y
        else:
            # Mean squared error
            delta = (output - y) * self.activation_derivative(
                self.pre_activations[-1], self.output_activation
            )
        
        # Backpropagate through layers
        for l in range(self.num_layers - 2, -1, -1):
            # Compute gradients for current layer
            dW[l] = np.dot(self.activations[l].T, delta) / m
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Compute error for previous layer
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(
                    self.pre_activations[l-1], self.activation
                )
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update parameters using computed gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss function"""
        if self.output_activation == 'sigmoid':
            # Binary cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Mean squared error
            return np.mean((y_true - y_pred) ** 2)
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def predict_classes(self, X):
        """Predict classes for classification"""
        predictions = self.predict(X)
        if self.output_activation == 'sigmoid':
            return (predictions > 0.5).astype(int)
        else:
            return np.argmax(predictions, axis=1)

def demonstrate_mlp():
    """Demonstrate Multi-Layer Perceptron"""
    print("\n=== Multi-Layer Perceptron Demo ===")
    
    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize MLP
    layer_sizes = [2, 10, 10, 1]  # Input, hidden1, hidden2, output
    mlp = MLP(layer_sizes, activation='relu', output_activation='sigmoid', 
              learning_rate=0.01, weight_init='xavier')
    
    # Training parameters
    epochs = 100
    batch_size = 32
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train_scaled))
        X_shuffled = X_train_scaled[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size].reshape(-1, 1)
            
            # Forward pass
            output = mlp.forward(batch_X)
            
            # Backward pass
            dW, db = mlp.backward(batch_X, batch_y, output)
            
            # Update parameters
            mlp.update_parameters(dW, db)
            
            # Compute loss
            batch_loss = mlp.compute_loss(batch_y, output)
            epoch_loss += batch_loss
            num_batches += 1
        
        # Average loss for epoch
        avg_train_loss = epoch_loss / num_batches
        
        # Test loss
        test_output = mlp.forward(X_test_scaled)
        test_loss = mlp.compute_loss(y_test.reshape(-1, 1), test_output)
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    # Evaluate model
    train_predictions = mlp.predict_classes(X_train_scaled)
    test_predictions = mlp.predict_classes(X_test_scaled)
    
    train_accuracy = np.mean(train_predictions.flatten() == y_train)
    test_accuracy = np.mean(test_predictions.flatten() == y_test)
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Decision boundary
    plt.subplot(1, 3, 2)
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Training data
    plt.subplot(1, 3, 3)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return mlp, train_losses, test_losses

# ============================================================================
# Convolutional Neural Networks
# ============================================================================

class Conv2D:
    """2D Convolutional layer implementation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(out_channels)
        
        # Store for backpropagation
        self.input = None
        self.output = None
        
    def forward(self, X):
        """Forward pass"""
        self.input = X
        batch_size, in_channels, height, width = X.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Apply padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            X_padded = X
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = X_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Apply filter
                        output[b, c_out, h_out, w_out] = np.sum(
                            patch * self.weights[c_out]
                        ) + self.bias[c_out]
        
        self.output = output
        return output

class MaxPool2D:
    """2D Max Pooling layer implementation"""
    
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        # Store for backpropagation
        self.input = None
        self.output = None
        self.max_indices = None
        
    def forward(self, X):
        """Forward pass"""
        self.input = X
        batch_size, channels, height, width = X.shape
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        # Initialize output and max indices
        output = np.zeros((batch_size, channels, out_height, out_width))
        max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = X[b, c, h_start:h_end, w_start:w_end]
                        
                        # Find maximum and its index
                        max_val = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        output[b, c, h_out, w_out] = max_val
                        max_indices[b, c, h_out, w_out] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        self.output = output
        self.max_indices = max_indices
        return output

class Flatten:
    """Flatten layer implementation"""
    
    def __init__(self):
        self.input_shape = None
        
    def forward(self, X):
        """Forward pass"""
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

class CNN:
    """Simple Convolutional Neural Network"""
    
    def __init__(self, input_shape, num_classes, learning_rate=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Build network architecture
        self.layers = []
        
        # Convolutional layers
        self.layers.append(Conv2D(in_channels=input_shape[0], out_channels=16, 
                                 kernel_size=3, stride=1, padding=1))
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        self.layers.append(Conv2D(in_channels=16, out_channels=32, 
                                 kernel_size=3, stride=1, padding=1))
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        # Flatten layer
        self.layers.append(Flatten())
        
        # Calculate flattened size
        # After 2 max pooling layers with stride 2, spatial dimensions are reduced by 4
        flattened_size = 32 * (input_shape[1] // 4) * (input_shape[2] // 4)
        
        # Fully connected layers
        self.fc1_weights = np.random.randn(flattened_size, 128) * 0.01
        self.fc1_bias = np.zeros(128)
        self.fc2_weights = np.random.randn(128, num_classes) * 0.01
        self.fc2_bias = np.zeros(num_classes)
        
        # Store for backpropagation
        self.activations = []
        self.pre_activations = []
        
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward pass"""
        self.activations = [X]
        self.pre_activations = []
        
        # Forward through convolutional layers
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                X = layer.forward(X)
                X = self.relu(X)
                self.activations.append(X)
            else:
                X = layer.forward(X)
                self.activations.append(X)
        
        # Fully connected layers
        z1 = np.dot(X, self.fc1_weights) + self.fc1_bias
        self.pre_activations.append(z1)
        a1 = self.relu(z1)
        self.activations.append(a1)
        
        z2 = np.dot(a1, self.fc2_weights) + self.fc2_bias
        self.pre_activations.append(z2)
        output = self.sigmoid(z2)
        self.activations.append(output)
        
        return output
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def predict_classes(self, X):
        """Predict classes"""
        predictions = self.predict(X)
        return (predictions > 0.5).astype(int)

def create_synthetic_image_data():
    """Create synthetic image data for demonstration"""
    np.random.seed(42)
    
    # Create simple patterns
    n_samples = 1000
    img_size = 16
    n_channels = 1
    
    # Generate different patterns
    X = np.zeros((n_samples, n_channels, img_size, img_size))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if i < n_samples // 2:
            # Horizontal lines
            X[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X[i, 0, img_size//4, :] = 1.0
            X[i, 0, 3*img_size//4, :] = 1.0
            y[i] = 0
        else:
            # Vertical lines
            X[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X[i, 0, :, img_size//4] = 1.0
            X[i, 0, :, 3*img_size//4] = 1.0
            y[i] = 1
    
    return X, y

def demonstrate_cnn():
    """Demonstrate Convolutional Neural Network"""
    print("\n=== Convolutional Neural Network Demo ===")
    
    # Generate synthetic image data
    X, y = create_synthetic_image_data()
    
    # Convert to one-hot encoding
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
    # Initialize CNN
    input_shape = (1, 16, 16)  # (channels, height, width)
    cnn = CNN(input_shape, num_classes=2, learning_rate=0.01)
    
    # Training parameters
    epochs = 50
    batch_size = 32
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Forward pass
            output = cnn.forward(batch_X)
            
            # Compute loss
            batch_loss = cnn.compute_loss(batch_y, output)
            epoch_loss += batch_loss
            num_batches += 1
        
        # Average loss for epoch
        avg_train_loss = epoch_loss / num_batches
        
        # Test loss
        test_output = cnn.forward(X_test)
        test_loss = cnn.compute_loss(y_test, test_output)
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    # Evaluate model
    train_predictions = cnn.predict_classes(X_train)
    test_predictions = cnn.predict_classes(X_test)
    
    train_accuracy = np.mean(train_predictions.flatten() == np.argmax(y_train, axis=1))
    test_accuracy = np.mean(test_predictions.flatten() == np.argmax(y_test, axis=1))
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Sample images
    plt.subplot(1, 3, 2)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(X_train[i, 0], cmap='gray')
        plt.title(f'Class {np.argmax(y_train[i])}')
        plt.axis('off')
    plt.suptitle('Sample Training Images')
    
    # Feature maps (first conv layer)
    plt.subplot(1, 3, 3)
    sample_output = cnn.forward(X_train[:1])
    feature_maps = cnn.activations[1][0]  # First conv layer output
    
    for i in range(min(4, feature_maps.shape[0])):
        plt.subplot(2, 2, i + 1)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.title(f'Feature Map {i}')
        plt.axis('off')
    plt.suptitle('Feature Maps (First Conv Layer)')
    
    plt.tight_layout()
    plt.show()
    
    return cnn, train_losses, test_losses

# ============================================================================
# Main Demonstration
# ============================================================================

if __name__ == "__main__":
    # Run all demonstrations
    print("Neural Networks & Deep Learning Examples")
    print("=" * 50)
    
    # SGD demonstration
    nn_model, train_losses, test_losses = demonstrate_sgd()
    
    # MLP demonstration
    mlp_model, mlp_train_losses, mlp_test_losses = demonstrate_mlp()
    
    # CNN demonstration
    cnn_model, cnn_train_losses, cnn_test_losses = demonstrate_cnn()
    
    print("\nAll demonstrations completed!") 