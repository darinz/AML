# Deep Learning Optimization & Computer Vision

## Overview

This guide covers advanced optimization techniques for deep learning, including batch normalization, dropout, residual connections, and their applications in computer vision tasks.

## Key Concepts

### 1. Deep Learning Challenges

1. **Vanishing/Exploding Gradients**: Gradients become too small or large in deep networks
2. **Internal Covariate Shift**: Distribution of layer inputs changes during training
3. **Overfitting**: Model memorizes training data instead of generalizing
4. **Training Instability**: Difficult to train very deep networks

### 2. Optimization Techniques

1. **Batch Normalization**: Normalize layer inputs to stabilize training
2. **Dropout**: Randomly disable neurons to prevent overfitting
3. **Residual Connections**: Skip connections to enable training of very deep networks
4. **Data Augmentation**: Increase training data diversity

## Batch Normalization

### Mathematical Foundation

Batch normalization normalizes the inputs to each layer:

```math
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
```

```math
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
```

```math
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
```

```math
y_i = \gamma \hat{x}_i + \beta
```

Where:
- $`\mu_B`$ and $`\sigma_B^2`$ are batch mean and variance
- $`\gamma`$ and $`\beta`` are learnable parameters
- $`\epsilon`$ is a small constant for numerical stability

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

class BatchNormalization:
    """Batch Normalization layer implementation"""
    
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Store for backpropagation
        self.cache = None
        self.training = True
        
    def forward(self, X):
        """Forward pass"""
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            
            # Normalize
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Scale and shift
            output = self.gamma * X_norm + self.beta
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Store for backpropagation
            self.cache = (X, X_norm, batch_mean, batch_var)
        else:
            # Use running statistics for inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            output = self.gamma * X_norm + self.beta
        
        return output
    
    def backward(self, d_output):
        """Backward pass"""
        X, X_norm, batch_mean, batch_var = self.cache
        m = X.shape[0]
        
        # Gradients for gamma and beta
        d_gamma = np.sum(d_output * X_norm, axis=0)
        d_beta = np.sum(d_output, axis=0)
        
        # Gradient for normalized input
        d_X_norm = d_output * self.gamma
        
        # Gradient for input
        d_X = (1 / m) * (1 / np.sqrt(batch_var + self.epsilon)) * (
            m * d_X_norm - np.sum(d_X_norm, axis=0) - 
            X_norm * np.sum(d_X_norm * X_norm, axis=0)
        )
        
        return d_X, d_gamma, d_beta
    
    def update_parameters(self, d_gamma, d_beta, learning_rate):
        """Update learnable parameters"""
        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta

class Dropout:
    """Dropout layer implementation"""
    
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
        
    def forward(self, X):
        """Forward pass"""
        if self.training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.p, size=X.shape) / (1 - self.p)
            return X * self.mask
        else:
            return X
    
    def backward(self, d_output):
        """Backward pass"""
        return d_output * self.mask

class ResidualBlock:
    """Residual block implementation"""
    
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Two-layer residual block
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, input_size) * 0.01
        self.b2 = np.zeros(input_size)
        
        # Batch normalization layers
        self.bn1 = BatchNormalization(hidden_size)
        self.bn2 = BatchNormalization(input_size)
        
        # Dropout layers
        self.dropout1 = Dropout(p=0.3)
        self.dropout2 = Dropout(p=0.3)
        
        # Store for backpropagation
        self.cache = None
        
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """Forward pass"""
        # Store input for residual connection
        identity = X
        
        # First layer
        z1 = np.dot(X, self.W1) + self.b1
        z1_norm = self.bn1.forward(z1)
        a1 = self.relu(z1_norm)
        a1_drop = self.dropout1.forward(a1)
        
        # Second layer
        z2 = np.dot(a1_drop, self.W2) + self.b2
        z2_norm = self.bn2.forward(z2)
        
        # Residual connection
        output = z2_norm + identity
        
        # Store for backpropagation
        self.cache = (identity, z1, z1_norm, a1, a1_drop, z2, z2_norm)
        
        return output
    
    def backward(self, d_output):
        """Backward pass"""
        identity, z1, z1_norm, a1, a1_drop, z2, z2_norm = self.cache
        
        # Gradient for residual connection
        d_z2_norm = d_output
        d_identity = d_output
        
        # Backpropagate through second layer
        d_z2, d_gamma2, d_beta2 = self.bn2.backward(d_z2_norm)
        d_a1_drop = np.dot(d_z2, self.W2.T)
        d_W2 = np.dot(a1_drop.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)
        
        # Backpropagate through dropout
        d_a1 = self.dropout1.backward(d_a1_drop)
        
        # Backpropagate through first layer
        d_z1_norm = d_a1 * self.relu_derivative(z1_norm)
        d_z1, d_gamma1, d_beta1 = self.bn1.backward(d_z1_norm)
        d_X = np.dot(d_z1, self.W1.T)
        d_W1 = np.dot(identity.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)
        
        # Update parameters
        self.bn1.update_parameters(d_gamma1, d_beta1, self.learning_rate)
        self.bn2.update_parameters(d_gamma2, d_beta2, self.learning_rate)
        
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        
        return d_X + d_identity

class DeepNetwork:
    """Deep network with optimization techniques"""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.batch_norms = []
        
        for i in range(self.num_layers - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Xavier initialization
            scale = np.sqrt(2.0 / (input_size + output_size))
            W = np.random.randn(input_size, output_size) * scale
            b = np.zeros(output_size)
            
            self.weights.append(W)
            self.biases.append(b)
            
            # Add batch normalization for hidden layers
            if i < self.num_layers - 2:
                self.batch_norms.append(BatchNormalization(output_size))
        
        # Dropout layers
        self.dropouts = [Dropout(p=0.3) for _ in range(self.num_layers - 2)]
        
        # Store for backpropagation
        self.activations = []
        self.pre_activations = []
        
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward pass"""
        self.activations = [X]
        self.pre_activations = []
        
        # Hidden layers
        for i in range(self.num_layers - 2):
            # Linear transformation
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.pre_activations.append(z)
            
            # Batch normalization
            z_norm = self.batch_norms[i].forward(z)
            
            # Activation function
            a = self.relu(z_norm)
            
            # Dropout
            a_drop = self.dropouts[i].forward(a)
            
            self.activations.append(a_drop)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.pre_activations.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        """Backward pass"""
        m = X.shape[0]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = output - y
        
        # Backpropagate through layers
        for l in range(self.num_layers - 2, -1, -1):
            # Compute gradients for current layer
            dW[l] = np.dot(self.activations[l].T, delta) / m
            db[l] = np.sum(delta, axis=0) / m
            
            # Compute error for previous layer
            if l > 0:
                # Backpropagate through dropout
                delta = self.dropouts[l-1].backward(delta)
                
                # Backpropagate through activation
                delta = delta * self.relu_derivative(self.pre_activations[l-1])
                
                # Backpropagate through batch normalization
                delta, d_gamma, d_beta = self.batch_norms[l-1].backward(delta)
                self.batch_norms[l-1].update_parameters(d_gamma, d_beta, self.learning_rate)
                
                # Backpropagate through linear transformation
                delta = np.dot(delta, self.weights[l-1].T)
        
        # Update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict(self, X):
        """Make predictions"""
        # Set training mode to False for inference
        for bn in self.batch_norms:
            bn.training = False
        for dropout in self.dropouts:
            dropout.training = False
        
        output = self.forward(X)
        
        # Reset training mode
        for bn in self.batch_norms:
            bn.training = True
        for dropout in self.dropouts:
            dropout.training = True
        
        return output
    
    def predict_classes(self, X):
        """Predict classes"""
        predictions = self.predict(X)
        return (predictions > 0.5).astype(int)

def demonstrate_batch_normalization():
    """Demonstrate Batch Normalization"""
    print("=== Batch Normalization Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=0, 
                             n_informative=10, random_state=42, n_clusters_per_class=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train networks with and without batch normalization
    networks = {
        'Without BN': DeepNetwork([10, 20, 20, 1], learning_rate=0.01),
        'With BN': DeepNetwork([10, 20, 20, 1], learning_rate=0.01)
    }
    
    # Disable batch normalization for first network
    for bn in networks['Without BN'].batch_norms:
        bn.training = False
    
    results = {}
    
    for name, network in networks.items():
        print(f"\nTraining {name}...")
        
        # Training parameters
        epochs = 50
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
                output = network.forward(batch_X)
                
                # Backward pass
                network.backward(batch_X, batch_y, output)
                
                # Compute loss
                batch_loss = network.compute_loss(batch_y, output)
                epoch_loss += batch_loss
                num_batches += 1
            
            # Average loss for epoch
            avg_train_loss = epoch_loss / num_batches
            
            # Test loss
            test_output = network.predict(X_test_scaled)
            test_loss = network.compute_loss(y_test.reshape(-1, 1), test_output)
            
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Test Loss: {test_losses[-1]:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Training loss comparison
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Test loss comparison
    plt.subplot(1, 3, 2)
    for name, result in results.items():
        plt.plot(result['test_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Final loss comparison
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    final_train_losses = [results[name]['final_train_loss'] for name in names]
    final_test_losses = [results[name]['final_test_loss'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, final_train_losses, width, label='Train Loss')
    plt.bar(x + width/2, final_test_losses, width, label='Test Loss')
    plt.xlabel('Network Type')
    plt.ylabel('Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
bn_results = demonstrate_batch_normalization()
```

## Data Augmentation

### Implementation

```python
def demonstrate_data_augmentation():
    """Demonstrate data augmentation techniques"""
    print("\n=== Data Augmentation Demo ===")
    
    # Generate synthetic image data
    np.random.seed(42)
    n_samples = 500
    img_size = 16
    
    # Create simple patterns
    X = np.zeros((n_samples, 1, img_size, img_size))
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
    
    # Data augmentation functions
    def rotate_image(image, angle):
        """Rotate image by given angle"""
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False)
    
    def add_noise(image, noise_factor=0.1):
        """Add random noise to image"""
        noise = np.random.randn(*image.shape) * noise_factor
        return np.clip(image + noise, 0, 1)
    
    def shift_image(image, shift_x, shift_y):
        """Shift image by given amount"""
        from scipy.ndimage import shift
        return shift(image, (shift_y, shift_x), cval=0)
    
    # Apply augmentations
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original image
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Rotated versions
        for angle in [90, 180, 270]:
            rotated = rotate_image(X[i, 0], angle)
            augmented_X.append(rotated.reshape(1, img_size, img_size))
            augmented_y.append(y[i])
        
        # Noisy versions
        for _ in range(2):
            noisy = add_noise(X[i, 0])
            augmented_X.append(noisy.reshape(1, img_size, img_size))
            augmented_y.append(y[i])
        
        # Shifted versions
        for shift_x, shift_y in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            shifted = shift_image(X[i, 0], shift_x, shift_y)
            augmented_X.append(shifted.reshape(1, img_size, img_size))
            augmented_y.append(y[i])
    
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    print(f"Original dataset size: {len(X)}")
    print(f"Augmented dataset size: {len(augmented_X)}")
    print(f"Augmentation factor: {len(augmented_X) / len(X):.1f}x")
    
    # Visualize augmentations
    plt.figure(figsize=(15, 10))
    
    # Original images
    plt.subplot(2, 4, 1)
    plt.imshow(X[0, 0], cmap='gray')
    plt.title('Original (Class 0)')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(X[n_samples//2, 0], cmap='gray')
    plt.title('Original (Class 1)')
    plt.axis('off')
    
    # Rotated versions
    plt.subplot(2, 4, 3)
    plt.imshow(rotate_image(X[0, 0], 90), cmap='gray')
    plt.title('Rotated 90°')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(rotate_image(X[0, 0], 180), cmap='gray')
    plt.title('Rotated 180°')
    plt.axis('off')
    
    # Noisy versions
    plt.subplot(2, 4, 5)
    plt.imshow(add_noise(X[0, 0], 0.2), cmap='gray')
    plt.title('With Noise')
    plt.axis('off')
    
    # Shifted versions
    plt.subplot(2, 4, 6)
    plt.imshow(shift_image(X[0, 0], 2, 0), cmap='gray')
    plt.title('Shifted Right')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(shift_image(X[0, 0], 0, 2), cmap='gray')
    plt.title('Shifted Down')
    plt.axis('off')
    
    # Combined augmentation
    plt.subplot(2, 4, 8)
    combined = add_noise(rotate_image(shift_image(X[0, 0], 1, 1), 45), 0.1)
    plt.imshow(combined, cmap='gray')
    plt.title('Combined')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return augmented_X, augmented_y

# Run demonstration
augmented_X, augmented_y = demonstrate_data_augmentation()
```

## Computer Vision Applications

### Image Classification

```python
def demonstrate_image_classification():
    """Demonstrate image classification with CNN"""
    print("\n=== Image Classification Demo ===")
    
    # Use augmented data from previous demonstration
    if 'augmented_X' not in globals():
        augmented_X, augmented_y = demonstrate_data_augmentation()
    
    # Convert to one-hot encoding
    y_onehot = np.zeros((len(augmented_y), 2))
    y_onehot[np.arange(len(augmented_y)), augmented_y] = 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(augmented_X, y_onehot, test_size=0.2, random_state=42)
    
    # Initialize CNN
    input_shape = (1, 16, 16)
    cnn = CNN(input_shape, num_classes=2, learning_rate=0.01)
    
    # Training parameters
    epochs = 30
    batch_size = 32
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Mini-batch training
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Forward pass
            output = cnn.forward(batch_X)
            
            # Backward pass
            cnn.backward(batch_X, batch_y, output)
            
            # Compute loss and accuracy
            batch_loss = cnn.compute_loss(batch_y, output)
            batch_predictions = cnn.predict_classes(batch_X)
            batch_true = np.argmax(batch_y, axis=1)
            
            epoch_loss += batch_loss
            correct_predictions += np.sum(batch_predictions == batch_true)
            total_predictions += len(batch_predictions)
            num_batches += 1
        
        # Average loss and accuracy for epoch
        avg_train_loss = epoch_loss / num_batches
        avg_train_accuracy = correct_predictions / total_predictions
        
        # Test performance
        test_output = cnn.forward(X_test)
        test_loss = cnn.compute_loss(y_test, test_output)
        test_predictions = cnn.predict_classes(X_test)
        test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(avg_train_accuracy)
        test_accuracies.append(test_accuracy)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_accuracy:.4f}")
            print(f"          Test Loss = {test_loss:.4f}, Test Acc = {test_accuracy:.4f}")
    
    # Final evaluation
    final_train_accuracy = train_accuracies[-1]
    final_test_accuracy = test_accuracies[-1]
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {final_train_accuracy:.4f}")
    print(f"Test Accuracy: {final_test_accuracy:.4f}")
    
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
    
    # Accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(1, 3, 3)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), test_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    return cnn, train_losses, test_losses, train_accuracies, test_accuracies

# Run demonstration
classification_cnn, train_losses, test_losses, train_accuracies, test_accuracies = demonstrate_image_classification()
```

## Transfer Learning

### Implementation

```python
def demonstrate_transfer_learning():
    """Demonstrate transfer learning concepts"""
    print("\n=== Transfer Learning Demo ===")
    
    # Simulate pre-trained features
    np.random.seed(42)
    
    # Generate source task data (more complex)
    n_source = 1000
    img_size = 16
    
    X_source = np.zeros((n_source, 1, img_size, img_size))
    y_source = np.zeros(n_source, dtype=int)
    
    for i in range(n_source):
        if i < n_source // 3:
            # Complex pattern 1
            X_source[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X_source[i, 0, img_size//4, :] = 1.0
            X_source[i, 0, 3*img_size//4, :] = 1.0
            X_source[i, 0, :, img_size//4] = 0.5
            y_source[i] = 0
        elif i < 2 * n_source // 3:
            # Complex pattern 2
            X_source[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X_source[i, 0, :, img_size//4] = 1.0
            X_source[i, 0, :, 3*img_size//4] = 1.0
            X_source[i, 0, img_size//4, :] = 0.5
            y_source[i] = 1
        else:
            # Complex pattern 3
            X_source[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X_source[i, 0, img_size//2, :] = 1.0
            X_source[i, 0, :, img_size//2] = 1.0
            y_source[i] = 2
    
    # Generate target task data (simpler, different distribution)
    n_target = 200
    X_target = np.zeros((n_target, 1, img_size, img_size))
    y_target = np.zeros(n_target, dtype=int)
    
    for i in range(n_target):
        if i < n_target // 2:
            # Simple horizontal lines
            X_target[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X_target[i, 0, img_size//3, :] = 1.0
            y_target[i] = 0
        else:
            # Simple vertical lines
            X_target[i, 0, :, :] = np.random.randn(img_size, img_size) * 0.1
            X_target[i, 0, :, img_size//3] = 1.0
            y_target[i] = 1
    
    # Convert to one-hot encoding
    y_source_onehot = np.zeros((len(y_source), 3))
    y_source_onehot[np.arange(len(y_source)), y_source] = 1
    
    y_target_onehot = np.zeros((len(y_target), 2))
    y_target_onehot[np.arange(len(y_target)), y_target] = 1
    
    # Split target data
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target_onehot, test_size=0.2, random_state=42
    )
    
    # Train source model (pre-trained)
    print("Training source model...")
    source_cnn = CNN((1, 16, 16), num_classes=3, learning_rate=0.01)
    
    # Train source model
    for epoch in range(20):
        indices = np.random.permutation(len(X_source))
        X_shuffled = X_source[indices]
        y_shuffled = y_source_onehot[indices]
        
        for i in range(0, len(X_shuffled), 32):
            batch_X = X_shuffled[i:i+32]
            batch_y = y_shuffled[i:i+32]
            
            output = source_cnn.forward(batch_X)
            source_cnn.backward(batch_X, batch_y, output)
    
    # Transfer learning: use pre-trained features
    print("Training target model with transfer learning...")
    target_cnn = CNN((1, 16, 16), num_classes=2, learning_rate=0.01)
    
    # Copy convolutional layers from source model
    target_cnn.layers[0].weights = source_cnn.layers[0].weights.copy()
    target_cnn.layers[0].bias = source_cnn.layers[0].bias.copy()
    target_cnn.layers[2].weights = source_cnn.layers[2].weights.copy()
    target_cnn.layers[2].bias = source_cnn.layers[2].bias.copy()
    
    # Train target model
    transfer_train_losses = []
    transfer_test_losses = []
    
    for epoch in range(15):
        indices = np.random.permutation(len(X_target_train))
        X_shuffled = X_target_train[indices]
        y_shuffled = y_target_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_shuffled), 32):
            batch_X = X_shuffled[i:i+32]
            batch_y = y_shuffled[i:i+32]
            
            output = target_cnn.forward(batch_X)
            target_cnn.backward(batch_X, batch_y, output)
            
            batch_loss = target_cnn.compute_loss(batch_y, output)
            epoch_loss += batch_loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        test_output = target_cnn.forward(X_target_test)
        test_loss = target_cnn.compute_loss(y_target_test, test_output)
        
        transfer_train_losses.append(avg_train_loss)
        transfer_test_losses.append(test_loss)
    
    # Train from scratch for comparison
    print("Training target model from scratch...")
    scratch_cnn = CNN((1, 16, 16), num_classes=2, learning_rate=0.01)
    
    scratch_train_losses = []
    scratch_test_losses = []
    
    for epoch in range(15):
        indices = np.random.permutation(len(X_target_train))
        X_shuffled = X_target_train[indices]
        y_shuffled = y_target_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_shuffled), 32):
            batch_X = X_shuffled[i:i+32]
            batch_y = y_shuffled[i:i+32]
            
            output = scratch_cnn.forward(batch_X)
            scratch_cnn.backward(batch_X, batch_y, output)
            
            batch_loss = scratch_cnn.compute_loss(batch_y, output)
            epoch_loss += batch_loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        test_output = scratch_cnn.forward(X_target_test)
        test_loss = scratch_cnn.compute_loss(y_target_test, test_output)
        
        scratch_train_losses.append(avg_train_loss)
        scratch_test_losses.append(test_loss)
    
    # Evaluate final performance
    transfer_test_pred = target_cnn.predict_classes(X_target_test)
    scratch_test_pred = scratch_cnn.predict_classes(X_target_test)
    
    transfer_accuracy = np.mean(transfer_test_pred == np.argmax(y_target_test, axis=1))
    scratch_accuracy = np.mean(scratch_test_pred == np.argmax(y_target_test, axis=1))
    
    print(f"\nFinal Results:")
    print(f"Transfer Learning Accuracy: {transfer_accuracy:.4f}")
    print(f"Training from Scratch Accuracy: {scratch_accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Training loss comparison
    plt.subplot(1, 3, 1)
    plt.plot(transfer_train_losses, label='Transfer Learning')
    plt.plot(scratch_train_losses, label='From Scratch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Test loss comparison
    plt.subplot(1, 3, 2)
    plt.plot(transfer_test_losses, label='Transfer Learning')
    plt.plot(scratch_test_losses, label='From Scratch')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Final accuracy comparison
    plt.subplot(1, 3, 3)
    methods = ['Transfer Learning', 'From Scratch']
    accuracies = [transfer_accuracy, scratch_accuracy]
    
    bars = plt.bar(methods, accuracies)
    plt.xlabel('Method')
    plt.ylabel('Test Accuracy')
    plt.title('Final Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return source_cnn, target_cnn, scratch_cnn

# Run demonstration
source_model, transfer_model, scratch_model = demonstrate_transfer_learning()
```

## Summary

Deep learning optimization and computer vision techniques provide powerful tools for building effective neural networks:

1. **Batch Normalization**: Stabilizes training and improves convergence
2. **Dropout**: Prevents overfitting by randomly disabling neurons
3. **Residual Connections**: Enables training of very deep networks
4. **Data Augmentation**: Increases training data diversity
5. **Transfer Learning**: Leverages pre-trained models for new tasks

Key takeaways:

- **Batch Normalization**: Normalizes layer inputs to stabilize training
- **Dropout**: Regularization technique that prevents overfitting
- **Residual Connections**: Skip connections that enable deep networks
- **Data Augmentation**: Increases effective training data size
- **Transfer Learning**: Reuses learned features for new tasks
- **Computer Vision**: CNNs excel at image processing tasks

Understanding these techniques provides a solid foundation for building effective deep learning systems. 