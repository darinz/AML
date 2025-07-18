# Stochastic Gradient Descent & Optimization

## Overview

Stochastic Gradient Descent (SGD) is the fundamental optimization algorithm that powers modern deep learning. This guide covers the mathematical foundations, implementation details, and advanced optimization techniques used in neural network training.

## Key Concepts

### 1. What is Gradient Descent?

Gradient descent is an iterative optimization algorithm for finding the minimum of a function. For a loss function $`L(\theta)`$, the update rule is:

```math
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
```

Where:
- $`\theta_t`$ are the parameters at iteration $`t`$
- $`\alpha`$ is the learning rate
- $`\nabla L(\theta_t)`$ is the gradient of the loss function

### 2. Types of Gradient Descent

1. **Batch Gradient Descent**: Uses entire dataset for each update
2. **Stochastic Gradient Descent**: Uses single sample for each update
3. **Mini-batch Gradient Descent**: Uses subset of data for each update

## Mathematical Foundation

### Loss Function

For a neural network with parameters $`\theta`$ and training data $`\{(x_i, y_i)\}_{i=1}^n`$:

```math
L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)
```

Where $`\ell`$ is the loss function and $`f_\theta`$ is the neural network.

### Gradient Computation

The gradient is computed using the chain rule:

```math
\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \ell(f_\theta(x_i), y_i)
```

### SGD Update Rule

For mini-batch SGD with batch size $`B`$:

```math
\theta_{t+1} = \theta_t - \alpha \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla_\theta \ell(f_\theta(x_i), y_i)
```

Where $`\mathcal{B}_t`$ is the mini-batch at iteration $`t`$.

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

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
        
        # Store gradients for momentum
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        
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

# Run demonstration
nn_model, train_losses, test_losses = demonstrate_sgd()
```

## Advanced Optimizers

### Momentum

Momentum helps accelerate SGD in the relevant direction and dampens oscillations:

```math
v_{t+1} = \beta v_t + \alpha \nabla L(\theta_t)
```

```math
\theta_{t+1} = \theta_t - v_{t+1}
```

### RMSprop

RMSprop adapts the learning rate for each parameter:

```math
v_{t+1} = \beta v_t + (1 - \beta) (\nabla L(\theta_t))^2
```

```math
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_{t+1} + \epsilon}} \nabla L(\theta_t)
```

### Adam

Adam combines the benefits of momentum and RMSprop:

```math
m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(\theta_t)
```

```math
v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(\theta_t))^2
```

```math
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^t}
```

```math
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^t}
```

```math
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}
```

### Implementation

```python
def compare_optimizers():
    """Compare different optimization algorithms"""
    print("\n=== Optimizer Comparison Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define optimizers
    optimizers = {
        'SGD': SGD(learning_rate=0.1),
        'SGD with Momentum': SGD(learning_rate=0.1, momentum=0.9),
        'Adam': Adam(learning_rate=0.001)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}...")
        
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
                
                # Update parameters using optimizer
                if name == 'Adam':
                    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
                    params = {'W1': nn.W1, 'b1': nn.b1, 'W2': nn.W2, 'b2': nn.b2}
                    optimizer.update(params, grads)
                    nn.W1, nn.b1, nn.W2, nn.b2 = params['W1'], params['b1'], params['W2'], params['b2']
                else:
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
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Test Loss: {test_losses[-1]:.4f}")
    
    # Visualize comparison
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
    plt.xlabel('Optimizer')
    plt.ylabel('Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run comparison
optimizer_results = compare_optimizers()
```

## Learning Rate Scheduling

### Step Decay

```python
class StepDecayScheduler:
    """Step decay learning rate scheduler"""
    
    def __init__(self, initial_lr=0.1, decay_factor=0.5, step_size=30):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size
        self.current_lr = initial_lr
        
    def step(self, epoch):
        """Update learning rate"""
        if epoch > 0 and epoch % self.step_size == 0:
            self.current_lr *= self.decay_factor
        return self.current_lr

class CosineAnnealingScheduler:
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, initial_lr=0.1, min_lr=0.001, T_max=100):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.T_max = T_max
        
    def step(self, epoch):
        """Update learning rate"""
        if epoch >= self.T_max:
            return self.min_lr
        
        lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        return lr

def demonstrate_learning_rate_scheduling():
    """Demonstrate learning rate scheduling"""
    print("\n=== Learning Rate Scheduling Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define schedulers
    schedulers = {
        'Constant LR': lambda epoch: 0.1,
        'Step Decay': StepDecayScheduler(initial_lr=0.1, decay_factor=0.5, step_size=30),
        'Cosine Annealing': CosineAnnealingScheduler(initial_lr=0.1, min_lr=0.001, T_max=100)
    }
    
    results = {}
    
    for name, scheduler in schedulers.items():
        print(f"\nTraining with {name}...")
        
        # Initialize network
        nn = SimpleNeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1)
        
        # Training parameters
        epochs = 100
        batch_size = 32
        train_losses = []
        test_losses = []
        learning_rates = []
        
        # Training loop
        for epoch in range(epochs):
            # Update learning rate
            if hasattr(scheduler, 'step'):
                lr = scheduler.step(epoch)
            else:
                lr = scheduler(epoch)
            
            nn.learning_rate = lr
            learning_rates.append(lr)
            
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
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'learning_rates': learning_rates
        }
        
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Test Loss: {test_losses[-1]:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Learning rate schedules
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['learning_rates'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True)
    
    # Training loss comparison
    plt.subplot(1, 3, 2)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Test loss comparison
    plt.subplot(1, 3, 3)
    for name, result in results.items():
        plt.plot(result['test_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
scheduler_results = demonstrate_learning_rate_scheduling()
```

## Gradient Clipping

### Implementation

```python
def clip_gradients(gradients, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for key in gradients:
            gradients[key] *= clip_coef
    
    return gradients

def demonstrate_gradient_clipping():
    """Demonstrate gradient clipping"""
    print("\n=== Gradient Clipping Demo ===")
    
    # Generate data with high variance to cause gradient explosion
    np.random.seed(42)
    X = np.random.randn(1000, 10) * 10  # High variance features
    y = np.random.randint(0, 2, 1000)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with and without gradient clipping
    results = {}
    
    for use_clipping in [False, True]:
        name = "With Clipping" if use_clipping else "Without Clipping"
        print(f"\nTraining {name}...")
        
        # Initialize network
        nn = SimpleNeuralNetwork(input_size=10, hidden_size=20, output_size=1, learning_rate=0.1)
        
        # Training parameters
        epochs = 50
        batch_size = 32
        train_losses = []
        test_losses = []
        gradient_norms = []
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train_scaled))
            X_shuffled = X_train_scaled[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            epoch_grad_norm = 0
            
            # Mini-batch training
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size].reshape(-1, 1)
                
                # Forward pass
                output = nn.forward(batch_X)
                
                # Backward pass
                dW1, db1, dW2, db2 = nn.backward(batch_X, batch_y, output)
                
                # Compute gradient norm
                grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
                total_norm = 0
                for grad in grads.values():
                    total_norm += np.sum(grad ** 2)
                total_norm = np.sqrt(total_norm)
                epoch_grad_norm += total_norm
                
                # Apply gradient clipping if enabled
                if use_clipping:
                    grads = clip_gradients(grads, max_norm=1.0)
                    dW1, db1, dW2, db2 = grads['W1'], grads['b1'], grads['W2'], grads['b2']
                
                # Update parameters
                nn.update_parameters(dW1, db1, dW2, db2)
                
                # Compute loss
                batch_loss = nn.compute_loss(batch_y, output)
                epoch_loss += batch_loss
                num_batches += 1
            
            # Average loss and gradient norm for epoch
            avg_train_loss = epoch_loss / num_batches
            avg_grad_norm = epoch_grad_norm / num_batches
            
            # Test loss
            test_output = nn.forward(X_test_scaled)
            test_loss = nn.compute_loss(y_test.reshape(-1, 1), test_output)
            
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            gradient_norms.append(avg_grad_norm)
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'gradient_norms': gradient_norms
        }
        
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Test Loss: {test_losses[-1]:.4f}")
        print(f"Final Gradient Norm: {avg_grad_norm:.4f}")
    
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
    
    # Gradient norm comparison
    plt.subplot(1, 3, 3)
    for name, result in results.items():
        plt.plot(result['gradient_norms'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
clipping_results = demonstrate_gradient_clipping()
```

## Summary

Stochastic Gradient Descent and optimization techniques provide the foundation for training neural networks:

1. **Basic SGD**: Simple but effective for many problems
2. **Momentum**: Accelerates convergence and reduces oscillations
3. **Adaptive Methods**: Adam, RMSprop adapt learning rates automatically
4. **Learning Rate Scheduling**: Improves convergence with decaying learning rates
5. **Gradient Clipping**: Prevents exploding gradients in deep networks

Key takeaways:

- **Mini-batch SGD**: Balance between computational efficiency and convergence
- **Optimizer Selection**: Choose based on problem characteristics
- **Learning Rate**: Critical hyperparameter for training success
- **Scheduling**: Improves convergence and final performance
- **Gradient Clipping**: Essential for training deep networks
- **Monitoring**: Track loss curves and gradient norms during training

Understanding these techniques provides a solid foundation for training neural networks effectively. 