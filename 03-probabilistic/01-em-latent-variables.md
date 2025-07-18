# Expectation-Maximization & Latent Variables

## Overview

Expectation-Maximization (EM) is a powerful iterative algorithm for finding maximum likelihood estimates in models with latent (hidden) variables. This guide covers the mathematical foundations, implementation from scratch, and applications to various probabilistic models.

## Key Concepts

### 1. What is the EM Algorithm?

The EM algorithm is an iterative method for finding maximum likelihood estimates when the data has missing values or latent variables. It alternates between two steps:

1. **E-step (Expectation)**: Compute the expected value of the log-likelihood function with respect to the current estimate of the parameters
2. **M-step (Maximization)**: Find the parameters that maximize the expected log-likelihood

### 2. Mathematical Foundation

The EM algorithm maximizes the log-likelihood function:

```math
\log p(\mathbf{X} | \boldsymbol{\theta}) = \log \sum_{\mathbf{Z}} p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})
```

Where:
- $`\mathbf{X}`$ is the observed data
- $`\mathbf{Z}`$ are the latent variables
- $`\boldsymbol{\theta}`$ are the model parameters

The algorithm works by introducing a distribution $`q(\mathbf{Z})`$ over the latent variables and using Jensen's inequality:

```math
\log p(\mathbf{X} | \boldsymbol{\theta}) \geq \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})}{q(\mathbf{Z})}
```

### 3. EM Algorithm Steps

**E-step**: Compute the posterior distribution of latent variables:

```math
q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \boldsymbol{\theta}^{(t)})
```

**M-step**: Maximize the expected complete log-likelihood:

```math
\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} \sum_{\mathbf{Z}} q(\mathbf{Z}) \log p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})
```

## Gaussian Mixture Models (GMM)

### Mathematical Foundation

A Gaussian Mixture Model assumes the data is generated from a mixture of $`K`$ Gaussian distributions:

```math
p(\mathbf{x} | \boldsymbol{\theta}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
```

Where:
- $`\pi_k`$ are the mixing coefficients ($`\sum_{k=1}^{K} \pi_k = 1`$)
- $`\boldsymbol{\mu}_k`$ are the means of the Gaussian components
- $`\boldsymbol{\Sigma}_k`$ are the covariance matrices

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import seaborn as sns

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
    from sklearn.metrics import adjusted_rand_score, silhouette_score
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

# Run demonstration
gmm_model, X_data, y_true, y_pred = demonstrate_gmm()
```

## Hidden Markov Models (HMM)

### Mathematical Foundation

Hidden Markov Models model sequential data where the underlying state sequence is hidden. The model consists of:

1. **Initial state distribution**: $`\pi_i = P(Z_1 = i)`$
2. **Transition matrix**: $`A_{ij} = P(Z_{t+1} = j | Z_t = i)`$
3. **Emission probabilities**: $`B_{ik} = P(X_t = k | Z_t = i)`$

### Implementation

```python
class HiddenMarkovModel:
    """Hidden Markov Model implementation using EM algorithm"""
    
    def __init__(self, n_states=3, n_observations=None, max_iter=100, tolerance=1e-6):
        self.n_states = n_states
        self.n_observations = n_observations
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.pi = None  # Initial state distribution
        self.A = None   # Transition matrix
        self.B = None   # Emission matrix
        self.log_likelihood_history = []
        
    def _initialize_parameters(self, observations):
        """Initialize HMM parameters"""
        if self.n_observations is None:
            self.n_observations = len(np.unique(observations))
        
        # Initialize initial state distribution uniformly
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Initialize transition matrix randomly
        self.A = np.random.rand(self.n_states, self.n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)
        
        # Initialize emission matrix randomly
        self.B = np.random.rand(self.n_states, self.n_observations)
        self.B /= self.B.sum(axis=1, keepdims=True)
    
    def _forward_algorithm(self, observations):
        """Forward algorithm to compute alpha values"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        alpha[0] = self.pi * self.B[:, observations[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1] * self.A[:, j])
        
        return alpha
    
    def _backward_algorithm(self, observations):
        """Backward algorithm to compute beta values"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[T-1] = 1.0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :])
        
        return beta
    
    def _e_step(self, observations):
        """E-step: Compute gamma and xi"""
        T = len(observations)
        
        # Forward-backward algorithm
        alpha = self._forward_algorithm(observations)
        beta = self._backward_algorithm(observations)
        
        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        # Compute xi (transition probabilities)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                 self.B[j, observations[t+1]] * beta[t+1, j])
            xi[t] /= xi[t].sum()
        
        return gamma, xi
    
    def _m_step(self, observations, gamma, xi):
        """M-step: Update parameters"""
        T = len(observations)
        
        # Update initial state distribution
        self.pi = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                self.A[i, j] = numerator / denominator
        
        # Update emission matrix
        for i in range(self.n_states):
            for k in range(self.n_observations):
                numerator = np.sum(gamma[observations == k, i])
                denominator = np.sum(gamma[:, i])
                self.B[i, k] = numerator / denominator
    
    def _compute_log_likelihood(self, observations):
        """Compute log-likelihood"""
        alpha = self._forward_algorithm(observations)
        return np.log(np.sum(alpha[-1]))
    
    def fit(self, observations):
        """Fit HMM using EM algorithm"""
        observations = np.array(observations)
        
        # Initialize parameters
        self._initialize_parameters(observations)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            gamma, xi = self._e_step(observations)
            
            # M-step
            self._m_step(observations, gamma, xi)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(observations)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                if abs(log_likelihood - self.log_likelihood_history[-2]) < self.tolerance:
                    break
        
        return self
    
    def predict_states(self, observations):
        """Predict hidden states using Viterbi algorithm"""
        observations = np.array(observations)
        T = len(observations)
        
        # Viterbi algorithm
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        delta[0] = np.log(self.pi) + np.log(self.B[:, observations[0]])
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t-1] + np.log(self.A[:, j])
                psi[t, j] = np.argmax(temp)
                delta[t, j] = temp[psi[t, j]] + np.log(self.B[j, observations[t]])
        
        # Backward pass
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def generate_sequence(self, length=100):
        """Generate a sequence from the fitted HMM"""
        observations = []
        states = []
        
        # Sample initial state
        state = np.random.choice(self.n_states, p=self.pi)
        states.append(state)
        
        # Generate sequence
        for t in range(length):
            # Sample observation
            observation = np.random.choice(self.n_observations, p=self.B[state])
            observations.append(observation)
            
            # Sample next state
            if t < length - 1:
                state = np.random.choice(self.n_states, p=self.A[state])
                states.append(state)
        
        return np.array(observations), np.array(states)

def demonstrate_hmm():
    """Demonstrate Hidden Markov Model"""
    print("\n=== Hidden Markov Model Demo ===")
    
    # Generate synthetic sequence data
    np.random.seed(42)
    
    # Define true HMM parameters
    n_states = 3
    n_observations = 4
    
    pi_true = np.array([0.3, 0.3, 0.4])
    A_true = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    B_true = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.05, 0.8, 0.1, 0.05],
        [0.05, 0.05, 0.1, 0.8]
    ])
    
    # Generate sequence
    sequence_length = 1000
    observations = []
    true_states = []
    
    state = np.random.choice(n_states, p=pi_true)
    for t in range(sequence_length):
        observation = np.random.choice(n_observations, p=B_true[state])
        observations.append(observation)
        true_states.append(state)
        
        if t < sequence_length - 1:
            state = np.random.choice(n_states, p=A_true[state])
    
    observations = np.array(observations)
    true_states = np.array(true_states)
    
    # Fit HMM
    hmm = HiddenMarkovModel(n_states=3, n_observations=4, max_iter=100)
    hmm.fit(observations)
    
    # Predict states
    predicted_states = hmm.predict_states(observations)
    
    # Evaluate
    accuracy = np.mean(predicted_states == true_states)
    print(f"State prediction accuracy: {accuracy:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # True states
    plt.subplot(1, 3, 1)
    plt.plot(true_states[:100], 'b-', label='True States')
    plt.title('True Hidden States')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    
    # Predicted states
    plt.subplot(1, 3, 2)
    plt.plot(predicted_states[:100], 'r-', label='Predicted States')
    plt.title('Predicted Hidden States')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    
    # Log-likelihood convergence
    plt.subplot(1, 3, 3)
    plt.plot(hmm.log_likelihood_history)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('EM Convergence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return hmm, observations, true_states, predicted_states

# Run demonstration
hmm_model, observations, true_states, predicted_states = demonstrate_hmm()
```

## Latent Dirichlet Allocation (LDA)

### Mathematical Foundation

LDA is a generative probabilistic model for collections of discrete data such as text corpora. It assumes documents are mixtures of topics, and topics are distributions over words.

The generative process:
1. For each document $`d`$:
   - Draw topic distribution $`\theta_d \sim \text{Dir}(\alpha)`$
   - For each word $`w_{dn}`$:
     - Draw topic $`z_{dn} \sim \text{Mult}(\theta_d)`$
     - Draw word $`w_{dn} \sim \text{Mult}(\beta_{z_{dn}})`$

### Implementation

```python
class LatentDirichletAllocation:
    """Latent Dirichlet Allocation implementation using variational inference"""
    
    def __init__(self, n_topics=5, alpha=0.1, beta=0.01, max_iter=100, tolerance=1e-6):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.topic_word_dist = None
        self.doc_topic_dist = None
        
    def _initialize_parameters(self, documents, vocab_size):
        """Initialize LDA parameters"""
        # Initialize topic-word distribution randomly
        self.topic_word_dist = np.random.dirichlet([self.beta] * vocab_size, size=self.n_topics)
        
        # Initialize document-topic distribution
        self.doc_topic_dist = np.random.dirichlet([self.alpha] * self.n_topics, size=len(documents))
    
    def _e_step(self, documents):
        """E-step: Update document-topic distributions"""
        n_docs = len(documents)
        
        for d in range(n_docs):
            doc = documents[d]
            n_words = len(doc)
            
            # Initialize gamma for this document
            gamma = np.ones((n_words, self.n_topics)) / self.n_topics
            
            # Iterative updates
            for iteration in range(10):
                for n in range(n_words):
                    word = doc[n]
                    
                    # Update gamma
                    gamma[n] = (self.doc_topic_dist[d] * 
                               self.topic_word_dist[:, word])
                    gamma[n] /= np.sum(gamma[n])
            
            # Update document-topic distribution
            self.doc_topic_dist[d] = self.alpha + np.sum(gamma, axis=0)
    
    def _m_step(self, documents, vocab_size):
        """M-step: Update topic-word distributions"""
        n_docs = len(documents)
        
        # Reset topic-word distribution
        self.topic_word_dist = np.zeros((self.n_topics, vocab_size))
        
        for d in range(n_docs):
            doc = documents[d]
            
            # Compute gamma for this document
            gamma = np.ones((len(doc), self.n_topics)) / self.n_topics
            
            for iteration in range(10):
                for n in range(len(doc)):
                    word = doc[n]
                    gamma[n] = (self.doc_topic_dist[d] * 
                               self.topic_word_dist[:, word])
                    gamma[n] /= np.sum(gamma[n])
            
            # Update topic-word distribution
            for n, word in enumerate(doc):
                self.topic_word_dist[:, word] += gamma[n]
        
        # Normalize
        self.topic_word_dist += self.beta
        self.topic_word_dist /= self.topic_word_dist.sum(axis=1, keepdims=True)
    
    def fit(self, documents, vocab_size):
        """Fit LDA model"""
        # Initialize parameters
        self._initialize_parameters(documents, vocab_size)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            self._e_step(documents)
            
            # M-step
            self._m_step(documents, vocab_size)
        
        return self
    
    def get_topics(self, vocab, top_words=10):
        """Get top words for each topic"""
        topics = []
        for topic_idx in range(self.n_topics):
            top_indices = np.argsort(self.topic_word_dist[topic_idx])[-top_words:][::-1]
            topic_words = [vocab[idx] for idx in top_indices]
            topics.append(topic_words)
        return topics
    
    def transform(self, documents):
        """Transform documents to topic distributions"""
        n_docs = len(documents)
        doc_topics = np.zeros((n_docs, self.n_topics))
        
        for d in range(n_docs):
            doc = documents[d]
            gamma = np.ones((len(doc), self.n_topics)) / self.n_topics
            
            # Iterative updates
            for iteration in range(10):
                for n in range(len(doc)):
                    word = doc[n]
                    gamma[n] = (self.doc_topic_dist[d] * 
                               self.topic_word_dist[:, word])
                    gamma[n] /= np.sum(gamma[n])
            
            doc_topics[d] = np.sum(gamma, axis=0)
            doc_topics[d] /= np.sum(doc_topics[d])
        
        return doc_topics

def demonstrate_lda():
    """Demonstrate Latent Dirichlet Allocation"""
    print("\n=== Latent Dirichlet Allocation Demo ===")
    
    # Generate synthetic document data
    np.random.seed(42)
    
    # Define vocabulary
    vocab = ['machine', 'learning', 'data', 'algorithm', 'model', 'neural', 'network',
             'deep', 'computer', 'vision', 'natural', 'language', 'processing',
             'statistics', 'probability', 'optimization', 'gradient', 'descent',
             'classification', 'regression', 'clustering', 'dimensionality', 'reduction']
    
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Generate documents
    n_docs = 100
    documents = []
    
    # Topic 1: Machine Learning
    topic1_words = ['machine', 'learning', 'algorithm', 'model', 'data']
    # Topic 2: Deep Learning
    topic2_words = ['neural', 'network', 'deep', 'gradient', 'descent']
    # Topic 3: Computer Vision
    topic3_words = ['computer', 'vision', 'image', 'processing', 'classification']
    
    topics = [topic1_words, topic2_words, topic3_words]
    
    for doc in range(n_docs):
        doc_length = np.random.randint(10, 30)
        doc_words = []
        
        for word in range(doc_length):
            # Choose topic
            topic = np.random.choice(3)
            # Choose word from topic
            word = np.random.choice(topics[topic])
            doc_words.append(word_to_idx[word])
        
        documents.append(doc_words)
    
    # Fit LDA
    lda = LatentDirichletAllocation(n_topics=3, alpha=0.1, beta=0.01, max_iter=50)
    lda.fit(documents, vocab_size)
    
    # Get topics
    topics = lda.get_topics(vocab, top_words=5)
    
    print("Learned Topics:")
    for i, topic in enumerate(topics):
        print(f"Topic {i+1}: {', '.join(topic)}")
    
    # Transform documents
    doc_topics = lda.transform(documents)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Document-topic distribution
    plt.subplot(1, 2, 1)
    plt.imshow(doc_topics[:20], cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title('Document-Topic Distribution')
    plt.xlabel('Topic')
    plt.ylabel('Document')
    
    # Topic-word distribution
    plt.subplot(1, 2, 2)
    plt.imshow(lda.topic_word_dist, cmap='Reds', aspect='auto')
    plt.colorbar()
    plt.title('Topic-Word Distribution')
    plt.xlabel('Word Index')
    plt.ylabel('Topic')
    
    plt.tight_layout()
    plt.show()
    
    return lda, documents, vocab, doc_topics

# Run demonstration
lda_model, documents, vocab, doc_topics = demonstrate_lda()
```

## Missing Data Imputation

### Mathematical Foundation

Missing data imputation using EM involves modeling the complete data distribution and iteratively updating missing values and parameters.

### Implementation

```python
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

# Run demonstration
imputer, X_complete, X_missing, X_imputed = demonstrate_missing_data_imputation()
```

## Summary

The EM algorithm and latent variable models provide powerful tools for:

1. **Handling Missing Data**: Imputing missing values using probabilistic models
2. **Clustering**: GMM provides soft clustering with probabilistic assignments
3. **Sequence Modeling**: HMM captures temporal dependencies in sequential data
4. **Topic Modeling**: LDA discovers latent topics in document collections
5. **Uncertainty Quantification**: Probabilistic models provide uncertainty estimates

Key takeaways:

- **EM Algorithm**: Iterative optimization for models with latent variables
- **GMM**: Flexible clustering with probabilistic interpretation
- **HMM**: Powerful for sequential data and time series
- **LDA**: Effective for topic modeling and document analysis
- **Missing Data**: EM provides principled approach to imputation
- **Convergence**: EM guarantees convergence to local optimum

Understanding these techniques provides a solid foundation for probabilistic modeling and unsupervised learning. 