# Words and Attention

## Overview

This guide covers the foundations of modern NLP and the attention mechanism, which underpins transformer architectures. Topics include text preprocessing, word embeddings, attention, and sequence modeling.

## 1. Natural Language Processing Basics

### Text Preprocessing
- Tokenization: Splitting text into words/tokens
- Lowercasing, removing punctuation, stopwords
- Example:
```python
import re
text = "Transformers are revolutionizing NLP!"
tokens = re.findall(r'\b\w+\b', text.lower())
print(tokens)  # ['transformers', 'are', 'revolutionizing', 'nlp']
```

### Word Embeddings
- **Word2Vec, GloVe, FastText**: Map words to dense vectors
- **Why embeddings?**: Capture semantic similarity
- **Mathematical view**: Each word $w$ is mapped to $\mathbf{v}_w \in \mathbb{R}^d$

#### Example: Training Word2Vec (using gensim)
```python
from gensim.models import Word2Vec
sentences = [["deep", "learning", "is", "fun"], ["transformers", "are", "powerful"]]
model = Word2Vec(sentences, vector_size=50, min_count=1)
print(model.wv["transformers"])
```

## 2. Attention Mechanisms

### Motivation
- RNNs/LSTMs struggle with long-range dependencies
- Attention allows the model to focus on relevant parts of the input

### Scaled Dot-Product Attention
Given queries $Q$, keys $K$, values $V$:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

- $Q, K, V$ are matrices (batch, seq_len, d_k)
- $d_k$ is the key dimension

#### Example: Simple Attention in NumPy
```python
import numpy as np
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(weights, V)
```

### Multi-Head Attention
- Multiple attention heads allow the model to jointly attend to information from different representation subspaces

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```

## 3. Sequence Modeling

### RNNs, LSTMs, GRUs
- Process sequences step by step
- Suffer from vanishing gradients

### Attention vs. RNNs
- Attention enables parallel computation and better long-range modeling

## 4. Text Classification Example

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
texts = ["I love transformers", "Deep learning is amazing"]
labels = [1, 1]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
clf = LogisticRegression().fit(X, labels)
print(clf.predict(vectorizer.transform(["transformers are great"])))
```

## 5. Applications
- Sentiment analysis
- Topic classification
- Machine translation
- Question answering

## Summary
- Embeddings map words to vectors
- Attention enables flexible context modeling
- Sequence models process text, but attention/transformers are now state-of-the-art 