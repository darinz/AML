"""
Modern Deep Learning & Transformers Examples

This file contains examples for:
- Word embeddings
- Attention mechanisms
- Transformer blocks
- BERT/GPT demo
- Vision Transformer patching
- CLIP contrastive loss
- Prompt engineering
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Word Embeddings Example
sentences = [["deep", "learning", "is", "fun"], ["transformers", "are", "powerful"]]
vocab = {w for s in sentences for w in s}
vocab = {w: i for i, w in enumerate(sorted(vocab))}
embedding_dim = 8
embeddings = np.random.randn(len(vocab), embedding_dim)
print("Word embedding for 'transformers':", embeddings[vocab["transformers"]])

# 2. Scaled Dot-Product Attention
Q = np.random.randn(2, 4, 8)  # (batch, seq_len, d_k)
K = np.random.randn(2, 4, 8)
V = np.random.randn(2, 4, 8)
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(weights, V)
attn_out = scaled_dot_product_attention(Q, K, V)
print("Attention output shape:", attn_out.shape)

# 3. Transformer Block (PyTorch)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# 4. BERT/GPT Demo (using HuggingFace Transformers)
try:
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    print(classifier("Transformers are amazing!"))
except ImportError:
    print("Install transformers library for BERT/GPT demo.")

# 5. Vision Transformer (ViT) Patch Embedding
img = torch.randn(1, 3, 32, 32)
patch_size = 8
patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
patches = patches.contiguous().view(1, 3, -1, patch_size, patch_size)
print("ViT patches shape:", patches.shape)

# 6. CLIP Contrastive Loss Example
z_img = np.random.randn(4, 16)
z_txt = np.random.randn(4, 16)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
sims = np.array([[cosine_similarity(z_img[i], z_txt[j]) for j in range(4)] for i in range(4)])
tau = 0.07
loss = -np.log(np.exp(np.diag(sims)/tau) / np.sum(np.exp(sims/tau), axis=1)).mean()
print("CLIP contrastive loss:", loss)

# 7. Prompt Engineering Example
prompt = "Text: 'I love this movie!'\nSentiment:"
print("Prompt example:", prompt) 