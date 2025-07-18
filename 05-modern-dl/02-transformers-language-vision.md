# Transformers in Language and Vision

## Overview

Transformers are the backbone of modern NLP and are increasingly used in vision. This guide covers the transformer architecture, positional encoding, BERT/GPT, Vision Transformers (ViT), and cross-modal models.

## 1. Transformer Architecture

### Encoder-Decoder Structure
- **Encoder**: Processes input sequence
- **Decoder**: Generates output sequence (e.g., translation)

### Multi-Head Self-Attention
- Each token attends to all others in the sequence
- Multiple heads capture different relationships

#### Mathematical Formulation
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

#### Transformer Block
- LayerNorm, Multi-Head Attention, Feedforward, Residual connections

### Implementation (PyTorch-style pseudocode)
```python
import torch
import torch.nn as nn
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
```

## 2. Positional Encoding

Transformers lack recurrence, so positional information is added:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```
```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

## 3. BERT and GPT

- **BERT**: Bidirectional encoder, pre-trained with masked language modeling
- **GPT**: Decoder-only, autoregressive text generation
- **Fine-tuning**: Adapt pre-trained models to downstream tasks

## 4. Vision Transformers (ViT)

- Images are split into patches, each patch is linearly embedded
- Sequence of patch embeddings is processed like words in NLP

#### Example: Patch Embedding
```python
import torch
img = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
patch_size = 8
patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
patches = patches.contiguous().view(1, 3, -1, patch_size, patch_size)
print(patches.shape)  # (1, 3, num_patches, 8, 8)
```

## 5. Cross-Modal Transformers
- Models like CLIP process both text and images
- Learn joint representations for multiple modalities

## Applications
- Machine translation
- Text generation
- Image captioning
- Multimodal search

## Summary
- Transformers use self-attention and positional encoding
- BERT/GPT are pre-trained language models
- ViT applies transformers to vision
- Cross-modal models bridge language and vision 