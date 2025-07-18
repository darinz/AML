# Foundation Models

## Overview

Foundation models are large, pre-trained models (often transformers) that can be adapted to a wide range of tasks. This guide covers CLIP, GPT-3, scaling laws, prompt engineering, and ethical considerations.

## 1. CLIP: Vision-Language Foundation Model

- Trained to match images and text descriptions
- Uses contrastive loss: bring matching pairs closer, push mismatched apart

### Contrastive Loss
```math
L = -\log \frac{\exp(\text{sim}(x, y^+)/\tau)}{\sum_{y} \exp(\text{sim}(x, y)/\tau)}
```
- $\text{sim}(x, y)$: cosine similarity
- $\tau$: temperature parameter

#### Example: Cosine Similarity
```python
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

## 2. GPT-3 and Large Language Models

- **GPT-3**: 175B parameters, autoregressive text generation
- **Few-shot learning**: Model can perform tasks from a few examples in the prompt

### Prompt Engineering
- Carefully design prompts to elicit desired behavior
- Example prompt for sentiment analysis:
```
Text: "I love this movie!"\nSentiment:
```

## 3. Scaling Laws

- Model performance improves predictably with more data, parameters, and compute
- Empirical scaling law:
```math
\text{Loss} \propto N^{-\alpha}
```
- $N$: model size, $\alpha$: scaling exponent

## 4. Model Compression
- Quantization, pruning, distillation to reduce size and inference cost

## 5. Ethical Considerations
- Bias and fairness
- Responsible AI: transparency, accountability
- Risks of misuse (deepfakes, misinformation)

## Applications
- Text generation
- Image-text retrieval
- Multimodal search
- Few-shot and zero-shot learning

## Summary
- Foundation models are large, pre-trained, and adaptable
- CLIP bridges vision and language
- GPT-3 enables few-shot learning
- Scaling laws guide model growth
- Ethics are critical in deployment 