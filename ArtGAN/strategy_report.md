# Task 1: Convolutional-Recurrent Architectures — Strategy Discussion

## 1. Problem Statement

The goal is to build a model that classifies artworks by **Style** (27 classes), **Artist** (23 classes), and **Genre** (10 classes) simultaneously using a convolutional-recurrent architecture on the WikiArt/ArtGAN dataset.

This is a **multi-label, multi-class** image classification problem with the unique constraint that the architecture must combine both CNN and RNN components.

## 2. Architecture Design — Why CNN-RNN?

### 2.1 Why Not a Pure CNN?

A standard CNN (e.g., ResNet50 alone) would be the typical choice for image classification. However, the task explicitly requires a convolutional-recurrent approach. Beyond this requirement, there are genuine advantages:

- **Spatial Sequences Matter in Art**: Artistic style is not just about *what* objects are present but *how* they are arranged and rendered across the canvas. A recurrent network processing spatial features as a sequence can capture relationships between different regions of the painting.
- **Multi-label Synergy**: Style, genre, and artist are correlated labels. An RNN processing shared spatial features allows the model to learn joint representations that benefit all three classification heads.

### 2.2 Architecture: ResNet50 + Bidirectional LSTM

Our `ConvRNNHybrid` model has four key components:

```
Input Image (224×224×3)
    │
    ▼
┌─────────────────────┐
│  CNN Backbone        │  ResNet50 (pretrained on ImageNet)
│  (Feature Extractor) │  Output: 2048 × 7 × 7 feature maps
└─────────────────────┘
    │
    ▼  Reshape to sequence: (49 steps × 2048 features)
┌─────────────────────┐
│  Bidirectional LSTM  │  2-layer, hidden_dim=512
│  (Sequence Processor)│  Captures spatial dependencies
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Attention Mechanism │  Learns which spatial regions
│                      │  are most informative
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Classification      │  3 independent heads:
│  Heads               │  Style (27), Artist (23), Genre (10)
└─────────────────────┘
```

### 2.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ResNet50 backbone** | Strong transfer learning from ImageNet; deep enough for fine art features |
| **Freeze early layers** | Only fine-tune layer3 + layer4 to prevent catastrophic forgetting on 11K images |
| **BiLSTM (not GRU)** | Bidirectional processing captures both forward and backward spatial context; LSTM gates handle long-range dependencies better |
| **Spatial Attention** | Not all 49 spatial regions contribute equally — attention learns to focus on style-defining regions |
| **Deeper classification heads** | Two-layer heads (Linear → ReLU → Dropout → Linear) allow for non-linear task-specific transformations |
| **Shared backbone** | All three tasks share CNN + RNN features, enabling transfer learning between tasks |

## 3. Training Strategy

### 3.1 Differential Learning Rates
- **Backbone (ResNet50)**: Learning rate = 1e-5 (prevents forgetting pretrained features)
- **New layers (LSTM + Heads)**: Learning rate = 1e-3 (faster learning for randomly initialized parameters)

### 3.2 Regularization Techniques
- **Dropout (30%)**: Applied after RNN and within classification heads
- **Label Smoothing (0.1)**: Prevents overconfident predictions, improves generalization
- **Weight Decay (1e-4)**: L2 regularization via AdamW optimizer
- **Data Augmentation**: ColorJitter, RandomRotation, RandomAffine, RandomErasing

### 3.3 Learning Rate Schedule
- **Cosine Annealing with Warm Restarts** (T_0=10, T_mult=2)
- Gradually reduces learning rate, allowing the model to settle into better minima

### 3.4 Training Efficiency
- **Mixed Precision (FP16)**: Reduces GPU memory usage by ~50%, enables larger batch sizes
- **Gradient Clipping** (max_norm=1.0): Prevents exploding gradients in the LSTM
- **Early Stopping** (patience=7): Automatically stops training when validation loss plateaus

## 4. Evaluation Metrics

### 4.1 Why Multiple Metrics?
A single metric like accuracy can be misleading, especially with imbalanced class distributions. We use a comprehensive set:

| Metric | Purpose |
|--------|---------|
| **Top-1 Accuracy** | Standard classification accuracy |
| **Top-3/5 Accuracy** | Shows if the correct class is among the model's best guesses — critical for subjective tasks like art classification where styles overlap |
| **Macro Precision** | How often the model's predictions are correct, averaged equally across all classes |
| **Macro Recall** | How many actual examples of each class the model finds |
| **Macro F1-Score** | Harmonic mean of precision and recall — single balanced metric |
| **Confusion Matrix** | Visualizes which classes get confused, revealing systematic patterns |

### 4.2 Results

| Task | Top-1 Acc | Top-3 Acc | Top-5 Acc | Macro F1 |
|------|-----------|-----------|-----------|----------|
| **Style** | 80.5% | 94.7% | 97.9% | 36.8% |
| **Artist** | 78.6% | 91.9% | 95.3% | 76.5% |
| **Genre** | 77.4% | 96.6% | 99.1% | 66.8% |

**Key Observations**:
- **Top-5 accuracy is 95-99%** across all tasks, meaning the correct answer is almost always in the model's top predictions.
- **Style Macro F1 is lower** (36.8%) despite high accuracy because some rare styles (Action Painting, Analytical Cubism) have very few validation samples, dragging down the macro average.
- **Artist has the highest F1** (76.5%) because each artist has a distinctive visual signature that the model learns well.

## 5. Outlier Analysis

The task specifically asks to *"find outliers, e.g. paintings that do not fit a particular artist or genre despite their assignment."*

### 5.1 Methodology
We identify outliers as paintings that the model **misclassifies with high confidence** (>85%). If the model is very confident that a painting belongs to a different category than its label, it suggests the painting shares visual characteristics with that predicted category.

### 5.2 Key Findings

1. **Cross-style paintings**: Several Realism paintings were classified as Art Nouveau with >97% confidence, suggesting some Realist painters occasionally employed Art Nouveau visual elements.

2. **Artist style similarities**: Boris Kustodiev paintings were confidently classified as John Singer Sargent works, revealing that these two artists share similar painterly techniques despite working in different artistic contexts.

3. **Genre boundary ambiguity**: Landscape paintings frequently confused with Cityscape (96% confidence), which makes intuitive sense — many paintings contain both urban and natural elements.

These findings demonstrate that the model has learned meaningful visual features and can identify genuine artistic ambiguities in the dataset.

## 6. Future Improvements

- **Class-weighted loss**: Address class imbalance to improve F1 for rare styles
- **Transformer attention**: Replace LSTM with a Vision Transformer for non-sequential spatial reasoning
- **Larger image resolution**: 299×299 or 384×384 for capturing fine brushwork details
- **Cross-task loss weighting**: Learn optimal weight for each classification head
