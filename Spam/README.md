# YouTube Spam Comment Detector

A high-performance spam detection model for YouTube comments using ALBERT (A Lite BERT) transformer architecture, achieving **99% test accuracy**.

## Overview

This project implements a binary classification system to detect spam comments on YouTube videos. The model is trained on comments from popular music videos across multiple artists and can distinguish between legitimate comments and spam with exceptional accuracy.

## Performance

- **Test Accuracy**: 99.00%
- **Model**: ALBERT (albert-base-v2)
- **Training Time**: ~1000 steps across 3 epochs
- **Dataset Size**: Multi-artist YouTube comments dataset

## Why ALBERT?

### Model Selection Rationale

ALBERT (A Lite BERT) was chosen for this spam detection task for several compelling reasons:

#### 1. **Parameter Efficiency**
- ALBERT uses **parameter sharing** across layers, resulting in 18x fewer parameters than BERT-base
- Smaller model size (~45MB) makes it practical for deployment
- Faster inference time suitable for real-time comment moderation

#### 2. **Strong Performance on Text Classification**
- Pre-trained on massive text corpora, giving it robust language understanding
- Excels at short-text classification tasks (like comments)
- Proven track record on sentiment analysis and spam detection benchmarks

#### 3. **SentencePiece Tokenization**
- Breaks words into sub-units (e.g., "playing" → "play" + "ing")
- Handles **unseen words** and misspellings common in spam comments
- Better vocabulary coverage with smaller vocabulary size (30K tokens)
- Critical for internet comments with slang, typos, and creative spelling

#### 4. **Memory Efficiency**
- Cross-layer parameter sharing reduces memory footprint
- Enables training on consumer-grade hardware
- Supports larger batch sizes for faster training

#### 5. **Factorized Embedding Parameterization**
- Separates embedding size from hidden layer size
- More efficient representation of word meanings
- Better generalization on smaller datasets

### Alternatives Considered

| Model | Pros | Cons | Why Not Selected |
|-------|------|------|------------------|
| **BERT** | Strong baseline, well-documented | 110M parameters, slower inference | Too large for deployment needs |
| **DistilBERT** | Faster than BERT, 40% smaller | Less accurate than ALBERT | Lower accuracy trade-off |
| **RoBERTa** | Superior performance on benchmarks | 125M parameters, high compute cost | Overkill for binary classification |
| **Naive Bayes/SVM** | Very fast, interpretable | Poor handling of context and semantics | Insufficient for nuanced spam |

**ALBERT strikes the optimal balance between accuracy, efficiency, and deployability for this use case.**

## Dataset

**Source**: [YouTube Spam Collection Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/images) (Kaggle)

**Artists Covered**:
1. Psy
2. Katy Perry
3. LMFAO
4. Eminem
5. Shakira

**Features**:
- `CONTENT`: Comment text
- `CLASS`: Binary label (0 = Ham/Legitimate, 1 = Spam)
- `COMMENT_ID`: Unique identifier
- `AUTHOR`: Comment author
- `DATE`: Timestamp

**Dataset Split**:
- Training: All data minus 400 samples
- Test: 400 samples (stratified)
- Shuffled with seed=42 for reproducibility

## Technical Architecture

### Model Details

```
Architecture: ALBERT-base-v2
- Embedding Size: 128
- Hidden Size: 768
- Attention Heads: 12
- Layers: 12 (with parameter sharing)
- Parameters: ~12M (vs BERT's 110M)
- Vocabulary: 30,000 tokens (SentencePiece)
```

### Tokenization Strategy

```python
- Max Sequence Length: 128 tokens
- Padding: Max length (shorter sequences padded with zeros)
- Truncation: Enabled (handles long comments gracefully)
- Tokenizer: SentencePiece (subword tokenization)
```

**Why 128 tokens?**
- Covers 95%+ of YouTube comments without truncation
- Balances context preservation with computational efficiency
- Standard length for short-text classification tasks

### Training Configuration

```python
Training Hyperparameters:
- Learning Rate: 2e-5 (standard for fine-tuning transformers)
- Batch Size: 8 per device
- Epochs: 3
- Max Steps: 1000
- Weight Decay: 0.01 (L2 regularization)
- Optimizer: AdamW (implied by Trainer)
- Mixed Precision: FP16 (faster training, lower memory)
```

**Key Features**:
- **Streaming**: Handles datasets larger than RAM
- **Early Stopping**: Loads best model based on validation accuracy
- **Checkpoint Management**: Saves top 2 models, auto-cleanup
- **Evaluation Frequency**: Every 100 steps for monitoring

## Installation

### Requirements

```bash
pip install torch transformers datasets evaluate kagglehub numpy
```

### System Requirements

- **GPU**: CUDA-compatible GPU recommended (falls back to CPU)
- **RAM**: 8GB minimum
- **Storage**: 500MB for model and cache
- **Python**: 3.8+


## Usage

### Training the Model

```bash
python model.py
```

The script will:
1. Download the dataset from Kaggle
2. Load and preprocess all 5 CSV files
3. Train the ALBERT model
4. Evaluate on test set
5. Save the final model to `./Spam/final_model/`

### Making Predictions

```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# Load trained model
model = AlbertForSequenceClassification.from_pretrained("./Spam/final_model")
tokenizer = AlbertTokenizer.from_pretrained("./Spam/final_model")

# Predict on new comment
comment = "Check out my channel for free gifts!!!"
inputs = tokenizer(comment, return_tensors="pt", max_length=128, 
                   truncation=True, padding="max_length")

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
print("SPAM" if prediction == 1 else "LEGITIMATE")
```

## Project Structure

```
.
├── model.py           # Main training script
├── results/               # Training checkpoints
├── final_model/           # Saved model and tokenizer
└── README.md                  
```

## Key Implementation Features

### 1. **Streaming Data Processing**
```python
data = load_dataset("csv", data_files=file_path, split="train", streaming=True)
```
- Enables training on datasets larger than available RAM
- Memory-efficient for production environments
- Crucial for scaling to larger comment datasets

### 2. **Dataset Interleaving**
```python
full_dataset = interleave_datasets(raw_streams)
```
- Mixes comments from all 5 artists
- Prevents artist-specific bias
- Creates balanced, diverse training data

### 3. **Mixed Precision Training (FP16)**
```python
fp16=True
```
- 2x faster training on compatible GPUs
- 50% memory reduction
- Negligible accuracy impact

### 4. **Robust Evaluation Strategy**
```python
eval_strategy="steps"
eval_steps=100
```
- Continuous monitoring during training
- Early detection of overfitting
- Automatic best model selection

## Training Output

```
Start training ...
Step 100: train_loss=0.234, eval_accuracy=0.962
Step 200: train_loss=0.156, eval_accuracy=0.978
Step 300: train_loss=0.098, eval_accuracy=0.985
...
Step 1000: train_loss=0.021, eval_accuracy=0.990

Test Output:
Test Accuracy: 99.00%
```

## Model Insights

### What Makes a Comment Spam?

The model learns to identify patterns such as:
- **Promotional language**: "Check out my channel", "Subscribe here"
- **Excessive punctuation**: "!!!!", "????"
- **Call-to-action**: "Click", "Visit", "Free"
- **Repetitive text**: Multiple identical phrases
- **Suspicious links**: Shortened URLs, external redirects
- **Generic comments**: "Nice video", "Great" (when contextually odd)

### False Positives/Negatives

**Potential False Positives**:
- Enthusiastic legitimate comments with many exclamation marks
- Users genuinely sharing their channels

**Potential False Negatives**:
- Sophisticated spam that mimics natural conversation
- Context-specific spam (e.g., crypto scams during trending events)

## Performance Optimization

### For Faster Training
```python
per_device_train_batch_size=16  # Increase if GPU memory allows
max_steps=500                    # Reduce steps (may impact accuracy)
fp16=True                        # Already enabled
```

### For Better Accuracy
```python
num_train_epochs=5               # More epochs
learning_rate=1e-5               # Lower learning rate
weight_decay=0.02                # Stronger regularization
```

### For Deployment
```python
# Export to ONNX for faster inference
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "./final_model",
    export=True
)
```

## Future Improvements

1. **Multi-Class Classification**: Categorize spam types (promotional, malicious, bot)
2. **Active Learning**: Continuously improve with user feedback
3. **Multilingual Support**: Extend to non-English comments
4. **Temporal Features**: Incorporate comment timing patterns
5. **Ensemble Methods**: Combine with rule-based filters
6. **Explainability**: Add attention visualization for interpretability
7. **Online Learning**: Real-time model updates

## License

This project is licensed under the MIT License. The dataset is subject to Kaggle's terms of service.

## Acknowledgments

- **Dataset**: Lakshmi Narayanan ([Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/images))
- **Model**: Google Research (ALBERT)
- **Framework**: Hugging Face Transformers
- **Compute**: Kaggle / Local GPU

---

**Model Performance**: 99% Test Accuracy | **Last Updated**: December 2025