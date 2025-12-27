# Fake News Detection using BERT

A comprehensive Jupyter Notebook implementation for detecting fake news using BERT (Bidirectional Encoder Representations from Transformers), achieving **~82% test accuracy** through fine-tuning on a large-scale news dataset.

## Overview

This project provides an end-to-end machine learning pipeline for fake news detection, implemented entirely in a Jupyter Notebook. The notebook includes data exploration, preprocessing, model training, evaluation, and saves a production-ready model.

## Performance Metrics

- **Test Accuracy**: 82.34%
- **Precision**: 82% (macro average)
- **Recall**: 82% (macro average)
- **F1 Score**: 82% (macro average)

## Why BERT?

### Model Selection Rationale

**BERT (bert-base-uncased)** was chosen as the optimal architecture for this fake news detection task for several critical reasons:

#### 1. **Bidirectional Context Understanding**
- Unlike traditional models that read text left-to-right or right-to-left, BERT reads **both directions simultaneously**
- Captures nuanced meanings and context that are crucial for detecting subtle manipulation in fake news
- Example: Understanding "The bank is near the river" vs "I need to bank my money" requires bidirectional context

#### 2. **Pre-trained on Massive Text Corpus**
- Trained on 3.3 billion words (Wikipedia + BookCorpus)
- Already understands:
  - Grammar and sentence structure
  - Common facts and world knowledge
  - Reasoning patterns
  - Semantic relationships
- Requires minimal fine-tuning to specialize in fake news detection

#### 3. **Attention Mechanism**
- Identifies which words/phrases are most important for classification
- Can focus on suspicious patterns:
  - Sensational language: "BREAKING", "SHOCKING", "Scientists HATE this"
  - Conspiracy keywords
  - Logical inconsistencies
  - Emotional manipulation tactics

#### 4. **Handles Long Documents**
- Max sequence length: 256 tokens (in this implementation)
- Can process substantial article content
- Captures long-range dependencies between title, content, and context

#### 5. **Transfer Learning Excellence**
- Fine-tuning on domain-specific data (news articles) adapts the model effectively
- Strong performance with relatively small training epochs (2 epochs)
- Efficient training with frozen base layers and trainable classifier

#### 6. **Proven Track Record**
- Industry standard for text classification tasks
- Consistent performance across various NLP benchmarks
- Robust to adversarial content when properly fine-tuned

### Alternatives Considered

| Model | Pros | Cons | Why Not Selected |
|-------|------|------|------------------|
| **Traditional ML (TF-IDF + SVM)** | Fast, interpretable, simple | No context understanding, poor on nuanced fake news | Insufficient for sophisticated misinformation |
| **LSTM/GRU** | Handles sequences, faster than BERT | Unidirectional, smaller capacity | Lower accuracy, less robust |
| **DistilBERT** | 40% faster, 60% smaller | 3-5% accuracy drop | Accuracy is critical for this task |
| **RoBERTa** | More robust, better performance | 125M parameters, slower training | Marginal gains not worth compute cost |
| **GPT-2/3** | Strong language understanding | Unidirectional, designed for generation | Not optimized for classification |

**BERT provides the optimal balance between accuracy, training efficiency, and practical deployability for fake news detection.**

## Dataset

**Source**: [Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) (Kaggle)

### Data Structure

Two CSV files containing labeled news articles:
- **Fake.csv**: Fake news articles (label = 1)
- **True.csv**: Legitimate news articles (label = 0)

**Features**:
- `title`: Article headline
- `text`: Full article content
- `subject`: Topic category (not used in current implementation)

### Dataset Statistics

```
Total samples: 44,898
- Fake news: 23,481 (52.3%)
- True news: 21,417 (47.7%)

Split (from 50k sample):
- Training: 35,918 samples (80%)
- Validation: 4,490 samples (10%)
- Test: 4,490 samples (10%)
```

### Data Balance

The dataset is relatively balanced, which is ideal for training:
- **Fake**: 52.3%
- **True**: 47.7%

This prevents bias toward either class and ensures fair evaluation metrics.

## Technical Architecture

### Model Specifications

```
Architecture: BERT-base-uncased
- Parameters: 110M total
  - Base BERT: Frozen (for efficiency)
  - Classifier Head: Trainable
- Encoder Layers: 12
- Attention Heads: 12 per layer
- Hidden Size: 768
- Vocabulary: 30,522 WordPiece tokens
- Max Sequence Length: 256 tokens (configured)
```

### Training Strategy

#### Layer Freezing
```python
# Freeze base BERT layers to prevent catastrophic forgetting
for param in model.bert.parameters():
    param.requires_grad = False
```

**Why freeze base layers?**
- Preserves pre-trained language understanding
- Faster training (fewer parameters to update)
- Prevents overfitting on smaller dataset
- Focus learning on task-specific classifier

#### Regularization Techniques

```python
# Dropout to prevent overfitting
hidden_dropout_prob = 0.3
attention_probs_dropout_prob = 0.3

# Weight decay (L2 regularization)
weight_decay = 0.05
```

### Training Configuration

```python
Hyperparameters:
- Learning Rate: 1e-5 (conservative for fine-tuning)
- Batch Size: 16 per device
- Epochs: 2 (early stopping enabled)
- Weight Decay: 0.05 (L2 regularization)
- Warmup Ratio: 0.1 (gradual learning rate warmup)
- Optimizer: AdamW (built into Trainer)
- Mixed Precision: FP16 (faster training, lower memory)
```

**Key Training Features**:
- **Early Stopping**: Patience of 2 epochs to prevent overfitting
- **Best Model Loading**: Automatically saves model with highest validation accuracy
- **Warmup Schedule**: Gradually increases learning rate for stable training
- **Stratified Split**: Maintains class balance across train/validation/test sets

### Input Processing

```python
Tokenization:
- Max Length: 256 tokens
- Padding: Max length (fixed size)
- Truncation: Enabled (handles long articles)
- Text Field: "text" column only (titles not concatenated)
```

### Evaluation Metrics

```python
Metrics Computed:
- Accuracy: Overall correctness
- Precision (Macro): Per-class precision averaged
- Recall (Macro): Per-class recall averaged
- F1 Score (Macro): Harmonic mean of precision/recall
```

## Notebook Structure

The notebook is organized into clear sections:

1. **Setup & Imports**
   - Library imports
   - Model configuration
   - Device setup (GPU/CPU)

2. **Data Loading**
   - Download from Kaggle
   - Load CSV files
   - Combine and label datasets

3. **Exploratory Data Analysis (EDA)**
   - Class distribution visualization
   - N-gram analysis (bigrams) for titles and text
   - Token frequency analysis

4. **Data Preprocessing**
   - Dataset shuffling
   - Train/validation/test split (80/10/10)
   - Tokenization with padding and truncation
   - Label verification

5. **Model Training**
   - BERT model initialization
   - Layer freezing configuration
   - Training with Trainer API
   - Progress monitoring

6. **Evaluation**
   - Test set evaluation
   - Classification report generation
   - Performance metrics calculation

7. **Model Saving**
   - Save trained model
   - Save tokenizer
   - Ready for deployment

## Installation

### Requirements

```bash
pip install torch transformers datasets evaluate kagglehub scikit-learn numpy pandas matplotlib seaborn nltk
```

### Full Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
kagglehub>=0.1.0
scikit-learn>=1.2.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
```

### NLTK Data

```python
import nltk
nltk.download('punkt')  # Required for tokenization
```

## Usage

### 1. Run the Notebook

```bash
jupyter notebook model.ipynb
```

Execute cells sequentially from top to bottom.

### 2. Key Configuration Variables

At the top of the notebook:

```python
MODEL_NAME = "bert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 3. Data Download

The notebook automatically downloads the dataset from Kaggle:

```python
path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
```

### 4. Training Progress

Training outputs show:
```
Epoch 1/2: Loss: 0.6026, Val Accuracy: 79.53%
Epoch 2/2: Loss: 0.5684, Val Accuracy: 82.09%
Test Accuracy: 82.34%
```

### 5. Using the Trained Model

After training, load and use the model:

```python
from transformers import BertForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./FakeNewsDetector/final_model")
tokenizer = AutoTokenizer.from_pretrained("./FakeNewsDetector/final_model")
model.eval()

# Predict
def predict_fake_news(text):
    inputs = tokenizer(text, max_length=256, truncation=True, 
                       padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
    
    label = "FAKE NEWS" if prediction == 1 else "TRUE NEWS"
    confidence = probabilities[prediction].item() * 100
    
    return label, confidence

# Example usage
text = "Breaking: Scientists discover shocking cure for all diseases!"
label, confidence = predict_fake_news(text)
print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
```

## Exploratory Data Analysis (EDA)

### Class Distribution

The notebook includes visualization showing:
- Balanced dataset with ~52% fake and ~48% true news
- Ensures model won't be biased toward either class

### N-gram Analysis

**Bigram (2-gram) Analysis** reveals common patterns:

**Fake News Title Patterns**:
- Sensational language
- CAPS usage
- Exclamation marks
- Emotional triggers

**Fake News Text Patterns**:
- Repetitive phrases
- Conspiracy-related terms
- Unverified claims

The notebook visualizes the top 10 most common bigrams in both fake news titles and text, helping understand linguistic patterns.

## Model Performance

### Classification Report

```
              precision    recall  f1-score   support

   True News       0.82      0.81      0.82      2163
   Fake News       0.83      0.83      0.83      2327

    accuracy                           0.82      4490
   macro avg       0.82      0.82      0.82      4490
weighted avg       0.82      0.82      0.82      4490
```

### Key Insights

1. **Balanced Performance**: Similar precision/recall for both classes
2. **High Precision**: 82-83% - When model flags as fake, it's usually correct
3. **High Recall**: 81-83% - Catches most fake news articles
4. **F1 Score**: 82% - Good balance between precision and recall

### Training Efficiency

- **Epochs**: Only 2 epochs needed (with early stopping)
- **Training Time**: ~6-7 minutes on GPU
- **Memory**: FP16 reduces memory by ~50%
- **Frozen Layers**: Faster convergence by freezing base BERT

## Advanced Features

### 1. Early Stopping

```python
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
```

Prevents overfitting by stopping training if validation accuracy doesn't improve for 2 consecutive evaluations.

### 2. Mixed Precision Training (FP16)

```python
fp16=True
```

- 2x faster training on compatible GPUs
- 50% memory reduction
- Minimal accuracy impact

### 3. Learning Rate Warmup

```python
warmup_ratio=0.1
```

Gradually increases learning rate during first 10% of training steps for stable optimization.

### 4. Stratified Split

Maintains class balance across train/validation/test sets using:

```python
train_test_split(test_size=0.2, seed=42, stratify=labels)
```

## Customization Options

### Adjust Training Parameters

```python
# For longer training
training_args = TrainingArguments(
    num_train_epochs=5,
    learning_rate=5e-6,  # Lower learning rate
)

# For faster training
training_args = TrainingArguments(
    per_device_train_batch_size=32,  # Larger batches
    num_train_epochs=1,
)
```

### Change Model Architecture

```python
# Use a different BERT variant
MODEL_NAME = "distilbert-base-uncased"  # Faster, smaller
# OR
MODEL_NAME = "roberta-base"  # More robust
```

### Adjust Sequence Length

```python
# For longer articles
preprocess_function = lambda examples: tokenizer(
    examples["text"],
    max_length=512,  # Full BERT capacity
    truncation=True,
    padding="max_length"
)
```

## File Structure

```
.
├── model.ipynb                    # Main Jupyter Notebook
├── FakeNewsDetector/
│   ├── results/                   # Training checkpoints
│   └── final_model/               # Saved model & tokenizer
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── vocab.txt
└──  README.md                    
```

## Limitations & Considerations

### Model Limitations

1. **Sequence Length**: 256 tokens (~200 words)
   - Very long articles are truncated
   - May lose context at the end

2. **Domain Specificity**
   - Trained on specific news categories
   - May not generalize to all fake news types
   - Social media posts may differ from articles

3. **Temporal Bias**
   - Training data from specific time period
   - New fake news tactics may not be recognized
   - Requires periodic retraining

4. **Language**: English-only
   - Cannot detect fake news in other languages
   - Would need multilingual BERT for other languages

5. **False Positives/Negatives**
   - Satire may be flagged as fake (lacks real-world knowledge)
   - Sophisticated fake news may pass detection
   - Human review still recommended for critical content

### Ethical Considerations

**Bias & Fairness**:
- Model may have political/ideological bias from training data
- Could disproportionately flag certain viewpoints
- Regular audits needed for fairness

**Usage Recommendations**:
- Use as **assistance tool**, not sole decision maker
- Combine with human fact-checking
- Provide appeals process for flagged content
- Transparency: Inform users when content is AI-moderated

**Adversarial Robustness**:
- Sophisticated actors can craft content to fool the model
- Adding/removing specific words can change predictions
- Continuous monitoring and updates required

## Performance Optimization

### For Faster Training

```python
# Increase batch size (if GPU memory allows)
per_device_train_batch_size=32

# Use smaller model
MODEL_NAME = "distilbert-base-uncased"

# Reduce sequence length
max_length=128
```

### For Better Accuracy

```python
# More training
num_train_epochs=5

# Unfreeze more layers
# (Remove layer freezing code)

# Lower learning rate
learning_rate=5e-6

# Ensemble multiple models
# Train 3-5 models with different seeds, average predictions
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
per_device_train_batch_size=8

# Reduce sequence length
max_length=128

# Use gradient accumulation
gradient_accumulation_steps=2
```

### Poor Performance

- Check class balance in splits
- Verify labels are correct (1=fake, 0=true)
- Increase training epochs
- Try unfreezing base BERT layers

### Notebook Crashes

- Restart kernel and clear outputs
- Run cells sequentially
- Check available RAM/GPU memory
- Reduce dataset size for testing

## Future Improvements

1. **Multi-Modal Analysis**: Incorporate images from articles
2. **Explainability**: Add attention visualization to show which words influenced prediction
3. **Real-Time Detection**: Deploy as API for live content moderation
4. **Active Learning**: Learn from user corrections
5. **Cross-Lingual**: Train multilingual model for global fake news detection
6. **Fact Verification**: Integrate external knowledge bases for claim verification
7. **Source Credibility**: Add domain reputation scores as features

## References

- **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Fake News Research**: [A Survey on Fake News and Rumour Detection](https://arxiv.org/abs/1811.00770)
- **Hugging Face Docs**: [Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- **Dataset**: [Kaggle Fake News Detection](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

## License

This project is licensed under the MIT License. BERT is licensed under Apache 2.0.

## Acknowledgments

- **Dataset**: Emine Yetim ([Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets))
- **Model**: Google Research (BERT)
- **Framework**: Hugging Face Transformers
- **Community**: Open-source contributors

---

**Model Type**: Binary Text Classification | **Accuracy**: 82.34% | **Framework**: PyTorch + Transformers | **Last Updated**: December 2025

