# Fake News Detector

A deep learning-based binary classifier that distinguishes between fake and legitimate news articles using BERT (Bidirectional Encoder Representations from Transformers), achieving high accuracy in detecting misinformation.

## Overview

This project implements a robust fake news detection system that analyzes news article titles, content, and subjects to determine authenticity. The model is trained on a large corpus of labeled news articles and can identify subtle patterns that distinguish credible journalism from fabricated content.

## Performance Metrics

The model is optimized for **F1 score** to balance precision and recall, making it effective at:
- Minimizing false positives (legitimate news marked as fake)
- Minimizing false negatives (fake news passing as legitimate)

**Evaluation Metrics**:
- Accuracy
- Precision (how many flagged articles are actually fake)
- Recall (how many fake articles are caught)
- F1 Score (harmonic mean of precision and recall)

## Why BERT?

### Model Selection Rationale

**BERT (bert-base-uncased)** was selected as the optimal architecture for fake news detection due to several critical advantages:

#### 1. **Bidirectional Context Understanding**
- Unlike traditional models that read text left-to-right, BERT reads **both directions simultaneously**
- Captures nuanced meanings: "The bank is near the river" vs "I need to bank my money"
- Critical for detecting subtle manipulation in fake news (misleading context, cherry-picked facts)

#### 2. **Pre-trained on Massive Text Corpus**
- Trained on 3.3 billion words (Wikipedia + BookCorpus)
- Already understands grammar, facts, reasoning, and common knowledge
- Requires minimal fine-tuning to specialize in fake news detection

#### 3. **Attention Mechanism**
- Identifies which words/phrases are most important for classification
- Can focus on suspicious patterns: sensational language, conspiracy keywords, logical inconsistencies
- Example: "BREAKING: Scientists SHOCKED by this ONE trick!" → High attention to hyperbolic language

#### 4. **Handles Long Documents**
- Max sequence length: 512 tokens (~400-500 words)
- Processes complete articles, not just snippets
- Captures long-range dependencies between title, content, and subject

#### 5. **Robust to Adversarial Content**
- Pre-training on diverse text makes it resistant to manipulation
- Can detect fake news even when grammar/spelling is correct
- Identifies deeper semantic inconsistencies

#### 6. **State-of-the-Art Baseline**
- BERT is the industry standard for text classification
- Proven effectiveness on misinformation detection benchmarks
- Strong transfer learning capabilities

### Alternatives Considered

| Model | Pros | Cons | Why Not Selected |
|-------|------|------|------------------|
| **Logistic Regression (TF-IDF)** | Fast, interpretable, lightweight | Misses context, can't handle semantics | Poor on sophisticated fake news |
| **LSTM/GRU** | Handles sequences, faster than BERT | Unidirectional, smaller capacity | Lower accuracy, less robust |
| **DistilBERT** | 40% faster, 60% smaller | 3-5% accuracy drop | Accuracy is critical for misinformation |
| **RoBERTa** | More robust, better performance | 125M parameters, slower | Marginal gains, higher compute cost |
| **GPT-2/3** | Strong language understanding | Unidirectional, designed for generation | Not optimized for classification |
| **T5** | Versatile, strong generalist | Overkill for binary classification | Slower, more complex |

**BERT strikes the perfect balance between accuracy, robustness, and computational efficiency for fake news detection.**

## Dataset

**Expected Format**: Two CSV files containing labeled news articles

### Data Structure

**Fake.csv** (Fake News)
```csv
title,text,subject
"BREAKING: Shocking discovery!","Scientists claim...",politics
```

**True.csv** (Legitimate News)
```csv
title,text,subject
"Senate passes new bill","The U.S. Senate voted...","Government News"
```

**Features**:
- **Title**: Article headline (often sensationalized in fake news)
- **Text**: Full article content (main source of information)
- **Subject**: Topic category (politics, health, science, etc.)

**Label Distribution**:
- Binary classification: 0 = True News, 1 = Fake News
- Dataset should be balanced for optimal training

### Popular Datasets

You can use publicly available datasets such as:
- [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- LIAR Dataset (fact-checking statements)
- FakeNewsNet (social media fake news)

## Technical Architecture

### Model Specifications

```
Architecture: BERT (Bidirectional Encoder)
- Base Model: bert-base-uncased
- Parameters: 110M
- Encoder Layers: 12
- Attention Heads: 12 per layer
- Hidden Size: 768
- Vocabulary: 30,522 WordPiece tokens
- Max Sequence Length: 512 tokens
```

### Input Processing

The model concatenates three fields into a structured format:
```
Title: [article title]
Text: [article content]
Subject: [category]
```

This structured format helps BERT learn:
- Title patterns (clickbait, sensationalism)
- Content quality (sources, coherence)
- Subject consistency (does the content match the category?)

### Training Configuration

```python
Hyperparameters:
- Learning Rate: 2e-5 (standard for BERT fine-tuning)
- Batch Size: 8 per device
- Epochs: 3
- Weight Decay: 0.01 (L2 regularization)
- Warmup Steps: 500 (gradual learning rate increase)
- Max Sequence Length: 512 tokens
- Test Split: 40% (ensures robust evaluation)
- Optimizer: AdamW (default for Trainer)
```

**Key Training Features**:
- **Stratified Split**: Maintains class balance in train/test sets
- **F1 Optimization**: Model selection based on F1 score (not just accuracy)
- **Early Stopping**: Saves best model based on validation F1
- **Learning Rate Warmup**: Prevents early training instability
- **Checkpoint Management**: Keeps only top 2 models to save space

### Custom Dataset Class

The `NewsDataset` class handles:
1. **Tokenization**: Converts text to BERT-compatible token IDs
2. **Padding**: Ensures all sequences are exactly 512 tokens
3. **Truncation**: Cuts text longer than max_length
4. **Label Encoding**: Converts fake/true to 0/1

### Evaluation Metrics Explained

```python
- Accuracy: (TP + TN) / Total
  → Overall correctness

- Precision: TP / (TP + FP)
  → Of articles marked fake, how many are actually fake?
  → Important to avoid censoring legitimate news

- Recall: TP / (TP + FN)
  → Of all fake articles, how many did we catch?
  → Important to minimize missed misinformation

- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
  → Balanced metric when classes are imbalanced
  → Model is optimized for this metric
```

## Installation

### Full Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.2.0
numpy>=1.24.0
```

### System Requirements

- **GPU**: CUDA-compatible GPU recommended (CPU training is slow)
- **RAM**: 16GB minimum for full dataset
- **Storage**: 2GB for model and cache
- **Python**: 3.8+

## Usage

### 1. Prepare Your Data

Ensure you have two CSV files in the `FakeNewsDetector/` directory:
- `Fake.csv`: Fake news articles (label = 1)
- `True.csv`: True news articles (label = 0)

**CSV Format**:
```csv
title,text,subject
"Article Title","Full article text here...","Category"
```

### 2. Train the Model

```bash
python fake_news_detector.py
```

**Training Process**:
1. Loads and combines fake/true datasets
2. Splits into 60% train / 40% test (stratified)
3. Tokenizes text for BERT
4. Trains for 3 epochs (~30-60 minutes on GPU)
5. Evaluates on test set
6. Saves best model to `./FakeNewsDetector/model/`

**Expected Output**:
```
Loading data...
Total samples: 44898 (Fake: 23481, True: 21417)
Loading tokenizer and model...
Starting training...

Epoch 1/3: 100%|████████████| 3368/3368 [12:34<00:00, 4.46it/s]
Eval accuracy: 0.9542, F1: 0.9538
Epoch 2/3: 100%|████████████| 3368/3368 [12:28<00:00, 4.50it/s]
Eval accuracy: 0.9687, F1: 0.9683
Epoch 3/3: 100%|████████████| 3368/3368 [12:31<00:00, 4.48it/s]
Eval accuracy: 0.9712, F1: 0.9709

CLASSIFICATION REPORT (Test Set)
              precision    recall  f1-score   support

   True News       0.97      0.97      0.97      8567
   Fake News       0.97      0.97      0.97      9392

    accuracy                           0.97     17959
   macro avg       0.97      0.97      0.97     17959
weighted avg       0.97      0.97      0.97     17959

Model saved successfully!
```

### 3. Using the Trained Model

```python
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# Load model
model = BertForSequenceClassification.from_pretrained("./FakeNewsDetector/model")
tokenizer = AutoTokenizer.from_pretrained("./FakeNewsDetector/model")
model.eval()

# Predict
def predict_fake_news(title, text, subject):
    input_text = f"Title: {title}\nText: {text}\nSubject: {subject}"
    
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
    
    label = "FAKE NEWS" if prediction == 1 else "TRUE NEWS"
    confidence = probabilities[prediction].item() * 100
    
    return label, confidence

# Example usage
title = "Scientists discover shocking cure for all diseases!"
text = "A groundbreaking study reveals that eating magical berries..."
subject = "health"

label, confidence = predict_fake_news(title, text, subject)
print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
```

### 4. Batch Prediction

```python
articles = [
    {
        "title": "Senate passes infrastructure bill",
        "text": "The U.S. Senate voted 69-30 to approve...",
        "subject": "politics"
    },
    {
        "title": "SHOCKING: Government hiding aliens!",
        "text": "Sources claim that the government...",
        "subject": "conspiracy"
    }
]

for article in articles:
    label, confidence = predict_fake_news(
        article["title"],
        article["text"],
        article["subject"]
    )
    print(f"{article['title'][:50]}... → {label} ({confidence:.1f}%)")
```

## Project Structure

```
.
├── model.py      # Main training script
├── Fake.csv               # Fake news dataset
├── True.csv               # True news dataset
├──  model/                 # Saved model & tokenizer
├── results/                   # Training checkpoints
├── logs/                      # TensorBoard logs
└──  README.md                  # This file 
```

## Understanding the Results

### Classification Report Breakdown

```
              precision    recall  f1-score   support

   True News       0.97      0.97      0.97      8567
   Fake News       0.97      0.97      0.97      9392
```

**Precision (0.97)**:
- When model says "fake," it's correct 97% of the time
- Low false positive rate (3% of legitimate news flagged)

**Recall (0.97)**:
- Model catches 97% of all fake news
- Only 3% of fake news escapes detection

**F1 Score (0.97)**:
- Excellent balance between precision and recall
- Model is both accurate and comprehensive

### Common Fake News Patterns Detected

1. **Sensational Headlines**
   - ALL CAPS, excessive punctuation
   - Emotionally charged language
   - Clickbait patterns

2. **Content Quality Issues**
   - Poor grammar (though not always)
   - Lack of credible sources
   - Logical inconsistencies

3. **Subject Mismatch**
   - Content doesn't match declared category
   - Mixing unrelated topics

4. **Conspiracy Language**
   - "They don't want you to know..."
   - "BREAKING: Shocking discovery..."
   - Appeals to fear/anger

## Performance Optimization

### For Faster Training

```python
BATCH_SIZE = 16                # Increase if GPU memory allows
MAX_LENGTH = 256               # Reduce for shorter articles
EPOCHS = 2                     # Fewer epochs
```

### For Better Accuracy

```python
EPOCHS = 5                     # More training
learning_rate = 1e-5           # Lower learning rate
weight_decay = 0.02            # Stronger regularization
MAX_LENGTH = 512               # Full context (already optimal)
```

### For Production Deployment

```python
# Use DistilBERT for faster inference (small accuracy drop)
MODEL = "distilbert-base-uncased"

# Quantization for mobile deployment
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=./logs
```

View real-time metrics:
- Training/validation loss
- Accuracy, precision, recall, F1
- Learning rate schedule

### Manual Checkpoints

```python
# Load a specific checkpoint
trainer = Trainer.from_checkpoint("./results/checkpoint-1000")
```

## Advanced Features

### 1. Confidence Thresholding

```python
def predict_with_threshold(title, text, subject, threshold=0.8):
    label, confidence = predict_fake_news(title, text, subject)
    
    if confidence < threshold:
        return "UNCERTAIN (Manual Review Needed)", confidence
    return label, confidence
```

### 2. Explainability (Attention Visualization)

```python
from transformers import BertTokenizer
import torch

def get_attention_weights(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # Extract attention from last layer
    attention = outputs.attentions[-1][0].mean(dim=0)
    
    # Map to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    return list(zip(tokens, attention.mean(dim=0).tolist()))
```

### 3. Ensemble with Multiple Models

```python
models = [
    BertForSequenceClassification.from_pretrained("./model1"),
    BertForSequenceClassification.from_pretrained("./model2"),
]

def ensemble_predict(text):
    predictions = [model_predict(m, text) for m in models]
    return max(set(predictions), key=predictions.count)  # Majority vote
```

## Limitations & Considerations

### Model Limitations

1. **Not 100% Accurate**
   - 97% accuracy means 3% error rate
   - Should assist humans, not replace them

2. **Training Data Bias**
   - Model learns patterns from training data
   - May struggle with new fake news tactics
   - Needs regular retraining

3. **Context Limitation**
   - 512 tokens (~400 words)
   - Very long articles may be truncated
   - Loses information beyond cutoff

4. **Satire Confusion**
   - May flag satirical news (The Onion, Babylon Bee)
   - Lacks real-world knowledge to verify facts
   - Can't distinguish satire from genuine misinformation

5. **Adversarial Attacks**
   - Sophisticated actors can craft content to fool the model
   - Adding/removing specific words can change predictions
   - Requires continuous model updates

## Future Improvements

1. **Multi-Class Classification**: Categorize types of misinformation (satire, propaganda, clickbait)
2. **Source Credibility**: Incorporate domain reputation scores
3. **Fact Verification**: Cross-reference claims with knowledge bases
4. **Temporal Analysis**: Track how fake news evolves over time
5. **Multilingual Support**: Extend to non-English news
6. **Explainable AI**: Highlight specific sentences that indicate fakeness
7. **Active Learning**: Learn from user corrections
8. **Real-Time API**: Deploy as a web service

## Citation

If you use this code or approach, please cite:

```bibtex
@misc{fake_news_detector_bert,
  title={Fake News Detector using BERT},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fake-news-detector}
}
```

## References

- **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Fake News Detection**: [A Survey on Fake News and Rumour Detection](https://arxiv.org/abs/1811.00770)
- **Hugging Face Docs**: [Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## License

This project is licensed under the MIT License. BERT is licensed under Apache 2.0.

## Acknowledgments

- **Dataset**: Various Kaggle contributors
- **Model**: Google Research (BERT)
- **Framework**: Hugging Face Transformers
- **Community**: Open-source contributors

---

**Task**: Binary Classification | **Accuracy**: ~97% | **Model**: BERT-base | **Last Updated**: December 2025

**⚠️ Disclaimer**: This model is a tool to assist in identifying potential misinformation. It should not be the sole basis for content moderation decisions. Always combine AI predictions with human judgment and fact-checking.