# Fake News Detection System with LLM Verification

A comprehensive fake news detection system combining **BERT classification** (82.34% accuracy) with **LLM-powered verification** using Mistral-7B for enhanced analysis and explainability.

## Overview

This project implements a two-stage fake news detection system:
1. **Stage 1 (BERT)**: Fast, accurate binary classification (Real/Fake)
2. **Stage 2 (LLM)**: Deep content analysis, verification, and reasoning

The system provides not just a prediction, but also:
- Confidence scores
- Match score (how well content aligns with prediction)
- Reasoning and justification
- Secondary verification by LLM

## System Architecture

```
┌─────────────────┐
│   News Article  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   BERT Classifier       │
│   (82.34% accuracy)     │
└────────┬────────────────┘
         │
         │ Category + Confidence
         ▼
┌─────────────────────────┐
│   LLM Verifier          │
│   (Mistral-7B)          │
│   - Content Analysis    │
│   - Match Scoring       │
│   - Reasoning           │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Final Report          │
│   - BERT: Real/Fake     │
│   - Confidence: X%      │
│   - Match Score: Y%     │
│   - LLM: is_fake        │
│   - Reasoning: ...      │
└─────────────────────────┘
```

## Performance Metrics

### BERT Classifier
- **Test Accuracy**: 82.34%
- **Precision**: 82% (macro average)
- **Recall**: 82% (macro average)
- **F1 Score**: 82% (macro average)

### Per-Class Performance
```
              Precision  Recall  F1-Score  Support
True News         0.82    0.81      0.82    2,163
Fake News         0.83    0.83      0.83    2,327
────────────────────────────────────────────────
Accuracy                           0.82    4,490
Macro Avg         0.82    0.82      0.82    4,490
```

## Key Features

### 1. **Dual-Stage Verification**
- BERT provides fast initial classification
- LLM performs deep content analysis
- Combines statistical patterns with reasoning

### 2. **Match Scoring**
- LLM evaluates how well content matches BERT's prediction
- Provides confidence calibration
- Highlights potential misclassifications

### 3. **Explainability**
- BERT: Statistical confidence score
- LLM: Human-readable reasoning
- Transparent decision-making process

### 4. **Interactive CLI**
- Real-time classification
- Paste news text directly
- Continuous operation (type "quit" to exit)

## Why This Architecture?

### Why BERT for Stage 1?

#### **Fast & Accurate Initial Filter**
- 82% accuracy with <100ms inference time
- Trained on 44,898 news articles
- Efficient GPU/CPU deployment
- Perfect for high-volume screening

#### **Bidirectional Context Understanding**
- Captures subtle linguistic patterns
- Understands sensational language
- Detects emotional manipulation
- Identifies logical inconsistencies

#### **Frozen Base Layers Strategy**
```python
for param in model.bert.parameters():
    param.requires_grad = False
```
- Prevents catastrophic forgetting
- Faster training (only classifier trained)
- Better generalization
- More stable predictions

### Why LLM (Mistral-7B) for Stage 2?

#### **Deep Content Reasoning**
- Analyzes factual consistency
- Evaluates logical coherence
- Identifies manipulation tactics
- Provides human-interpretable explanations

#### **Verification & Calibration**
- Cross-checks BERT's prediction
- Identifies edge cases
- Provides match score
- Reduces false positives/negatives

#### **Mistral-7B Advantages**
- Free tier available via OpenRouter
- Strong instruction following
- Efficient (7B parameters)
- Good JSON output adherence
- Balance of speed and quality

### Why Two Stages?

| Aspect | BERT Only | LLM Only | BERT + LLM |
|--------|-----------|----------|------------|
| **Speed** | Very Fast | Slow | Fast initial + selective deep analysis |
| **Accuracy** | 82% | Variable | 82% + verification |
| **Explainability** | Confidence only | Full reasoning | Both statistical + reasoning |
| **Cost** | Low | High | Moderate (selective LLM calls) |
| **Scalability** | Excellent | Poor | Good |

**The two-stage approach provides the best of both worlds**: BERT's speed and accuracy with LLM's reasoning and verification.

## Installation

### Requirements

```bash
pip install torch transformers datasets evaluate kagglehub scikit-learn numpy pandas matplotlib nltk requests python-dotenv
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
nltk>=3.8.0
requests>=2.28.0
python-dotenv>=1.0.0
```

### Setup

1. **Train the BERT model** (or download pre-trained):
```bash
jupyter notebook model.ipynb
# Run all cells to train and save model
```

2. **Set up OpenRouter API**:
```bash
# Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

Get your free API key from [OpenRouter](https://openrouter.ai/)

3. **Run the system**:
```bash
python main.py
```

## Usage

### Interactive Mode

```bash
python main.py
```

**Example Session**:
```
=== Fake News Detector ===

Paste the news text here: 

Breaking: Scientists discover shocking cure that doctors don't want you to know!
New research reveals that eating raw garlic every morning can cure all diseases.
Big Pharma is trying to hide this information. Share before it gets deleted!

=== News Classification ===
Predicted Category: Fake
Confidence: 94.32%

=== LLM Feedback ===
Match Score: 95%

Is Fake:
true

Reason:
The text uses sensational language ("SHOCKING," "doctors don't want you to know"),
makes extraordinary medical claims without evidence, and employs conspiracy rhetoric
("Big Pharma is trying to hide"). These are classic fake news patterns.

Paste the news text here: 
quit
```

### Programmatic Usage

```python
from main import predict_category, call_openrouter, build_prompt

# Stage 1: BERT Classification
news_text = "Your news article here..."
category, confidence = predict_category(news_text)

print(f"BERT Prediction: {category}")
print(f"Confidence: {confidence*100:.2f}%")

# Stage 2: LLM Verification
prompt = build_prompt(news_text, category, confidence)
feedback = call_openrouter(prompt)

print(f"Match Score: {feedback['match_score']}%")
print(f"LLM Says Fake: {feedback['is_fake']}")
print(f"Reasoning: {feedback['reason']}")
```

### Batch Processing

```python
def process_news_batch(news_articles):
    results = []
    
    for article in news_articles:
        # BERT classification
        category, confidence = predict_category(article)
        
        # LLM verification (only for uncertain cases)
        if confidence < 0.8 or confidence > 0.95:
            prompt = build_prompt(article, category, confidence)
            feedback = call_openrouter(prompt)
        else:
            feedback = None
        
        results.append({
            'article': article[:100],  # First 100 chars
            'bert_prediction': category,
            'bert_confidence': confidence,
            'llm_feedback': feedback
        })
    
    return results

# Process multiple articles
articles = [article1, article2, article3, ...]
results = process_news_batch(articles)
```

## System Components

### 1. BERT Classifier (`predict_category`)

**Purpose**: Fast, accurate initial classification

**Implementation**:
```python
def predict_category(news_text):
    inputs = tokenizer(
        news_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = F.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_idx].item()
    
    return LABEL_MAP[pred_idx], confidence
```

**Key Features**:
- Processes up to 512 tokens
- Returns probability distribution
- GPU acceleration support
- Efficient batch processing

### 2. LLM Verifier (`call_openrouter`)

**Purpose**: Deep content analysis and reasoning

**Prompt Engineering**:
```python
def build_prompt(news, category, confidence):
    return f"""
    You are an AI fake news detector assistant.
    The news was classified as "{category}" with {confidence:.2f} confidence.
    
    Your task:
    - Analyze the news content
    - Verify if content aligns with classification
    
    News: "{news}"
    
    Return ONLY valid JSON:
    {{
        "match_score": 0-100,
        "is_fake": true or false,
        "reason": "concise explanation"
    }}
    """
```

**Why This Prompt Works**:
- Clear role definition
- Explicit output format (JSON)
- Structured reasoning requirement
- Temperature=0.2 for consistency

### 3. JSON Extraction (`extract_json`)

**Purpose**: Robust JSON parsing from LLM output

```python
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())
```

**Why Regex?**:
- Handles markdown code blocks
- Extracts JSON from verbose responses
- Robust to formatting variations
- Fails gracefully with clear error

## Training the BERT Model

### Dataset

**Source**: [Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

**Statistics**:
```
Total: 44,898 articles
- Fake: 23,481 (52.3%)
- True: 21,417 (47.7%)

Training Split:
- Train: 35,918 (80%)
- Validation: 4,490 (10%)
- Test: 4,490 (10%)
```

### Training Configuration

```python
Training Hyperparameters:
- Model: bert-base-uncased
- Max Length: 256 tokens
- Batch Size: 16
- Learning Rate: 1e-5
- Epochs: 2 (with early stopping)
- Weight Decay: 0.05
- Warmup Ratio: 0.1
- Dropout: 0.3 (hidden + attention)
- FP16: Enabled (2x faster training)
```

### Training Strategy

#### **Frozen Base Layers**
```python
for param in model.bert.parameters():
    param.requires_grad = False
```

**Benefits**:
- 60% faster training
- Prevents catastrophic forgetting
- Better generalization
- Only 2M trainable parameters (classifier head)

#### **Early Stopping**
```python
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
```

Stops training if validation accuracy doesn't improve for 2 epochs.

### Training Results

```
Epoch 1: Train Loss 0.603, Val Acc: 79.53%
Epoch 2: Train Loss 0.568, Val Acc: 82.09%

Final Test Accuracy: 82.34%
Training Time: ~7 minutes on GPU
```

## API Integration

### OpenRouter Configuration

```python
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
```

### Request Format

```python
payload = {
    "model": "mistralai/mistral-7b-instruct:free",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2  # Low temp for consistent output
}
```

### Error Handling

```python
if response.status_code != 200:
    raise Exception(f"OpenRouter error: {response.text}")

try:
    content = response.json()["choices"][0]["message"]["content"]
    return extract_json(content)
except (KeyError, json.JSONDecodeError) as e:
    print(f"Error parsing LLM response: {e}")
    return None
```

## LLM Output Format

### Successful Response

```json
{
  "match_score": 95,
  "is_fake": true,
  "reason": "Article uses sensational language and lacks credible sources."
}
```

### Field Definitions

- **`match_score`** (0-100): How strongly content supports BERT's classification
  - 90-100: Strong alignment
  - 70-89: Moderate alignment
  - <70: Weak alignment, potential misclassification

- **`is_fake`** (boolean): LLM's final judgment after analysis
  - `true`: Content appears to be fake news
  - `false`: Content appears to be legitimate

- **`reason`**: Brief justification (1-2 sentences)
  - Identifies specific patterns
  - Explains reasoning
  - Highlights red flags

## Performance Optimization

### For Faster Inference

```python
# Reduce sequence length
inputs = tokenizer(news_text, max_length=128, ...)  # 2x faster

# Batch BERT predictions
def predict_batch(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

# Selective LLM calls
if confidence > 0.95 or confidence < 0.60:
    # Only call LLM for edge cases
    feedback = call_openrouter(prompt)
```

### For Better Accuracy

```python
# Ensemble BERT models
models = [model1, model2, model3]
predictions = [predict_category(text, m) for m in models]
final_category = majority_vote(predictions)

# Use stronger LLM
OPENROUTER_MODEL = "anthropic/claude-3-sonnet"  # Higher quality

# Increase temperature for diverse reasoning
payload["temperature"] = 0.5
```

### For Production Deployment

```python
# Cache BERT predictions
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(news_text):
    return predict_category(news_text)

# Async LLM calls
import asyncio
import aiohttp

async def async_call_openrouter(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_URL, json=payload) as resp:
            return await resp.json()

# Rate limiting
from time import sleep
from functools import wraps

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(max_per_minute=30)
def call_openrouter_limited(prompt):
    return call_openrouter(prompt)
```

## Limitations & Considerations

### BERT Classifier Limitations

1. **Sequence Length**: 256 tokens max
   - Longer articles truncated
   - May miss important context at end

2. **Context Window**: 
   - BERT trained on news articles
   - May struggle with social media posts (different style)
   - Domain adaptation needed for specialized content

3. **Confidence Calibration**:
   - High confidence doesn't guarantee correctness
   - Edge cases (satire, opinion pieces) challenging

### LLM Verifier Limitations

1. **API Dependency**:
   - Requires internet connection
   - Subject to rate limits
   - Costs may apply (free tier limited)

2. **Response Variability**:
   - Temperature=0.2 reduces but doesn't eliminate variance
   - May occasionally produce invalid JSON
   - Reasoning quality varies

3. **Latency**:
   - LLM calls add 1-3 seconds
   - Not suitable for real-time high-volume

4. **Knowledge Cutoff**:
   - Mistral-7B trained on data up to certain date
   - May not know recent events
   - Can't verify current facts

### System-Wide Limitations

1. **No External Verification**:
   - Doesn't check sources
   - Can't access referenced URLs
   - No fact-checking database integration

2. **Language**: English only
   - Both BERT and LLM trained on English
   - Needs multilingual models for other languages

3. **Satire Detection**:
   - May misclassify satire as fake news
   - Lacks real-world knowledge to distinguish

## Future Improvements

1. **Source Verification Integration**
   - Fact-checking database lookup
   - URL credibility scoring
   - Cross-reference multiple sources

2. **Multi-Modal Analysis**
   - Image verification (detect manipulated photos)
   - Video deepfake detection
   - Audio analysis

3. **Real-Time Fact Extraction**
   - Named entity recognition
   - Claim extraction
   - Automated fact-checking

4. **Advanced LLM Features**
   - Chain-of-thought reasoning
   - Multi-agent debate (multiple LLMs vote)
   - Retrieval-augmented generation (RAG)

5. **User Feedback Loop**
   - Collect corrections from users
   - Active learning for BERT
   - Fine-tune LLM with feedback

6. **Explainability Dashboard**
   - Highlight suspicious phrases
   - Show attention weights
   - Visualize decision process

7. **Mobile App**
   - Camera scan of newspaper articles
   - Share from social media directly
   - Offline mode with cached model

## References

- **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Mistral-7B**: [Mistral 7B](https://arxiv.org/abs/2310.06825)
- **Dataset**: [Fake News Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- **OpenRouter**: [OpenRouter AI](https://openrouter.ai/)
- **Transformers**: [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## License

This project is licensed under the MIT License. 
- BERT: Apache 2.0
- Mistral-7B: Apache 2.0
- Dataset: Subject to Kaggle's terms

## Acknowledgments

- **Dataset**: Emine Yetim (Kaggle)
- **BERT**: Google Research
- **Mistral**: Mistral AI
- **Frameworks**: Hugging Face, PyTorch
- **API**: OpenRouter

---

**System**: Two-Stage Detection (BERT + LLM) | **BERT Accuracy**: 82.34% | **LLM**: Mistral-7B | **Last Updated**: December 2025

**⚠️ Disclaimer**: This system is designed to assist in identifying potential misinformation. It should NOT be the sole basis for content moderation or censorship decisions. Always combine AI predictions with human judgment, professional fact-checkers, and source verification. The system may produce false positives (flagging legitimate news) and false negatives (missing actual fake news). Regular audits and updates are essential.