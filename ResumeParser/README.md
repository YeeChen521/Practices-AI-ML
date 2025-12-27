# Resume Job Category Classifier

A deep learning-based multi-class classifier that automatically categorizes resumes into 24 different job categories using DistilBERT, achieving **87.67% test accuracy** through fine-tuning with class balancing and weighted loss.

## Overview

This project implements an intelligent resume screening system that can automatically classify resumes into their appropriate job categories. The model processes resume text and assigns it to one of 24 professional categories, making it ideal for automated recruitment systems, job portals, and HR departments.

## Performance Metrics

- **Test Accuracy**: 87.67%
- **Macro F1 Score**: 87%
- **Macro Precision**: 89%
- **Macro Recall**: 87%

### Per-Category Performance Highlights

**Perfect Classification (100% F1)**:
- HR (1.00 precision, 1.00 recall)
- Business Development (1.00/1.00)
- Finance (1.00/1.00)
- Accountant (1.00/1.00)

**Strong Performance (>95% F1)**:
- Engineering (0.95/1.00, F1: 0.97)
- Chef (0.94/1.00, F1: 0.97)
- Consultant (0.93/1.00, F1: 0.97)
- Designer (0.93/1.00, F1: 0.96)
- Information Technology (1.00/0.92, F1: 0.96)

## Why DistilBERT?

### Model Selection Rationale

**DistilBERT (distilbert-base-uncased)** was selected as the optimal architecture for resume classification due to several key advantages:

#### 1. **Efficiency Without Sacrificing Performance**
- **40% smaller** than BERT (66M vs 110M parameters)
- **60% faster** inference time
- Only ~3-5% accuracy drop compared to full BERT
- Perfect for production deployment in HR systems

#### 2. **Strong Transfer Learning from BERT**
- Trained via **knowledge distillation** from BERT-base
- Retains 97% of BERT's language understanding
- Pre-trained on same corpus (Wikipedia + BookCorpus)
- Understands professional terminology and context

#### 3. **Optimized for Text Classification**
- Proven performance on multi-class classification tasks
- Handles professional documents well
- Captures semantic meaning of job descriptions and skills
- Effective with 512-token context window

#### 4. **Multi-Class Classification Capability**
- Successfully handles 24 distinct job categories
- Learns subtle differences between similar roles:
  - HR vs Public Relations
  - Engineering vs Information Technology
  - Sales vs Business Development
  - Consultant vs Advocate

#### 5. **Resource-Friendly for Production**
- Lower inference latency (critical for real-time screening)
- Reduced memory footprint
- Can process more resumes per second
- Cost-effective for large-scale deployment

#### 6. **Resume-Specific Advantages**
- Captures key information:
  - Skills and technologies mentioned
  - Job titles and roles
  - Educational qualifications
  - Industry-specific terminology
  - Project descriptions and achievements

### Why Not Other Models?

| Model | Pros | Cons | Why Not Selected |
|-------|------|------|------------------|
| **BERT-base** | Highest accuracy, robust | 110M params, slower inference | Marginal gain doesn't justify 60% speed penalty |
| **RoBERTa** | More robust training | 125M params, even slower | Overkill for resume classification |
| **LSTM/BiLSTM** | Fast, simple | Poor at capturing context, lower accuracy | Insufficient for nuanced job categorization |
| **Word2Vec + SVM** | Very fast, interpretable | No context understanding, misses semantic meaning | Too simplistic for professional documents |
| **GPT-2/3** | Strong language model | Unidirectional, designed for generation | Not optimized for classification tasks |
| **ALBERT** | Parameter efficient | Slower than DistilBERT despite fewer params | Speed vs accuracy trade-off favors DistilBERT |

**DistilBERT provides the sweet spot: BERT-level accuracy with production-ready speed and efficiency.**

## Dataset

**Source**: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Kaggle)

### Job Categories (24 Classes)

```python
Categories:
1. HR                        13. Digital-Media
2. Designer                  14. Automobile
3. Information-Technology    15. Chef
4. Teacher                   16. Finance
5. Advocate                  17. Apparel
6. Business-Development      18. Engineering
7. Healthcare                19. Accountant
8. Fitness                   20. Construction
9. Agriculture               21. Public-Relations
10. BPO                      22. Banking
11. Sales                    23. Arts
12. Consultant               24. Aviation
```

### Dataset Statistics

```
Original Dataset: Imbalanced
Total samples used: 5,000 (after balancing)

Split:
- Training: 3,500 samples (70%)
- Validation: 750 samples (15%)
- Test: 750 samples (15%)

After Oversampling:
All classes balanced to max class count
Each category: ~208 samples (balanced)
```

### Data Preprocessing

#### Class Imbalance Handling

The original dataset had significant class imbalance. The notebook implements **oversampling**:

```python
Strategy:
1. Identify class with maximum samples
2. Oversample minority classes by repeating samples
3. Balance all classes to match maximum count
4. Shuffle to prevent sequential bias
```

**Why oversampling?**
- Prevents model bias toward majority classes
- Ensures fair learning for all job categories
- Improves recall for underrepresented categories
- More effective than undersampling (no data loss)

## Technical Architecture

### Model Specifications

```
Architecture: DistilBERT-base-uncased
- Parameters: 66M (40% fewer than BERT)
- Encoder Layers: 6 (vs BERT's 12)
- Attention Heads: 12
- Hidden Size: 768
- Vocabulary: 30,522 WordPiece tokens
- Max Sequence Length: 512 tokens
- Output: 24-class softmax classifier
```

### Advanced Training Techniques

#### 1. **Weighted Cross-Entropy Loss**

Despite oversampling, class weights are used for extra robustness:

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted loss based on inverse class frequency
        loss_fct = CrossEntropyLoss(weight=class_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

**Why weighted loss?**
- Double protection against class imbalance
- Penalizes misclassifications of minority classes more
- Improves macro F1 score (treats all classes equally)
- Essential for fair performance across all job categories

#### 2. **Regularization Strategy**

```python
Dropout Configuration:
- Hidden Dropout: 0.3
- Attention Dropout: 0.3
- Weight Decay: 0.05

Purpose:
- Prevents overfitting on repeated samples
- Reduces memorization of specific resumes
- Improves generalization to new resumes
```

#### 3. **Training Configuration**

```python
Hyperparameters:
- Learning Rate: 5e-5 (higher than typical BERT fine-tuning)
- Batch Size: 16 per device
- Epochs: 4 (with early stopping)
- Weight Decay: 0.05 (L2 regularization)
- Warmup Ratio: 0.1 (gradual LR warmup)
- Optimizer: AdamW (built into Trainer)
- Mixed Precision: FP16 (faster training)
```

**Why 5e-5 learning rate?**
- DistilBERT can handle slightly higher LR than BERT
- Faster convergence on smaller model
- Multi-class classification benefits from faster learning
- Still conservative enough to avoid catastrophic forgetting

#### 4. **Early Stopping**

```python
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
```

Stops training if validation F1 doesn't improve for 2 consecutive epochs, preventing overfitting.

### Input Processing

```python
Tokenization:
- Max Length: 512 tokens (full DistilBERT capacity)
- Padding: Max length (fixed size)
- Truncation: Enabled (handles long resumes)
- Field: "resume_str" (full resume text)
```

**Why 512 tokens?**
- Resumes can be lengthy (1-2 pages)
- Need full context to capture all skills and experience
- Critical information often appears throughout document
- 512 tokens ≈ 400-450 words (sufficient for most resumes)

### Training Progress

```
Epoch 1: Loss 2.848, Val Acc: 52.28%, Val F1: 40.86%
Epoch 2: Loss 1.518, Val Acc: 78.28%, Val F1: 75.54%
Epoch 3: Loss 0.793, Val Acc: 87.13%, Val F1: 86.60%
Epoch 4: Loss 0.503, Val Acc: 87.94%, Val F1: 87.55%

Final Test Accuracy: 87.67%
```

**Key Observations**:
- Significant improvement from epoch 1 to 2 (52% → 78%)
- Strong gains in epoch 3 (78% → 87%)
- Convergence by epoch 4
- Test accuracy consistent with validation (good generalization)

## Installation

### Requirements

```bash
pip install torch transformers datasets evaluate kagglehub scikit-learn numpy pandas matplotlib
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
```

## Usage

### 1. Training the Model

Run the Jupyter Notebook cells sequentially:

```bash
jupyter notebook resume_classifier.ipynb
```

Or convert to script:

```bash
jupyter nbconvert --to script resume_classifier.ipynb
python resume_classifier.py
```

### 2. Using the Trained Model

```python
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./FakeNewsDetector/final_model")
tokenizer = AutoTokenizer.from_pretrained("./FakeNewsDetector/final_model")
model.eval()

# Job categories
job_categories = [
    'HR', 'Designer', 'Information-Technology', 'Teacher', 'Advocate',
    'Business-Development', 'Healthcare', 'Fitness', 'Agriculture', 'BPO',
    'Sales', 'Consultant', 'Digital-Media', 'Automobile', 'Chef',
    'Finance', 'Apparel', 'Engineering', 'Accountant', 'Construction',
    'Public-Relations', 'Banking', 'Arts', 'Aviation'
]

def classify_resume(resume_text):
    """
    Classify a resume into one of 24 job categories
    
    Args:
        resume_text (str): Full text of the resume
        
    Returns:
        tuple: (predicted_category, confidence_score, top_3_predictions)
    """
    # Tokenize
    inputs = tokenizer(
        resume_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        
    # Get top prediction
    predicted_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_idx].item() * 100
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    top3_predictions = [
        (job_categories[idx], prob.item() * 100)
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    
    return job_categories[predicted_idx], confidence, top3_predictions

# Example usage
resume_text = """
Software Engineer with 5 years of experience in Python, machine learning,
and web development. Proficient in TensorFlow, PyTorch, Django, and React.
Led development of ML-powered recommendation system. Strong background in
data structures, algorithms, and system design. Bachelor's in Computer Science.
"""

category, confidence, top3 = classify_resume(resume_text)

print(f"Predicted Category: {category}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nTop 3 Predictions:")
for cat, prob in top3:
    print(f"  {cat}: {prob:.2f}%")
```

**Example Output**:
```
Predicted Category: Information-Technology
Confidence: 92.45%

Top 3 Predictions:
  Information-Technology: 92.45%
  Engineering: 4.32%
  Digital-Media: 1.87%
```

### 3. Batch Processing

```python
def classify_resumes_batch(resume_texts, batch_size=8):
    """Process multiple resumes efficiently"""
    results = []
    
    for i in range(0, len(resume_texts), batch_size):
        batch = resume_texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            confidences = torch.softmax(outputs.logits, dim=1).max(dim=1)[0]
        
        for pred, conf in zip(predictions, confidences):
            results.append({
                'category': job_categories[pred.item()],
                'confidence': conf.item() * 100
            })
    
    return results

# Process multiple resumes
resumes = [resume1, resume2, resume3, ...]
results = classify_resumes_batch(resumes)

for i, result in enumerate(results):
    print(f"Resume {i+1}: {result['category']} ({result['confidence']:.1f}%)")
```

## Model Performance Analysis

### Classification Report (Test Set)

```
Category                 Precision  Recall   F1-Score  Support
─────────────────────────────────────────────────────────────
HR                       1.00       1.00     1.00      12
Designer                 0.93       1.00     0.96      13
Information-Technology   1.00       0.92     0.96      12
Teacher                  0.90       0.95     0.92      19
Advocate                 0.72       0.81     0.76      16
Business-Development     1.00       1.00     1.00      17
Healthcare               1.00       0.83     0.91      12
Fitness                  0.80       0.86     0.83      14
Agriculture              0.91       0.56     0.69      18
BPO                      0.83       1.00     0.90      19
Sales                    0.85       0.92     0.88      12
Consultant               0.93       1.00     0.97      14
Digital-Media            0.80       0.86     0.83      14
Automobile               0.50       0.62     0.55      13
Chef                     0.94       1.00     0.97      17
Finance                  1.00       1.00     1.00      17
Apparel                  1.00       0.54     0.70      13
Engineering              0.95       1.00     0.97      18
Accountant               1.00       1.00     1.00      16
Construction             0.89       1.00     0.94      17
Public-Relations         0.72       0.81     0.76      16
─────────────────────────────────────────────────────────────
Accuracy                                      0.88      373
Macro Avg                0.89      0.87     0.87      373
Weighted Avg             0.89      0.88     0.87      373
```

### Performance Insights

#### Excellent Performance (F1 > 0.95)
- **HR, Finance, Accountant, Business-Development**: Perfect classification
- **Engineering, Chef, Consultant**: Near-perfect with minor recall issues
- **Strong distinction** between technical and non-technical roles

#### Good Performance (F1: 0.80-0.95)
- **IT, Teacher, Healthcare, Construction, Sales**: Solid performance
- **Slight confusion** between overlapping domains

#### Challenging Categories (F1 < 0.80)
- **Automobile** (0.55): Lowest performance - low precision (50%) and recall (62%)
- **Agriculture** (0.69): Low recall (56%) - often misclassified as other outdoor/manual roles
- **Apparel** (0.70): Very low recall (54%) - strong overlap with Designer/Arts
- **Advocate** (0.76): Moderate performance - confused with Public Relations
- **Public-Relations** (0.76): Similar confusion with Advocate
- **Fitness** (0.83): Confused with Healthcare category
- **Digital-Media** (0.83): Overlaps with IT and Designer roles

### Common Misclassifications

**Similar Role Confusion**:
- Information-Technology ↔ Engineering
- Advocate ↔ Public Relations (72-74% precision for both)
- Fitness ↔ Healthcare
- Designer ↔ Arts ↔ Apparel
- Digital-Media ↔ Information-Technology

**Why these confusions occur**:
- Overlapping skill sets (e.g., IT and Engineering both require technical skills)
- Similar job descriptions (e.g., Advocate and Public Relations both involve communication)
- Interdisciplinary roles (e.g., Digital Media spans design and technology)
- Ambiguous resume wording
- Automobile category is too broad (includes engineering, sales, design roles)

## Real-World Applications

### 1. **Automated Resume Screening**
```python
# Filter resumes for specific job opening
def screen_resumes_for_job(resumes, target_category, min_confidence=80):
    matches = []
    for resume in resumes:
        category, confidence, _ = classify_resume(resume)
        if category == target_category and confidence >= min_confidence:
            matches.append((resume, confidence))
    return sorted(matches, key=lambda x: x[1], reverse=True)

# Example: Find IT candidates
it_candidates = screen_resumes_for_job(all_resumes, "Information-Technology")
```

### 2. **Job Recommendation System**
```python
def recommend_jobs(resume_text):
    """Recommend top 3 suitable job categories"""
    _, _, top3 = classify_resume(resume_text)
    
    print("Recommended Job Categories:")
    for i, (category, prob) in enumerate(top3, 1):
        print(f"{i}. {category} (Match: {prob:.1f}%)")
```

### 3. **Resume Quality Check**
```python
def check_resume_clarity(resume_text):
    """Check if resume clearly indicates job category"""
    category, confidence, top3 = classify_resume(resume_text)
    
    if confidence > 90:
        return "Clear profile", category
    elif confidence > 70:
        return "Moderate clarity", category
    else:
        return "Ambiguous profile - consider clarifying", top3
```

### 4. **Skills Gap Analysis**
```python
def analyze_career_transition(current_resume, target_category):
    """Analyze feasibility of career transition"""
    current_cat, _, top3 = classify_resume(current_resume)
    
    # Check if target is in top 3 predictions
    target_scores = [prob for cat, prob in top3 if cat == target_category]
    
    if target_scores:
        transition_score = target_scores[0]
        if transition_score > 50:
            return f"Strong transition potential ({transition_score:.1f}%)"
        elif transition_score > 20:
            return f"Moderate transition potential ({transition_score:.1f}%)"
    
    return "Significant reskilling required"
```

## Advanced Features

### 1. **Confidence Threshold Filtering**

```python
def classify_with_threshold(resume_text, threshold=70):
    """Only return prediction if confidence exceeds threshold"""
    category, confidence, top3 = classify_resume(resume_text)
    
    if confidence >= threshold:
        return category, confidence
    else:
        return "UNCERTAIN", top3  # Return top 3 for manual review
```

### 2. **Multi-Label Consideration**

```python
def get_suitable_categories(resume_text, threshold=20):
    """Return all categories above threshold (handles multi-skilled candidates)"""
    _, _, top3 = classify_resume(resume_text)
    
    suitable = [cat for cat, prob in top3 if prob >= threshold]
    return suitable

# Example: Candidate suitable for multiple roles
categories = get_suitable_categories(resume)
# Output: ['Information-Technology', 'Engineering', 'Digital-Media']
```

### 3. **Ensemble Prediction**

```python
def ensemble_predict(resume_text, models):
    """Combine predictions from multiple models for robustness"""
    predictions = []
    
    for model in models:
        category, confidence, _ = classify_resume_with_model(resume_text, model)
        predictions.append((category, confidence))
    
    # Weighted voting
    category_votes = {}
    for cat, conf in predictions:
        category_votes[cat] = category_votes.get(cat, 0) + conf
    
    best_category = max(category_votes, key=category_votes.get)
    avg_confidence = category_votes[best_category] / len(models)
    
    return best_category, avg_confidence
```

## Limitations & Considerations

### Model Limitations

1. **Long Resume Truncation**
   - 512 tokens ≈ 400-450 words
   - Multi-page resumes may lose information
   - Critical details at end might be cut off
   
   **Solution**: Extract key sections (skills, experience) for classification

2. **Domain-Specific Jargon**
   - Model trained on specific dataset
   - May struggle with uncommon job titles
   - Industry-specific terminology might confuse model
   
   **Solution**: Regular retraining with new resume samples

3. **Multi-Skilled Candidates**
   - Resumes with diverse experience challenging
   - Career changers may confuse the model
   - Freelancers with varied projects
   
   **Solution**: Use top-3 predictions, not just top-1

4. **Oversampling Artifacts**
   - Repeated samples may cause minor overfitting
   - Model might memorize specific resumes
   - Dropout helps, but not perfect
   
   **Solution**: Use more diverse data when available

5. **Temporal Bias**
   - Job market evolves (new roles, skills)
   - Tech skills especially change rapidly
   - Model trained on historical data
   
   **Solution**: Periodic retraining with fresh data

6. **Broad Category Challenges**
   - **Automobile** category performs worst (F1: 0.55)
   - Too broad: includes engineers, designers, sales, mechanics
   - Would benefit from subcategories
   
   **Solution**: Consider hierarchical classification for broad categories

## Performance Optimization

### For Faster Inference

```python
# Reduce sequence length for speed
max_length=256  # 2x faster, minimal accuracy loss for short resumes

# Batch processing
batch_size=32  # Process multiple resumes simultaneously

# Model quantization
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "./model",
    torch_dtype=torch.float16  # Half precision
)
```

### For Better Accuracy

```python
# Increase training data
n_samples=10000  # Use more samples if available

# More training epochs
num_train_epochs=6

# Ensemble multiple models
# Train 3-5 models with different seeds, average predictions

# Fine-tune on domain-specific data
# If you have resumes from specific industry, fine-tune further

# Address challenging categories
# Collect more Automobile, Apparel, Agriculture samples
# Consider splitting broad categories into subcategories
```

### For Production Deployment

```python
# ONNX export for cross-platform inference
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "./model",
    export=True
)
# 2-3x faster inference
```

## Future Improvements

1. **Hierarchical Classification**: 
   - Level 1: Broad categories (Technical, Business, Creative, Service, Manual)
   - Level 2: Specific roles within each category
   - Would help with Automobile, Digital-Media confusion

2. **Entity Recognition**: Extract skills, companies, roles for better understanding

3. **Multi-Task Learning**: Simultaneous classification + experience level prediction

4. **Active Learning**: Prioritize uncertain predictions for human labeling (especially Automobile, Apparel)

5. **Cross-Lingual**: Support resumes in multiple languages

6. **Skill Extraction**: Not just category, but list specific skills found

7. **Experience Level**: Junior/Mid/Senior classification

8. **Explainability**: Highlight which resume sections influenced prediction

9. **Category Refinement**: 
   - Split Automobile into subcategories (Engineering, Sales, Design)
   - Merge similar low-performing categories (Advocate + Public Relations)

10. **Integration**: Connect with ATS (Applicant Tracking Systems)

## API Deployment Example

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_api():
    data = request.json
    resume_text = data.get('resume_text', '')
    
    if not resume_text:
        return jsonify({'error': 'No resume text provided'}), 400
    
    category, confidence, top3 = classify_resume(resume_text)
    
    # Flag low-confidence predictions
    needs_review = confidence < 70
    
    return jsonify({
        'predicted_category': category,
        'confidence': round(confidence, 2),
        'needs_manual_review': needs_review,
        'top_3_predictions': [
            {'category': cat, 'probability': round(prob, 2)}
            for cat, prob in top3
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage**:
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Software engineer with Python..."}'
```

## References

- **DistilBERT Paper**: [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Dataset**: [Sneha Anbhawal (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Transformers Docs**: [Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## License

This project is licensed under the MIT License. DistilBERT is licensed under Apache 2.0.

## Acknowledgments

- **Dataset**: Sneha Anbhawal (Kaggle)
- **Model**: Hugging Face (DistilBERT)
- **Framework**: PyTorch + Transformers
- **Community**: Open-source contributors

---

**Task**: Multi-Class Text Classification (24 categories) | **Accuracy**: 87.67% | **Model**: DistilBERT | **Last Updated**: December 2024

**⚠️ Important Note**: This model is designed to assist HR professionals in initial resume screening. It should NOT be the sole basis for hiring decisions. Always combine AI predictions with human judgment, interviews, and comprehensive evaluation. The model may exhibit biases present in training data and should be regularly audited for fairness. Categories with lower performance (Automobile: 55%, Apparel: 70%, Agriculture: 69%) require additional manual review.