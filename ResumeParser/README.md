# AI Resume-Job Description Matching System

An intelligent resume screening system combining **DistilBERT classification** (87.67% accuracy) with **LLM-powered analysis** using Mistral-7B to match resumes against job descriptions and provide actionable hiring insights.

## Overview

This system revolutionizes resume screening by providing:
1. **Automated Job Category Classification** (24 categories)
2. **Resume-JD Match Scoring** (0-100%)
3. **Skills Gap Analysis** (identifying missing skills)
4. **Actionable Improvement Suggestions**
5. **Comprehensive Hiring Insights**

Perfect for HR professionals, recruiters, and hiring managers who need to:
- Screen hundreds of resumes efficiently
- Identify the best candidates quickly
- Provide constructive feedback to applicants
- Reduce unconscious bias in hiring

## System Architecture

```
┌──────────────────┐
│  Resume Input    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│  DistilBERT Classifier      │
│  (87.67% accuracy)          │
│  Predicts: HR, IT, Finance, │
│  Engineering, etc. (24 cat) │
└────────┬────────────────────┘
         │
         │ Category + Confidence
         ▼
┌──────────────────┐
│ Job Description  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│   LLM Analyzer              │
│   (Mistral-7B)              │
│   - Match Scoring           │
│   - Strengths Analysis      │
│   - Missing Skills          │
│   - Improvement Suggestions │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Hiring Decision Report    │
│   - Match Score: X%         │
│   - Strengths: [...]        │
│   - Missing Skills: [...]   │
│   - Improvements: [...]     │
│   - Overall Feedback        │
└─────────────────────────────┘
```

## Performance Metrics

### DistilBERT Classifier
- **Test Accuracy**: 87.67%
- **Macro F1**: 87%
- **Macro Precision**: 89%
- **Macro Recall**: 87%

### Category-Specific Performance

**Excellent (F1 > 0.95)**:
- HR, Finance, Accountant, Business Development (100%)
- Chef, Consultant, Engineering (97%)

**Good (F1: 0.80-0.95)**:
- IT, Teacher, Healthcare, Construction, Sales (88-96%)

**Challenging (F1 < 0.80)**:
- Automobile (55%), Apparel (70%), Agriculture (69%)
- These categories require additional manual review

## Key Features

### 1. **Intelligent Resume Classification**
```python
Category: Information-Technology
Confidence: 92.45%
```
- Automatically categorizes resumes into 24 job categories
- High confidence predictions (>80%) for most categories
- Fast classification (<100ms per resume)

### 2. **Resume-JD Match Scoring**
```json
{
  "match_score": 85,
  "strengths": [
    "5 years Python experience matches requirement",
    "ML/AI expertise aligns with role needs",
    "Strong technical background in data science"
  ],
  "missing_skills": [
    "Kubernetes/Docker experience not mentioned",
    "AWS certification preferred but not listed"
  ],
  "resume_improvements": [
    "Quantify achievements (e.g., 'Improved model accuracy by 20%')",
    "Add specific tools/frameworks used in each project",
    "Include leadership experience in team management"
  ],
  "overall_feedback": "Strong technical match with 85% alignment..."
}
```

### 3. **Skills Gap Analysis**
- Identifies exactly what candidates are missing
- Helps prioritize training needs
- Guides interview focus areas

### 4. **Actionable Feedback**
- Specific improvement suggestions
- Quantifiable recommendations
- Helps candidates strengthen applications

## Why This Architecture?

### Why DistilBERT for Stage 1?

#### **Fast & Accurate Initial Classification**
- 87.67% accuracy across 24 job categories
- 40% smaller than BERT (66M vs 110M parameters)
- 60% faster inference time
- Perfect for high-volume screening

#### **Multi-Class Excellence**
- Handles 24 distinct job categories
- Learns subtle differences between similar roles
- Strong performance on technical roles (IT, Engineering)
- Robust to diverse resume formats

#### **Production-Ready**
- Low memory footprint
- Fast inference (<100ms)
- GPU/CPU support
- Batch processing capable

### Why LLM (Mistral-7B) for Stage 2?

#### **Deep Resume-JD Analysis**
- Understands job requirements semantically
- Compares qualifications vs requirements
- Identifies both technical and soft skills
- Provides human-interpretable reasoning

#### **Structured Feedback Generation**
- Match scoring (0-100%)
- Strengths identification
- Skills gap analysis
- Actionable recommendations

#### **Mistral-7B Advantages**
- Free tier via OpenRouter
- Strong instruction following
- Consistent JSON output
- Good balance of speed/quality
- Understands HR/recruitment context

### Why Two Stages?

| Aspect | DistilBERT Only | LLM Only | DistilBERT + LLM |
|--------|-----------------|----------|------------------|
| **Speed** | Very Fast | Slow | Fast initial + targeted analysis |
| **Accuracy** | 87.67% category | No category | Category + detailed matching |
| **Feedback** | None | Full analysis | Category + comprehensive feedback |
| **Cost** | Low | High per resume | Moderate (selective LLM use) |
| **Scalability** | Excellent | Poor | Good (batch BERT, selective LLM) |
| **Explainability** | Confidence only | Full reasoning | Both statistical + reasoning |

**Result**: Fast screening (DistilBERT) + deep analysis (LLM) = Best hiring decisions

## Installation

### Requirements

```bash
pip install torch transformers datasets evaluate kagglehub scikit-learn numpy pandas matplotlib requests python-dotenv
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
requests>=2.28.0
python-dotenv>=1.0.0
```

### Setup

1. **Train the DistilBERT model** (or download pre-trained):
```bash
jupyter notebook training_notebook.ipynb
# Run all cells to train and save to ./final_model
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
=== Resume—JD Matching Assistant ===

Paste your resume here: 

Software Engineer with 5 years of experience in Python, machine learning, and web 
development. Proficient in TensorFlow, PyTorch, Django, and React. Led development 
of ML-powered recommendation system. Strong background in data structures, 
algorithms, and system design. Bachelor's in Computer Science.

Paste the job description here: 

Senior ML Engineer position requiring 5+ years Python experience, expertise in 
TensorFlow/PyTorch, experience with cloud platforms (AWS/GCP), and proven track 
record in deploying ML models to production. Docker/Kubernetes knowledge preferred.

=== Resume Classification ===
Predicted Category: Information-Technology
Confidence: 92.45%

=== Job Match Feedback ===
Match Score: 78%

Strengths:
- 5 years Python experience directly matches requirement
- TensorFlow and PyTorch expertise aligns perfectly with role
- ML background with recommendation system shows practical experience
- Strong CS fundamentals mentioned (data structures, algorithms)

Missing Skills:
- Cloud platform experience (AWS/GCP) not mentioned
- Docker/Kubernetes containerization knowledge absent
- Production deployment experience unclear
- Specific details on ML model deployment pipeline missing

Resume Improvement Suggestions:
- Add cloud platform certifications or project experience (AWS/GCP)
- Include containerization tools (Docker, Kubernetes) if used
- Quantify achievements: "Improved recommendation accuracy by X%"
- Specify scale of systems worked on (users, data volume, throughput)
- Add production ML deployment details and monitoring experience

Overall Feedback:
Strong technical foundation with 78% match. Primary gaps are in cloud infrastructure 
and production deployment experience. Consider highlighting any cloud platform usage 
or adding relevant certifications before applying.
```

### Programmatic Usage

```python
from main import predict_category, call_openrouter, build_prompt

# Stage 1: Classify resume
resume_text = "Your resume text here..."
category, confidence = predict_category(resume_text)

print(f"Category: {category}")
print(f"Confidence: {confidence*100:.2f}%")

# Stage 2: Match against job description
jd_text = "Job description text here..."
prompt = build_prompt(resume_text, jd_text, category, confidence)
feedback = call_openrouter(prompt)

print(f"Match Score: {feedback['match_score']}%")
print(f"Strengths: {feedback['strengths']}")
print(f"Missing Skills: {feedback['missing_skills']}")
print(f"Improvements: {feedback['resume_improvements']}")
```

### Batch Processing Multiple Resumes

```python
def screen_resumes_for_job(resumes, job_description, min_match_score=70):
    """
    Screen multiple resumes against a single job description
    
    Args:
        resumes: List of resume texts
        job_description: JD text
        min_match_score: Minimum match score threshold (0-100)
        
    Returns:
        List of dicts with resume analysis
    """
    results = []
    
    for i, resume in enumerate(resumes):
        # Stage 1: Quick classification
        category, confidence = predict_category(resume)
        
        # Stage 2: Detailed matching
        prompt = build_prompt(resume, job_description, category, confidence)
        feedback = call_openrouter(prompt)
        
        if feedback['match_score'] >= min_match_score:
            results.append({
                'resume_id': i,
                'category': category,
                'confidence': confidence,
                'match_score': feedback['match_score'],
                'strengths': feedback['strengths'],
                'missing_skills': feedback['missing_skills'],
                'improvements': feedback['resume_improvements'],
                'overall': feedback['overall_feedback']
            })
    
    # Sort by match score (highest first)
    results.sort(key=lambda x: x['match_score'], reverse=True)
    return results

# Example: Screen 50 resumes for a job
top_candidates = screen_resumes_for_job(
    resumes=all_resumes,
    job_description=jd_text,
    min_match_score=75
)

print(f"Found {len(top_candidates)} candidates above 75% match")
for i, candidate in enumerate(top_candidates[:10], 1):
    print(f"{i}. Match: {candidate['match_score']}% - {candidate['category']}")
```

## System Components

### 1. Resume Classifier (`predict_category`)

**Purpose**: Fast job category prediction

```python
def predict_category(resume_text):
    inputs = tokenizer(
        resume_text,
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
    
    return JOB_LIST[pred_idx], confidence
```

**Features**:
- Processes 512 tokens (full resumes)
- Returns category + confidence
- GPU accelerated
- <100ms inference time

### 2. Resume-JD Matcher (`call_openrouter`)

**Purpose**: Deep matching analysis and feedback

**Prompt Engineering**:
```python
def build_prompt(resume, jd, category, confidence):
    return f"""
    You are an AI resume evaluation assistant.
    The resume was classified as "{category}" with {confidence:.2f} confidence.
    
    Analyze the resume against the job description.
    
    Resume: "{resume}"
    Job Description: "{jd}"
    
    Return ONLY valid JSON:
    {{
        "match_score": 0-100,
        "strengths": [list of strengths],
        "missing_skills": [list of gaps],
        "resume_improvements": [list of suggestions],
        "overall_feedback": "comprehensive summary"
    }}
    """
```

**Why This Prompt Works**:
- Clear role and task definition
- Structured output format (JSON)
- Explicit analysis dimensions
- Temperature=0.3 for consistency
- Prevents verbose explanations

### 3. JSON Extraction (`extract_json`)

```python
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())
```

**Robust Parsing**:
- Handles markdown code blocks
- Extracts JSON from verbose responses
- Regex-based for reliability
- Clear error messages

## Training the Model

### Dataset

**Source**: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

**Categories** (24):
```
HR, Designer, Information-Technology, Teacher, Advocate, 
Business-Development, Healthcare, Fitness, Agriculture, BPO, 
Sales, Consultant, Digital-Media, Automobile, Chef, Finance, 
Apparel, Engineering, Accountant, Construction, Public-Relations, 
Banking, Arts, Aviation
```

**Statistics**:
```
After Oversampling:
- Total: 5,000 samples
- Per category: ~208 samples (balanced)
- Train: 3,500 (70%)
- Validation: 750 (15%)
- Test: 750 (15%)
```

### Training Configuration

```python
Model: distilbert-base-uncased
Max Length: 512 tokens (full resumes)
Batch Size: 16
Learning Rate: 5e-5
Epochs: 4
Weight Decay: 0.05
Dropout: 0.3
Weighted Loss: Yes (addresses class imbalance)
```

### Advanced Techniques

#### **Oversampling for Class Balance**
```python
# Balance all classes to maximum count
for label in label_count.keys():
    class_dataset = full_dataset.filter(lambda x: x["label"] == label)
    repeat_factor = max_count // len(class_dataset) + 1
    oversampled_class_dataset = concatenate_datasets([class_dataset] * repeat_factor)
```

#### **Weighted Cross-Entropy Loss**
```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weight inversely proportional to class frequency
        loss_fct = CrossEntropyLoss(weight=class_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

**Why weighted loss?**
- Double protection against imbalance
- Penalizes minority class errors more
- Improves macro F1 score
- Fair performance across all categories

### Training Results

```
Epoch 1: Loss 2.848, Val Acc: 52.28%, Val F1: 40.86%
Epoch 2: Loss 1.518, Val Acc: 78.28%, Val F1: 75.54%
Epoch 3: Loss 0.793, Val Acc: 87.13%, Val F1: 86.60%
Epoch 4: Loss 0.503, Val Acc: 87.94%, Val F1: 87.55%

Final Test Accuracy: 87.67%
Training Time: ~3 minutes on GPU
```

## Real-World Applications

### 1. **High-Volume Recruitment**

```python
def screen_applications(job_posting_id, resume_database):
    """Screen 1000+ resumes in minutes"""
    jd = get_job_description(job_posting_id)
    resumes = get_all_applications(job_posting_id)
    
    # Stage 1: Fast DistilBERT classification
    classified = [(r, *predict_category(r)) for r in resumes]
    
    # Filter by category match
    relevant = [r for r, cat, conf in classified 
                if cat in jd['target_categories'] and conf > 0.7]
    
    # Stage 2: LLM detailed matching (only top candidates)
    top_matches = []
    for resume in relevant[:50]:  # Top 50 by confidence
        feedback = call_openrouter(build_prompt(resume, jd['text'], ...))
        if feedback['match_score'] >= 75:
            top_matches.append((resume, feedback))
    
    return sorted(top_matches, key=lambda x: x[1]['match_score'], reverse=True)
```

### 2. **Candidate Feedback System**

```python
def provide_application_feedback(resume_text, job_description):
    """Give candidates actionable feedback"""
    category, confidence = predict_category(resume_text)
    prompt = build_prompt(resume_text, job_description, category, confidence)
    feedback = call_openrouter(prompt)
    
    return {
        'suitable': feedback['match_score'] >= 60,
        'match_percentage': feedback['match_score'],
        'your_strengths': feedback['strengths'],
        'skills_to_develop': feedback['missing_skills'],
        'how_to_improve': feedback['resume_improvements'],
        'interviewer_feedback': feedback['overall_feedback']
    }
```

### 3. **Internal Mobility & Career Pathing**

```python
def find_internal_opportunities(employee_resume, open_positions):
    """Match employees to internal job openings"""
    opportunities = []
    
    for position in open_positions:
        feedback = call_openrouter(
            build_prompt(employee_resume, position['jd'], ...)
        )
        
        opportunities.append({
            'role': position['title'],
            'department': position['department'],
            'match_score': feedback['match_score'],
            'transferable_skills': feedback['strengths'],
            'skills_to_learn': feedback['missing_skills'],
            'development_plan': feedback['resume_improvements']
        })
    
    return sorted(opportunities, key=lambda x: x['match_score'], reverse=True)
```

### 4. **Talent Pool Management**

```python
def build_talent_pool(resumes):
    """Categorize and analyze talent database"""
    pool = {}
    
    for resume in resumes:
        category, confidence = predict_category(resume)
        
        if category not in pool:
            pool[category] = []
        
        pool[category].append({
            'resume': resume,
            'confidence': confidence,
            'indexed_at': datetime.now()
        })
    
    # Generate pool statistics
    stats = {cat: len(resumes) for cat, resumes in pool.items()}
    return pool, stats
```

## Performance Optimization

### For Faster Processing

```python
# Batch DistilBERT predictions
def predict_batch(resumes):
    inputs = tokenizer(
        resumes,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    return [(JOB_LIST[p.argmax()], p.max().item()) 
            for p in probs]

# Selective LLM calls
def smart_matching(resume, jd, category, confidence):
    # Only use LLM for high-confidence, relevant candidates
    if confidence < 0.6:
        return {"match_score": 0, "reason": "Low confidence category"}
    
    return call_openrouter(build_prompt(resume, jd, category, confidence))
```

### For Better Matching Quality

```python
# Use stronger LLM for important positions
OPENROUTER_MODEL = "anthropic/claude-3-sonnet"  # Higher quality

# Increase temperature for more diverse feedback
payload["temperature"] = 0.5

# Multi-model ensemble
def ensemble_matching(resume, jd):
    models = ["mistral-7b", "llama-2-13b", "claude-3-haiku"]
    scores = []
    
    for model in models:
        feedback = call_openrouter_with_model(resume, jd, model)
        scores.append(feedback['match_score'])
    
    return {
        'average_score': np.mean(scores),
        'consensus': np.std(scores) < 10  # High agreement
    }
```

### For Production Deployment

```python
# Async processing
import asyncio
import aiohttp

async def async_screen_resumes(resumes, jd):
    # DistilBERT in sync (fast enough)
    classified = [predict_category(r) for r in resumes]
    
    # LLM calls in parallel
    tasks = [
        async_call_openrouter(build_prompt(r, jd, cat, conf))
        for r, cat, conf in classified
    ]
    
    feedbacks = await asyncio.gather(*tasks)
    return list(zip(resumes, feedbacks))

# Caching frequent JDs
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_jd_analysis(jd_hash):
    # Cache JD embeddings or analysis
    return analyze_jd(jd_hash)

# Rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=30, period=60)
def rate_limited_openrouter(prompt):
    return call_openrouter(prompt)
```

### Privacy Protection

**Data Handling**:
- Encrypt resume data in transit/storage
- Clear data retention policies
- GDPR/privacy law compliance
- Candidate consent required

**API Security**:
```python
# Don't log sensitive data
def secure_call_openrouter(prompt):
    # Remove PII before API call
    sanitized_prompt = remove_pii(prompt)
    
    response = call_openrouter(sanitized_prompt)
    
    # Don't store API responses long-term
    return response  # Process and discard
```
## Future Improvements

1. **Advanced NLP**:
   - Named Entity Recognition (skills, companies, degrees)
   - Semantic similarity scoring
   - Keyword extraction and matching

2. **Multi-Modal Analysis**:
   - Portfolio/work samples evaluation
   - Video resume analysis
   - GitHub/LinkedIn integration

3. **Predictive Analytics**:
   - Success probability prediction
   - Flight risk assessment
   - Cultural fit scoring

4. **Interactive Features**:
   - Chatbot for candidate questions
   - Interview question generation
   - Automated scheduling

5. **Integration Capabilities**:
   - ATS (Applicant Tracking System) integration
   - LinkedIn/Indeed API connections
   - Calendar and email automation

6. **Enhanced Feedback**:
   - Detailed skill gap roadmaps
   - Course recommendations
   - Timeline to job-readiness

## References

- **DistilBERT Paper**: [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- **Mistral-7B**: [Mistral 7B](https://arxiv.org/abs/2310.06825)
- **Dataset**: [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **OpenRouter**: [OpenRouter AI](https://openrouter.ai/)

## License

This project is licensed under the MIT License.
- DistilBERT: Apache 2.0
- Mistral-7B: Apache 2.0
- Dataset: Subject to Kaggle terms

## Acknowledgments

- **Dataset**: Sneha Anbhawal (Kaggle)
- **DistilBERT**: Hugging Face
- **Mistral**: Mistral AI
- **API**: OpenRouter
- **Community**: Open-source contributors

---

**System**: Resume-JD Matching (DistilBERT + LLM) | **Accuracy**: 87.67% | **Categories**: 24 | **Last Updated**: December 2025

**⚠️ Disclaimer**: This system is designed to assist HR professionals in resume screening. It should NOT be the sole basis for hiring decisions. Always combine AI predictions with human judgment, interviews, credential verification, and comprehensive candidate evaluation. The system may exhibit biases from training data and should be regularly audited for fairness. Low-performing categories (Automobile, Apparel, Agriculture) require additional manual review.