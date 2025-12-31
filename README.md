# Practices-AI-ML

A collection of practical AI and machine learning projects demonstrating various applications of transformer models, neural networks, and NLP techniques.

## 📁 Projects Overview

### 🔍 [FakeNewsDetector](./FakeNewsDetector)
Two-stage fake news detection system combining BERT classification (82.34% accuracy) with LLM verification.
- **Tech**: BERT, Mistral-7B, OpenRouter API
- **Features**: Binary classification, explainable AI, confidence scoring
- **Use Case**: Content moderation, fact-checking assistance

### 📄 [ResumeParser](./ResumeParser)
AI-powered resume screening and job matching system with 87.67% accuracy.
- **Tech**: DistilBERT, Mistral-7B
- **Features**: 24 job categories, resume-JD matching, skills gap analysis
- **Use Case**: Automated recruitment, candidate screening

### 🌐 [Translator](./Translator)
English to Chinese neural machine translation using MarianMT.
- **Tech**: MarianMT (Helsinki-NLP)
- **Features**: Real-time translation, fine-tuned on domain data
- **Use Case**: Language translation, localization

### 💬 [Spam](./Spam)
YouTube comment spam detector achieving 99% accuracy.
- **Tech**: ALBERT
- **Features**: Real-time spam detection, parameter-efficient
- **Use Case**: Social media moderation, comment filtering

### 💰 [SalaryPrediction](./SalaryPrediction)
Comparative study of ML approaches for salary prediction.
- **Tech**: PyTorch Neural Networks, Scikit-learn Linear Regression
- **Features**: Multiple model comparison, feature engineering
- **Use Case**: HR analytics, compensation planning

### 🐾 [AnimalSpecies](./AnimalSpecies)
CNN-based animal classification using transfer learning.
- **Tech**: VGG16, TensorFlow/Keras, ImageDataGenerator
- **Performance**: 77.52% test accuracy on 10 classes
- **Features**: Pre-trained VGG16 backbone, data augmentation, dropout regularization
- **Use Case**: Wildlife monitoring, educational tools

### 🔎 [ObjectDetection](./ObjectDetection)
Real-time urban object detection with SSD300.
- **Tech**: SSD300, PyTorch, VGG16
- **Performance**: Multi-scale detection across 38 urban object classes
- **Features**: COCO pre-trained weights, anchor-based detection, mAP evaluation
- **Use Case**: Autonomous vehicles, smart city surveillance

## 🚀 Quick Start

Each project is self-contained in its own directory with:
- Complete source code
- Detailed README with setup instructions
- Training scripts and pre-trained models (where applicable)
- Example usage and demos

### General Requirements

```bash
# Core dependencies (varies by project)
pip install torch transformers datasets scikit-learn numpy pandas matplotlib
```

### Running a Project

```bash
# Navigate to project directory
cd FakeNewsDetector  # or any other project

# Follow project-specific README
python main.py
```

## 🛠️ Technology Stack

**Deep Learning Frameworks**:
- PyTorch
- Transformers (Hugging Face)
- TensorFlow (selected projects)

**Models Used**:
- BERT, DistilBERT, ALBERT
- MarianMT
- Mistral-7B (via OpenRouter)
- Custom Neural Networks

**Tools & Libraries**:
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Kaggle Hub (datasets)

## 📊 Project Comparison

| Project | Model | Accuracy | Type | Deployment |
|---------|-------|----------|------|------------|
| Fake News Detector | BERT | 82.34% | Binary Classification | CLI + API |
| Resume Parser | DistilBERT | 87.67% | Multi-class (24) | CLI |
| Spam Detector | ALBERT | 99.00% | Binary Classification | Model Only |
| Translator | MarianMT | - | Seq2Seq | CLI |
| Salary Prediction | NN + LR | - | Regression | Scripts |

## 🎯 Use Cases

**Content Moderation**:
- Fake news detection
- Spam filtering

**Human Resources**:
- Resume screening
- Salary predictions
- Job matching

**Language Processing**:
- Machine translation
- Text classification

**Research & Education**:
- Model comparison studies
- Transfer learning examples
- Production deployment patterns

## 📖 Learning Outcomes

This repository demonstrates:
- ✅ **Fine-tuning** pre-trained transformer models
- ✅ **Two-stage AI systems** (classifier + LLM)
- ✅ **Production deployment** patterns
- ✅ **Model comparison** (deep learning vs traditional ML)
- ✅ **API integration** (OpenRouter)
- ✅ **Data preprocessing** and feature engineering
- ✅ **Evaluation metrics** and performance monitoring

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, not required)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/YeeChen521/Practices-AI-ML.git
cd Practices-AI-ML

# Install dependencies (project-specific)
cd <project-name>
pip install -r requirements.txt  # if available
```

### Environment Variables

Some projects require API keys:

```bash
# Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

## 📚 Documentation

Each project contains:
- **README.md**: Detailed project documentation
- **Source code**: Well-commented implementation
- **Training scripts**: Model training pipelines
- **Usage examples**: Demo code and CLI tools

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome!

## 📝 License

Individual projects may have different licenses. Check each project's README for details.

## 🙏 Acknowledgments

**Datasets**:
- Kaggle community datasets
- Public domain sources

**Models**:
- Google Research (BERT)
- Hugging Face (DistilBERT, ALBERT)
- Helsinki-NLP (MarianMT)
- Mistral AI (Mistral-7B)

**Frameworks**:
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn

## 📬 Contact

**Author**: YeeChen521  
**Repository**: [Practices-AI-ML](https://github.com/YeeChen521/Practices-AI-ML)

---

⭐ **Star this repo** if you find it helpful!

**Last Updated**: December 2024