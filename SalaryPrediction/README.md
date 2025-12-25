# Salary Prediction Project

A comparative study of machine learning approaches for predicting salaries based on employee characteristics using both Linear Regression and Neural Networks.

## Overview

This project implements two different machine learning approaches to predict employee salaries:
1. **Neural Network** using PyTorch
2. **Linear Regression** using scikit-learn

Both models are trained on the same dataset and their performances are compared to understand the trade-offs between traditional statistical methods and deep learning approaches.

## Dataset

**Source**: [Salary Prediction for Beginner Dataset](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer) (Kaggle)

**Features**:
- Years of Experience
- Age
- Gender
- Education Level (Bachelor's, Master's, PhD)
- Job Title

**Target Variable**: Salary

## Project Structure

```
.
├── NeuralNetwork.py    # PyTorch implementation
├── linearRegression.py # scikit-learn implementation
└── README.md           # This file
```

## Approach 1: Neural Network (PyTorch)

### Architecture
- **Input Layer**: Variable size (depends on one-hot encoded features)
- **Hidden Layer**: 8 neurons with ReLU activation
- **Output Layer**: 1 neuron (salary prediction)

### Key Features
- Feature scaling using StandardScaler for both inputs and outputs
- One-hot encoding for categorical variables (Gender, Education Level, Job Title)
- Adam optimizer with learning rate of 0.01
- Mean Squared Error (MSE) loss function
- 100 training epochs
- 75/25 train-test split

### Training Process
The model uses:
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: MSE
- **Batch Processing**: Full batch gradient descent
- **Validation**: Separate test set evaluation at each epoch

### Results
Training progress is displayed every 10 epochs showing both training and test loss.

## Approach 2: Linear Regression (scikit-learn)

### Preprocessing
- Forward-fill for missing values
- Binary encoding for Gender (Male: 0, Female: 1)
- Ordinal encoding for Education Level (Bachelor's: 0, Master's: 1, PhD: 2)
- One-hot encoding for Job Title (drop first category to avoid multicollinearity)

### Model
Standard Linear Regression with ordinary least squares optimization.

### Evaluation Metrics
- **R² Score**: Model's coefficient of determination
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **MSE** (Mean Squared Error): Average squared difference
- **RMSE** (Root Mean Squared Error): Square root of MSE

### Visualization
Includes scatter plot comparing actual vs predicted salaries with a diagonal reference line.

## Installation

### Requirements
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn kagglehub
```

### Dependencies
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- kagglehub

## Usage

### Running the Neural Network Model
```bash
python NeuralNetwork.py
```

### Running the Linear Regression Model
```bash
python linearRegression.py
```

Both scripts will:
1. Automatically download the dataset from Kaggle
2. Preprocess the data
3. Train the model
4. Display results and visualizations

## Key Differences Between Approaches

| Aspect | Neural Network | Linear Regression |
|--------|---------------|-------------------|
| **Complexity** | Can learn non-linear patterns | Assumes linear relationships |
| **Feature Scaling** | Required (StandardScaler) | Not strictly required |
| **Training** | Iterative (100 epochs) | Direct solution (OLS) |
| **Interpretability** | Black box | Transparent coefficients |
| **Overfitting Risk** | Higher (needs regularization) | Lower with fewer features |
| **Computational Cost** | Higher | Lower |

## Model Selection Notes

From the neural network code comments:
> "with only one ReLu layer performs the best result in the test loss"

This suggests that for this particular dataset, a simpler architecture works better, possibly because:
- The relationship between features and salary may be relatively linear
- The dataset size may not justify deeper architectures
- Simpler models generalize better on this problem

## Future Improvements

1. **Hyperparameter Tuning**: Grid search or random search for optimal parameters
2. **Cross-Validation**: K-fold cross-validation for more robust evaluation
3. **Feature Engineering**: Creating interaction terms or polynomial features
4. **Regularization**: Add L1/L2 regularization to prevent overfitting
5. **Ensemble Methods**: Try Random Forest or Gradient Boosting
6. **Feature Importance**: Analyze which features contribute most to predictions

## Visualizations

Both approaches include visualizations:
- Neural Network: Scatter plot of predictions vs actual (by Years of Experience)
- Linear Regression: Actual vs Predicted salary plot with regression line

## License

This project uses the Kaggle dataset which is subject to its own license terms.

## Acknowledgments

- Dataset by Rkiattisak on Kaggle
- Built with PyTorch and scikit-learn frameworks