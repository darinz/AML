# Linear Models & Classical ML

This module covers traditional supervised learning approaches that form the backbone of applied machine learning.

## Learning Objectives

By the end of this module, you will be able to:
- Implement linear regression with various regularization techniques
- Build classification models using logistic regression, SVM, and Naïve Bayes
- Create decision trees and ensemble methods
- Understand the bias-variance tradeoff and model selection
- Apply these algorithms to real-world problems

## Topics Covered

### 1. Linear Regression & Regularization
- **Linear Regression**: Ordinary least squares, gradient descent implementation
- **Regularization**: Ridge (L2), Lasso (L1), and Elastic Net regularization
- **Feature Engineering**: Polynomial features, interaction terms, scaling
- **Model Evaluation**: R², MSE, MAE, cross-validation

### 2. Classification
- **Logistic Regression**: Binary and multiclass classification
- **Support Vector Machines (SVM)**: Linear and kernel SVMs
- **Naïve Bayes Classifier**: Gaussian, Multinomial, and Bernoulli variants
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, ROC curves

### 3. Trees
- **Decision Trees**: CART algorithm, information gain, Gini impurity
- **Ensemble Methods**: Bagging, Boosting (AdaBoost, Gradient Boosting)
- **Random Forests**: Feature importance, out-of-bag estimation
- **Advanced Ensembles**: XGBoost, LightGBM, CatBoost

## Practical Applications

- **Predictive Analytics**: Sales forecasting, customer churn prediction
- **Medical Diagnosis**: Disease classification, risk assessment
- **Financial Modeling**: Credit scoring, fraud detection
- **Marketing**: Customer segmentation, campaign optimization

## Implementation Focus

This module emphasizes **production-ready implementation**:
- Build linear models from scratch using numpy
- Implement gradient descent for optimization
- Code decision tree algorithms without using sklearn
- Create ensemble methods and understand their mechanics
- Optimize hyperparameters using grid search and cross-validation

## Key Concepts

- **Bias-Variance Tradeoff**: Understanding model complexity vs. generalization
- **Feature Selection**: Identifying important features and removing noise
- **Cross-Validation**: Proper model evaluation and selection
- **Overfitting Prevention**: Regularization, early stopping, validation sets

## Prerequisites

- Completion of Fundamentals & Basic Algorithms module
- Understanding of gradient descent and optimization
- Familiarity with probability and statistics

## Learning Resources

### Comprehensive Guides

This module includes detailed markdown guides for each topic:

1. **01-linear-regression-regularization.md** - Complete guide to linear regression and regularization techniques
   - Mathematical foundations and derivations
   - Implementation from scratch (normal equation and gradient descent)
   - Ridge, Lasso, and Elastic Net regularization
   - Feature engineering and polynomial regression
   - Cross-validation and model selection
   - Real-world applications

2. **02-classification.md** - Comprehensive guide to classification algorithms
   - Logistic regression (binary and multiclass)
   - Support Vector Machines (SVM) with different kernels
   - Naïve Bayes classifiers (Gaussian, Multinomial, Bernoulli)
   - Model evaluation metrics and ROC curves
   - Practical applications and comparisons

3. **03-trees-ensemble.md** - Complete guide to decision trees and ensemble methods
   - Decision tree implementation from scratch
   - Bagging (Bootstrap Aggregating)
   - Random Forest with feature importance
   - AdaBoost and other boosting algorithms
   - Model comparison and evaluation

### Python Examples

Each guide is accompanied by comprehensive Python examples:

- **linear_regression_examples.py** - Complete implementations and demonstrations
- **classification_examples.py** - Classification algorithms with practical examples
- **trees_ensemble_examples.py** - Decision trees and ensemble methods

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running Examples

Each example file can be run independently:

```bash
python linear_regression_examples.py
python classification_examples.py
python trees_ensemble_examples.py
```

## Key Learning Outcomes

By completing this module, you will:

- **Understand the mathematical foundations** of linear models and classification
- **Implement algorithms from scratch** using numpy and mathematical principles
- **Apply regularization techniques** to prevent overfitting
- **Build ensemble methods** and understand their mechanics
- **Evaluate model performance** using appropriate metrics
- **Solve real-world problems** with practical applications

## Practical Applications

The examples demonstrate real-world applications including:
- House price prediction using linear regression
- Spam detection with classification algorithms
- Credit card fraud detection with ensemble methods
- Customer churn prediction
- Medical diagnosis and risk assessment

## Next Steps

After completing this module, you'll be ready for **Probabilistic & Statistical Methods** where you'll learn about unsupervised learning and statistical modeling. 