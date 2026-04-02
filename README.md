# Titanic Survival Prediction — Machine Learning Pipeline

## Overview

This project builds a complete machine learning pipeline to predict passenger survival on the Titanic dataset. The goal is to apply practical ML concepts end-to-end — from raw data preprocessing to model evaluation and final prediction generation.

The pipeline focuses on:
- Avoiding data leakage
- Maintaining consistent preprocessing between train and test sets
- Comparing multiple models in a statistically sound way

Two models are implemented and evaluated:
- Logistic Regression
- Random Forest

---

## Problem

Given passenger data (age, sex, class, etc.), predict whether a passenger survived the Titanic disaster.

This is a **binary classification problem**:
- `0` → Did not survive
- `1` → Survived

---

## Approach

The project follows a structured ML workflow:

1. Data loading  
2. Preprocessing (handling missing values, encoding)  
3. Feature alignment between train and test  
4. Model training  
5. Cross-validation  
6. Model comparison using statistical testing  
7. Final predictions export  

---

## Workflow

### 1. Data Loading
- Load training and test datasets  
- Separate features (`X`) and target (`y`)  

---

### 2. Preprocessing

#### Missing Values
- Numerical features → filled with median  
- Categorical features → filled with most frequent value  

#### Encoding
- Categorical variables converted using one-hot encoding (`pd.get_dummies`)  
- Ensures models can process non-numeric data  

---

### 3. Feature Alignment (Critical Step)

Train and test datasets often end up with different columns after encoding.

To fix this:
- Align test set to match training feature space  
- Missing columns in test → filled with `0`  

This prevents:
- Shape mismatches  
- Inconsistent model input  

---

### 4. Train/Test Split
- Training data is split into:
  - Training set  
  - Validation set  

Used for:
- Model evaluation before final prediction  

---

### 5. Model Training

#### Logistic Regression
- Baseline linear model  
- Fast and interpretable  

#### Random Forest
- Ensemble method  
- Handles non-linear relationships better  

---

### 6. Cross-Validation
- K-fold cross-validation used for robust evaluation  
- Produces multiple accuracy scores instead of a single split  

**Example:**
- Scores like `[0.80, 0.82, 0.81, ...]`  
- Mean accuracy gives overall performance  

---

### 7. Model Comparison (Statistical Testing)

A **t-test** is used to compare models:
- Checks if performance difference is statistically significant  

**Interpretation:**
- High p-value → difference is likely due to randomness  
- Low p-value → real performance difference  

---

### 8. Prediction

Final steps:
- Train model on full training data  
- Predict on test dataset  
- Save results as:

```bash
submission.csv
```

**Format:**
- `PassengerId`  
- `Survived`  

---

## Results

- Logistic Regression and Random Forest achieved similar performance  
- Statistical test showed:
  - No significant difference between models  
- Indicates both models generalize similarly on this dataset  

---

## Key Learnings

- Importance of **feature consistency** between train and test  
- How to properly avoid **data leakage**  
- Why **cross-validation** is better than a single split  
- How to compare models beyond accuracy using **statistical tests**  
- Practical implementation of an end-to-end ML pipeline  

---

## Notes

- The pipeline is designed to be simple but correct  
- Focus is on **clean ML workflow**, not over-engineering  
- Easily extendable with:
  - Feature engineering  
  - Hyperparameter tuning  
  - Additional models  # Titanic-Survival-Prediction
# Titanic-Survival-Prediction
