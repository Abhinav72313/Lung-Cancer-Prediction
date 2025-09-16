# Lung Cancer Survival Prediction

A comprehensive machine learning project that analyzes lung cancer patient data to predict survival outcomes. This project implements both traditional machine learning and deep learning approaches with extensive data analysis and preprocessing.


## 🎯 Overview

This project focuses on predicting lung cancer patient survival using comprehensive medical and demographic data. The analysis includes extensive exploratory data analysis (EDA), feature engineering, and implementation of both traditional machine learning (Logistic Regression) and deep learning (Neural Networks) approaches.

### Key Objectives:
- Predict patient survival outcomes based on medical and demographic features
- Identify key factors influencing survival rates
- Compare traditional ML vs. deep learning performance
- Handle class imbalance using SMOTE techniques
- Provide comprehensive data visualization and insights

## 📊 Dataset

The dataset (`dataset_med.csv`) contains comprehensive lung cancer patient records with medical, demographic, and treatment information.

### Target Variable:
- **survived** - Binary outcome indicating patient survival (0: No, 1: Yes)

### Key Features:

#### Continuous Variables:
- **age** - Patient age at diagnosis
- **bmi** - Body Mass Index
- **cholesterol_level** - Blood cholesterol levels
- **duration** - Treatment duration (calculated from diagnosis to end of treatment)

#### Categorical Variables:
- **gender** - Patient gender
- **country** - Country of treatment
- **cancer_stage** - Stage of cancer at diagnosis
- **smoking_status** - Smoking history/status
- **treatment_type** - Type of treatment received
- **family_history** - Family history of cancer
- **genetic_mutations** - Presence of genetic mutations
- **comorbidities** - Additional health conditions
- **insurance_coverage** - Insurance status
- **hospital_type** - Type of hospital (public/private)
- **response_to_treatment** - Patient's response to treatment

#### Temporal Features:
- **diagnosis_date** - Date of cancer diagnosis
- **end_treatment_date** - Date of treatment completion
- **id** - Unique patient identifier

## 🛠️ Features

### Data Analysis & Preprocessing:
- **Comprehensive EDA**: Distribution analysis, correlation studies, survival pattern analysis
- **Feature Engineering**: Treatment duration calculation from dates
- **Data Quality Assessment**: Missing value and duplicate detection
- **Feature Scaling**: StandardScaler for continuous variables
- **Label Encoding**: Categorical variable transformation
- **Feature Selection**: ExtraTreesClassifier-based importance ranking
- **Class Imbalance Handling**: SMOTENC for balanced training data

### Visualization:
- Distribution plots for continuous and categorical variables
- Correlation heatmaps
- Pairwise relationship analysis
- Survival analysis by demographics and medical factors
- Box plots for age, cholesterol, and other key metrics

### Deep Learning Models / Machine Learning Models:
- **Logistic Regression**: Baseline traditional ML model
- **Deep Neural Network**: Multi-layer PyTorch implementation with:
  - 7-layer architecture (2048 → 1024 → 512 → 256 → 128 → 64 → 32 → 1)
  - Batch normalization
  - Dropout regularization
  - Early stopping based on validation accuracy

## 📁 Project Structure

```
lung_cancer/
│
├── README.md                    # Project documentation
├── train.ipynb                  # Main Jupyter notebook with complete analysis
└── dataset_med.csv             # Medical dataset with patient records
```

## 🔬 Methodology

### 1. Data Exploration & Quality Assessment
- **Initial Analysis**: Dataset dimensions, missing values, duplicates
- **Feature Categorization**: Separation of continuous vs. categorical variables
- **Temporal Analysis**: Date processing and duration calculation
- **Distribution Analysis**: KDE plots for continuous features, count plots for categorical

### 2. Exploratory Data Analysis (EDA)
- **Correlation Analysis**: Heatmap of numerical feature relationships
- **Survival Patterns**: Analysis by demographics (age, country, gender)
- **Medical Factors**: Cancer stage, treatment type, and cholesterol level analysis
- **Risk Factors**: Smoking status and family history impact on survival

### 3. Data Preprocessing
- **Feature Engineering**: Treatment duration from diagnosis and end dates
- **Scaling**: StandardScaler for continuous variables
- **Encoding**: LabelEncoder for categorical variables
- **Feature Selection**: ExtraTreesClassifier for importance-based selection
- **Class Balancing**: SMOTENC for handling survival outcome imbalance

### 4. Model Development
- **Baseline Model**: Logistic Regression for interpretable results
- **Deep Learning**: Multi-layer neural network with:
  - Batch normalization for stable training
  - Dropout layers for regularization
  - Early stopping for optimal performance
- **Training Strategy**: GPU acceleration, batch processing, validation monitoring

### 5. Evaluation & Comparison
- **Multiple Metrics**: Comprehensive performance assessment
- **Confusion Matrix**: Per-class performance analysis
- **Model Comparison**: Traditional ML vs. deep learning approaches
- **Feature Importance**: Analysis of key survival predictors
