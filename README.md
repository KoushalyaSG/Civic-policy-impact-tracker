# Civic Policy Impact Tracker

## Project Overview
The **Civic Policy Impact Tracker** is a Natural Language Processing (NLP) project designed to classify civic policy descriptions into two categories:
- **High Impact**
- **Low Impact**

The objective of this project is to analyze policy text and predict its potential impact on civic transparency and governance using modern machine learning techniques.

This project implements two different modeling approaches:
1. **Hybrid Model**: BERT embeddings with Logistic Regression
2. **Fine-Tuned Transformer Model**: Fine-tuned BERT for sequence classification

## Project Structure
```text
Civic-policy-impact-tracker/
├── data/
│   ├── raw/                  # Original, untouched datasets
│   │   ├── collected dataset.csv
│   │   ├── india_policy_impact.csv
│   │   └── Lok_Sabha_Governmentt_Bills.csv
│   └── pre_processed/        # Cleaned and feature-engineered data
│       ├── cleaned_policy_dataset.csv
│       ├── feature_engineered_dataset.csv
│       └── preprocessed_policy_dataset.csv
├── notebooks/                # Experimentation and Model Development
│   ├── data_cleaning.ipynb
│   ├── Text Preprocessing.ipynb
│   ├── Feature Engineering.ipynb
│   ├── logistic_regression_model.ipynb
│   ├── neural network model.ipynb
│   ├── Hybrid_model_with_fine_tunning .ipynb
│   └── eda_3000_analysis.py
├── visualization/            # Generated plots and dashboard assets
│   └── OP1_descriptive_statistics.png
├── documentation/            # Project-related documents
├── meeting_screenshots/      # Visual record of meetings
└── README.md                 # Project overview
```

## Project Pipeline
The project follows a complete machine learning workflow:
1. **Dataset Collection**
2. **Data Inspection**
3. **Exploratory Data Analysis (EDA)**
4. **Data Cleaning (Duplicate Removal)**
5. **Feature Engineering**
6. **Train-Test Split**
7. **Model Development**
8. **Model Evaluation**
9. **Model Comparison**

## Dataset Description
The dataset consists of civic policy descriptions labeled according to their impact.

| Attribute | Description |
|-----------|-------------|
| `policy_text` | Textual description of the policy |
| `label` | Impact classification (High Impact / Low Impact) |

- **Initial Dataset Size**: 3000 policy records
- **After Data Cleaning**: 1249 unique policy descriptions
*Duplicate policy texts were removed to prevent data leakage.*

## Exploratory Data Analysis
EDA highlights:
- Label distribution analysis
- Text length and word count distribution
- Key observations: Most descriptions are 14–18 words; structured and short.

## Models Implemented
### 1. Hybrid Model (BERT + Logistic Regression)
Uses BERT as a feature extractor (768-dimensional embeddings) followed by a Logistic Regression classifier.

### 2. Fine-Tuned BERT Model
Fine-tunes a pretrained BERT model specifically for the civic policy classification task.

## Results
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| BERT + Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Fine-Tuned BERT | 1.00 | 1.00 | 1.00 | 1.00 |

## Technologies Used
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, PyTorch, HuggingFace Transformers, Matplotlib, Seaborn
- **Environment**: Google Colab

## How to Run
1. Open the project notebook in Google Colab.
2. Upload the dataset: `india_policy_impact_dataset_3000.csv`
3. Run cells sequentially from inspection to evaluation.

## Future Improvements
- Expanding with real-world policy documents.
- Applying cross-validation.
- Deploying as an API for real-time classification.
