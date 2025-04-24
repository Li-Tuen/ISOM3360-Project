# Oral Cancer Survival Rate Prediction

This project predicts 5-year survival rates for oral cancer patients based on various clinical and demographic factors using machine learning.

## Features
- Predicts survival rate categories (<30%, 30-50%, 50-80%, 80-100%, 100%)
- Handles class imbalance using SMOTE
- Includes comprehensive model evaluation metrics
- Generates visualizations (confusion matrix, precision-recall curves, correlation heatmaps)

## Requirements
- Python 3.7+
- Libraries listed in requirements.txt

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run the prediction script: `python survival_rate_prediction.py`
3. Results will be saved as:
   - Model file: `survival_rate_model.pkl`
   - Test set: `test_set.csv`
   - Visualizations: `confusion_matrix.png`, `precision_recall_curve.png`, `correlation_heatmap_full.png`

## Data
The model uses the following features:
- Numerical:
  - Cost of Treatment (USD)
  - Tumor Size (cm)
  - Economic Burden (Lost Workdays per Year)
- Categorical:
  - Cancer Stage
  - Treatment Type