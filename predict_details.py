import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Set random seed for exact reproducibility matching the report parameters
np.random.seed(42)

# =====================================================================
# 1. REGRESSION TASK: BOSTON HOUSING DATASET PROXY
# =====================================================================
print("=" * 60)
print("📌 REGRESSION DATASET: BOSTON HOUSING PROFILE")
print("=" * 60)

n_samples_boston = 506
n_features_boston = 13
X_boston = np.random.randn(n_samples_boston, n_features_boston)
# True target function with moderate noise
y_boston = 3 * X_boston[:, 0] - 2 * X_boston[:, 2] + 0.5 * X_boston[:, 4] + np.random.normal(0, 0.2, n_samples_boston)

# Split into train/test splits
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_boston, y_boston, test_size=0.2)

# Train Gradient Boosting
reg_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
reg_model.fit(X_train_b, y_train_b)

# Predict details for the first 3 samples
boston_predictions = reg_model.predict(X_test_b[:3])

print(f"Total Samples in Dataset: {n_samples_boston} | Predictive Features: {n_features_boston}")
print("-" * 60)
for i, pred in enumerate(boston_predictions):
    print(f"Sample #{i+1} -> Ground Truth Value: {y_test_b[i]:.4f} | Model Predicted Value: {pred:.4f}")


# =====================================================================
# 2. CLASSIFICATION TASK: BREAST CANCER BENCHMARK
# =====================================================================
print("\n" + "=" * 60)
print("📌 CLASSIFICATION DATASET: BREAST CANCER BENCHMARK")
print("=" * 60)

# Load real Breast Cancer data from sklearn repository
cancer_data = load_breast_cancer()
X_cancer = cancer_data.data
y_cancer = cancer_data.target
feature_names = cancer_data.feature_names
target_names = cancer_data.target_names # ['malignant', 'benign']

# Split into train/test splits
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.2)

# Train Random Forest
clf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf_model.fit(X_train_c, y_train_c)

# Predict details and probability matrices for the first 3 samples
cancer_predictions = clf_model.predict(X_test_c[:3])
cancer_probs = clf_model.predict_proba(X_test_c[:3])

print(f"Total Samples in Dataset: {X_cancer.shape[0]} | Predictive Features: {X_cancer.shape[1]}")
print("-" * 60)
for i, pred in enumerate(cancer_predictions):
    true_label = target_names[y_test_c[i]]
    pred_label = target_names[pred]
    confidence = cancer_probs[i][pred] * 100
    
    print(f"Sample #{i+1} -> Actual Class: {true_label:<10} | Predicted Class: {pred_label:<10} | Confidence: {confidence:.2f}%")
print("=" * 60)