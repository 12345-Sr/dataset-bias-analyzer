import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

def analyze_dataset_bias(df, target_column=None):
    """
    Scans a dataset for structural and statistical biases.
    """
    print("=" * 60)
    print("📊 LIVE DATASET BIAS AUDIT REPORT")
    print("=" * 60)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    print(f"Dataset Dimensions: {total_rows} Rows | {total_cols} Columns\n")
    
    # -----------------------------------------------------------------
    # 1. MISSING DATA BIAS (Sparsity Check)
    # -----------------------------------------------------------------
    print("🔍 1. Structural Bias (Missing Value Sparsity)")
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"⚠️ WARNING: Found {total_missing} total missing values in the dataset.")
        for col, count in missing_counts[missing_counts > 0].items():
            pct = (count / total_rows) * 100
            print(f"   -> Column '{col}': {count} missing fields ({pct:.2f}%)")
        print("💡 Implication: Biased tracking. Models will drop rows or require imputation.")
    else:
        print("✅ SUCCESS: No missing or null values detected across columns.")
        
    print("-" * 60)

    # -----------------------------------------------------------------
    # 2. TARGET CLASS IMBALANCE BIAS (For Classification)
    # -----------------------------------------------------------------
    if target_column and target_column in df.columns:
        print(f"🎯 2. Label Bias (Class Imbalance on '{target_column}')")
        class_counts = df[target_column].value_counts()
        
        if len(class_counts) == 2:  # Binary classification
            c1_name, c2_name = class_counts.index[0], class_counts.index[1]
            c1_pct = (class_counts.values[0] / total_rows) * 100
            c2_pct = (class_counts.values[1] / total_rows) * 100
            print(f"   -> Label '{c1_name}': {class_counts.values[0]} samples ({c1_pct:.1f}%)")
            print(f"   -> Label '{c2_name}': {class_counts.values[1]} samples ({c2_pct:.1f}%)")
            
            # Check deviation ratio from uniform 50/50 balance
            if abs(c1_pct - 50) > 15:
                print("⚠️ CRITICAL WARNING: Severe target class imbalance detected!")
                print("💡 Implication: High Variance risk. The model will be heavily biased toward predicting the majority class.")
            else:
                print("✅ SUCCESS: Target labels are evenly distributed.")
    else:
        print("📝 Note: No target class provided for explicit label bias verification.")
        
    print("-" * 60)

    # -----------------------------------------------------------------
    # 3. DISTRIBUTION SKEW BIAS (Outlier Check)
    # -----------------------------------------------------------------
    print("📉 3. Distributional Bias (Feature Skewness)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_features_count = 0
    
    for col in numeric_cols:
        if col == target_column:
            continue
        # Use pandas built-in Fisher-Pearson skewness coefficient
        skew_val = df[col].skew()
        if abs(skew_val) > 1.5:
            skewed_features_count += 1
            direction = "Right-skewed (Long tail positive)" if skew_val > 0 else "Left-skewed (Long tail negative)"
            print(f"   ⚠️ Skewed Feature '{col}': Coeff {skew_val:.2f} -> {direction}")
            
    if skewed_features_count == 0:
        print("✅ SUCCESS: All continuous variables show normally balanced distributions.")
    else:
        print(f"\n💡 Implication: Found {skewed_features_count} heavily skewed inputs. Linear models will underperform due to unhandled outlier variances.")
        
    print("=" * 60)

# =====================================================================
# LIVE DEMO USING THE BREAST CANCER DATASET FROM YOUR REPORT
# =====================================================================
if __name__ == "__main__":
    # Load raw data matrix
    raw_data = load_breast_cancer()
    
    # Pack into a standard pandas DataFrame structure
    dataset_df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    dataset_df['TARGET_LABEL'] = raw_data.target
    
    # Introduce an intentional missing-value bias on purpose to verify detection mechanics
    dataset_df.iloc[10:14, 0] = np.nan 
    
    # Run audit pipeline
    analyze_dataset_bias(dataset_df, target_column='TARGET_LABEL')