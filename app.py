import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

# Page settings
st.set_page_config(page_title="Fast Dataset Bias Analyzer", page_icon="⚡", layout="wide")

st.title("⚡ Fast Automated Dataset Bias & Performance Analyzer")
st.markdown("""
**Speed Optimized Version:** This configuration utilizes multi-core parallel processing (`n_jobs=-1`) 
and optimized tree limits to calculate algorithmic bias vs. variance profiles in real time without lagging.
""")

# --- 1. DATA INPUT HANDLING (LOCAL AND REPOSITORY UPLOAD) ---
st.sidebar.header("📁 Data Source Selection")
data_source = st.sidebar.radio("Choose Data Input Method:", ["Upload Custom CSV File", "Select from Local Folder"])

df = None

if data_source == "Upload Custom CSV File":
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully!")
else:
    # Scan the project directory for any CSV files
    local_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if len(local_files) > 0:
        selected_local_file = st.sidebar.selectbox("Choose a local CSV dataset file:", local_files)
        if selected_local_file:
            df = pd.read_csv(selected_local_file)
            st.success(f"Local file '{selected_local_file}' loaded successfully!")
    else:
        st.sidebar.warning("⚠️ No `.csv` files found in your local project folder. Drop your dataset files right next to `app.py`.")
        st.info("ℹ️ Please switch to file upload or place a CSV file in the directory.")

# Proceed only if a dataset is successfully loaded from either stream
if df is not None:
    with st.expander("👀 View Raw Dataset Preview", expanded=False):
        st.dataframe(df.head(10))
        
    st.markdown("---")
    
    # --- 2. TARGET VARIABLE CONFIGURATION SIDEBAR ---
    st.sidebar.header("🎯 Target Configuration")
    columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Column (Y)", columns)
    task_type = st.sidebar.radio("Select Task Type", ["Regression", "Classification"])
    
    run_analysis = st.sidebar.button("🚀 Run Fast Analysis", use_container_width=True)

    # --- 3. PROCESSING & RESULTS GENERATION CONTAINER ---
    if run_analysis:
        clean_df = df.dropna(subset=[target_col]).copy()
        y = clean_df[target_col]
        
        if task_type == "Classification":
            if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                y = pd.Series(y, name=target_col)
        else:
            try:
                y = pd.to_numeric(y)
            except ValueError:
                st.error(f"❌ **Data Type Conflict**: Column **'{target_col}'** contains non-numeric strings. Please switch to **Classification**.")
                st.stop()

        # Isolate and encode feature matrices cleanly (X)
        X = clean_df.drop(columns=[target_col])
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.median(numeric_only=True))
        X = X.select_dtypes(include=[np.number]) 

        if X.shape[1] == 0:
            st.error("❌ **Feature Error**: No numeric feature columns remaining after encoding.")
            st.stop()

        # --- AUDIT BLOCK ---
        st.header("📊 1. Data Structural Bias Audit")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Data Sparsity Check")
            missing_matrix = df.isnull().sum()
            total_missing = missing_matrix.sum()
            if total_missing > 0:
                st.warning(f"⚠️ Found {total_missing} missing entries.")
                st.dataframe(missing_matrix[missing_matrix > 0])
            else:
                st.success("✅ Clean Record: No missing data detected.")
                
        with col2:
            st.subheader("Target Distribution Balance")
            if task_type == "Classification":
                class_counts = pd.Series(y).value_counts(normalize=True) * 100
                st.bar_chart(class_counts)
                if class_counts.iloc[0] > 70:
                    st.error("⚠️ Severe Class Imbalance! Models will pick up a heavy majority-class bias.")
                else:
                    st.success("✅ Labels are distributed within acceptable variances.")
            else:
                skew_val = y.skew()
                st.metric("Target Skewness Coefficient", f"{skew_val:.2f}")
                if abs(skew_val) > 1.5:
                    st.warning("⚠️ High target skewness detected!")
                else:
                    st.success("✅ Target values display a normally balanced profile.")

        st.markdown("---")

        # --- MODEL PROCESSING BLOCK ---
        st.header("🤖 2. Fast Generalisation Bias Evaluation")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task_type == "Regression":
            models = {
                "Linear Regression (Baseline)": LinearRegression(n_jobs=-1),
                "Random Forest Regressor (Fast)": RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42),
                "Gradient Boosting Regressor (Fast)": GradientBoostingRegressor(n_estimators=30, max_depth=4, random_state=42)
            }
            metric_name = "MSE"
        else:
            models = {
                "Logistic Regression (Baseline)": LogisticRegression(max_iter=500, n_jobs=-1),
                "Random Forest Classifier (Fast)": RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42),
                "Gradient Boosting Classifier (Fast)": GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42)
            }
            metric_name = "Accuracy"

        results = []

        with st.spinner("Executing accelerated training matrix across engine cores..."):
            for name, model in models.items():
                model.fit(X_train, y_train)
                
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                if task_type == "Regression":
                    train_score = mean_squared_error(y_train, train_preds)
                    test_score = mean_squared_error(y_test, test_preds)
                    status = "High Bias (Underfitting)" if train_score > 0.5 else "Balanced"
                    if (test_score - train_score) > (0.2 * train_score):
                        status = "High Variance (Overfitting)"
                else:
                    train_score = accuracy_score(y_train, train_preds)
                    test_score = accuracy_score(y_test, test_preds)
                    status = "High Bias (Underfitting)" if train_score < 0.75 else "Balanced"
                    if (train_score - test_score) > 0.10:
                        status = "High Variance (Overfitting)"
                
                results.append({
                    "Model Configuration": name,
                    f"Train {metric_name}": round(train_score, 4),
                    f"Test {metric_name}": round(test_score, 4),
                    "Generalization Profile Status": status
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        # --- CHART PLOTTING BLOCK ---
        st.markdown("### 📈 Generalization Gap Comparison")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        
        x_labels = results_df["Model Configuration"].tolist()
        train_scores = results_df[f"Train {metric_name}"].tolist()
        test_scores = results_df[f"Test {metric_name}"].tolist()
        
        x_axis = np.arange(len(x_labels))
        ax.bar(x_axis - 0.2, train_scores, 0.4, label=f'Train {metric_name}', color='skyblue')
        ax.bar(x_axis + 0.2, test_scores, 0.4, label=f'Test {metric_name}', color='salmon')
        
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels, rotation=10)
        ax.set_ylabel(metric_name)
        ax.set_title("Training Error vs. Validation Generalisation Gap")
        ax.legend()
        ax.grid(alpha=0.2)
        
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select your target options in the left sidebar configuration, then click 'Run Fast Analysis' to generate the reports.")