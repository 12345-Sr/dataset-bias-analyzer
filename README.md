# 🔬 Advanced Bias-Variance Diagnostic Suite

An interactive, multi-page enterprise-grade diagnostic platform built with **Python** and **Streamlit Cloud**. This platform implements the core theoretical frameworks and empirical methodologies detailed in the research report: *"Bias-Variance Decomposition for Model Evaluation"*. 

It provides machine learning practitioners with an automated, live engine to detect structural data anomalies and visualize algorithmic generalisation gaps in real time.

---

## 📁 System Architecture & Directory Structure

The project utilizes a modular, multi-page Streamlit configuration to keep the computational workflows separated from the frontend presentation layers:

```text
📁 dataset-bias-analyzer/
├── 📄 app.py                     # Main Landing Page & Central Data Hub
├── 📄 requirements.txt           # Package Dependencies & Constraints
├── 📄 synthetic_cancer_data.csv  # Benchmark Classification Dataset
├── 📄 synthetic_housing_data.csv # Benchmark Regression Dataset
└── 📁 pages/                     # Specialized Diagnostic Dashboards
    ├── 📄 1_📊_Data_Diagnostics.py
    ├── 📄 2_🤖_Model_Selection.py
    └── 📄 3_📈_Learning_Curves.py
