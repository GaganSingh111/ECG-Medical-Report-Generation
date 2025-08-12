# ECG Heartbeat Categorization and Medical Report Generation

## Overview
This project focuses on classifying ECG heartbeat signals and generating concise medical-style reports based on the analysis.  
The system is trained on the **MIT-BIH Arrhythmia Dataset** using a hybrid **CNN + LSTM** deep learning architecture.  
In addition to classification, the project integrates **T5 Transformer** and **Google Gemini API** to produce automated text-based medical reports, along with **SHAP** for explainable AI insights.

## Objectives
- Classify ECG heartbeats into five categories.
- Automatically generate structured and readable medical reports.
- Provide interpretability for model predictions through SHAP visualizations.

## Dataset
- **Source:** MIT-BIH Arrhythmia Dataset
- **Classes:**
  - N → Normal beat
  - S → Supraventricular beat
  - V → Ventricular beat
  - F → Fusion beat
  - Q → Unknown beat

## Project Workflow
1. **Data Preprocessing**
   - Standardization using `StandardScaler`
   - Class balancing using `SMOTE`
2. **Model Development**
   - CNN layers for feature extraction
   - LSTM layers for sequential pattern recognition
3. **Report Generation**
   - T5 Transformer Model for structured reports
   - Google Gemini API for alternative summaries
4. **Explainability**
   - SHAP-based feature importance analysis for predictions

## Technologies and Libraries
- Python
- TensorFlow / Keras
- Hugging Face Transformers
- Scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP
- Matplotlib, Seaborn

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecg-report-generation.git
