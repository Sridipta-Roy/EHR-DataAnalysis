
# ICU Mortality and Length of Stay Prediction using EHR Data

This project investigates the use of machine learning techniques to predict **ICU mortality** and **length of ICU stay** using clinical data from the [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) database. It demonstrates how early prediction using 24-hour patient records can enhance clinical decision-making, resource planning, and patient management in intensive care units (ICUs).

## 📌 Project Objectives

- **ICU Mortality Prediction**  
  Predict the likelihood of in-hospital mortality using data collected within the first 24 hours of ICU admission.

- **ICU Length of Stay (LOS) Prediction**  
  Predict short ICU stays (≤ 7 days) among survivors using the same time window of data.

---

## 🎯 OKRs (Objectives and Key Results)

### Mortality Prediction
- ✅ Achieve AUC ≥ 0.85 on the test dataset
- ✅ Identify top predictors for mortality (e.g., GCS scores, comorbidities)

### LOS Prediction
- ✅ Define “short stay” as LOS ≤ 7 days
- ✅ Achieve Mean Absolute Error (MAE) ≤ 1 day
- ✅ Identify key predictors for LOS

---

## 📊 KPIs (Key Performance Indicators)

- **Classification AUC** ≥ 0.85  
- **Regression MAE** ≤ 1.0  
- Feature importance extracted from models  
- Comparison of classification metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📁 Data Source

- **Database:** [MIMIC-IV Clinical Database](https://physionet.org/content/mimiciv/2.2/)
- **Access:** Available through PhysioNet with proper credentialing.
- **Dataset Size:** ~94,000 ICU admissions

---

## 🔍 Data Preparation Steps

1. **Data Loading**  
   Efficient loading of large MIMIC-IV tables using chunking (e.g., `chartevents`, `labevents`).

2. **Dataset Integration**  
   Merge clinical, demographic, and diagnostic tables by patient and admission identifiers.

3. **Preprocessing**
   - Age filtering: 18–90
   - Physiological plausibility checks
   - Imputation using Iterative Imputer
   - One-hot encoding of categorical features
   - Exclusion of readmissions and ICU stays ≤ 1 day

4. **Partitioning**
   - Classification: 90% train / 10% test with class balancing via undersampling
   - Regression: 80% train / 20% test using log-transformed LOS for survivors

---

## 📈 Machine Learning Models

### ICU Mortality Prediction (Classification)
| Model              | AUC    | Recall | F1 Score |
|-------------------|--------|--------|----------|
| Logistic Regression | 0.87 | 0.90   | 0.42     |
| Random Forest       | 0.89 | 0.95   | 0.39     |
| **XGBoost**         | **0.90** | **0.92**   | **0.44**     |
| MLP Classifier      | 0.89 | 0.91   | 0.43     |

🔹 **Best Classifier:** `XGBoost` due to superior AUC and recall.

### ICU Length of Stay Prediction (Regression)
| Model              | MAE   | RMSE  | R²    |
|-------------------|-------|-------|-------|
| **Elastic Net**       | **0.96** | 1.29  | 0.15  |
| KNN Regressor      | 0.96  | 1.31  | 0.13  |
| Random Forest      | 0.95  | 1.23  | 0.17  |
| XGBoost            | 0.94  | 1.27  | 0.18  |

🔹 **Best Regressor:** `Elastic Net` due to better generalization and simplicity.

---

## 🔑 Key Features Identified

- **Mortality:** Respiratory failure, Sepsis, GCS (eye/verbal), Urea Nitrogen, Metastasis
- **Length of Stay:** GCS scores, Urea Nitrogen, Age, Heart Rate, Hematocrit

---

## 📌 Project Structure

```
📁 ICU-Prediction/
│
├── data/                  # Processed dataset files
├── notebooks/             # Exploratory data analysis and modeling notebooks
├── models/                # Trained model pickle files
├── figures/               # ROC, PR curves, feature importance, etc.
├── scripts/               # Modular code for preprocessing and training
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 📅 Project Timeline

| Week | Task |
|------|------|
| 1–2  | Literature review, data access setup |
| 3–4  | Data preprocessing and integration |
| 5–6  | Feature selection, EDA, and classification modeling |
| 7    | Regression modeling for LOS prediction |
| 8    | Evaluation, reporting, and future scope planning |

---

## 🚀 Future Enhancements

- Patient stratification using clustering for risk-based care
- Advanced imputation based on patient subgroups
- Multi-class LOS prediction (short / average / long)
- Incorporating unstructured notes (NLP)

---

## 📚 Citations

1. [MIMIC-IV, PhysioNet](https://physionet.org/content/mimiciv/2.2/)
2. Iwase et al. *Prediction algorithm for ICU mortality and LOS using ML*. [Scientific Reports, 2022](https://doi.org/10.1038/s41598-022-17091-5)
3. Pang et al. *ICU Mortality Risk Prediction with ML*. [Diagnostics, 2022](https://doi.org/10.3390/diagnostics12051068)

---

## 🧠 Authors

- [S. Sudharsan](mailto:sudharsan.s@northeastern.edu)  
- [Sridipta Roy](mailto:roy.sr@northeastern.edu)

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Matplotlib, Seaborn
- Jupyter Notebooks
- MIMIC-IV SQL database
