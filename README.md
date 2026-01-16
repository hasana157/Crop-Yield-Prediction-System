# 🌾 Crop Yield Prediction System

A complete **end-to-end machine learning pipeline** for predicting agricultural crop yields based on historical data. This project includes EDA, preprocessing, feature engineering, ML models, and a Streamlit dashboard.

## 🚀 Project Overview

Food security is a critical global challenge. This project forecasts **crop yield (hg/ha)** using historical agricultural data from the Food and Agriculture Organization (FAO).

**The system:**

* Cleans raw agricultural data
* Handles missing values and duplicates
* Extracts key features and interactions
* Trains Machine Learning (Random Forest) and Deep Learning (ANN) models
* Evaluates performance using RMSE and 
* Visualizes insights through an interactive dashboard
* Provides a **GUI app** for single and batch predictions

## 📂 Repository Structure

```text
├── yield.csv                             # Raw FAO crop production data
├── modelcode.ipynb             # Jupyter notebook: EDA, modeling, and tuning
├── app.py                                # Python script for the Streamlit web app
├── report/
│   └── ProjectReport.pdf # Technical Project Report
└── README.md                             # Project Documentation

```

## 📊 Dataset

* **Source:** Food and Agriculture Organization (FAO)
* **Records:** 56,717 samples
* **Target:** Yield Value (hg/ha)
* **Features:** Year, Area (Country), Item (Crop type)

## 🧹 Data Preprocessing Highlights

✔ **Data Cleaning:** Removed non-essential columns and handled duplicate records.

✔ **Encoding:** Used Label Encoding for categorical features like 'Area' and 'Item'.

✔ **Feature Scaling:** Applied `StandardScaler` to normalize numerical data for the ANN model.

✔ **Outlier Management:** Analyzed distributions to ensure robust model performance.

## 🔍 Exploratory Data Analysis

Includes:

* Yield distribution analysis.
* Yearly and regional production trends.
* Correlation heatmaps.
* Top-performing crops and countries.

💡 **Insight:** Agricultural yields show significant regional variance and a steady increase over the decades due to technological improvements.

`<img width="1538" height="934" alt="image" src="https://github.com/user-attachments/assets/263fbb01-314d-4ef3-913d-338cbc1a35cd" />
`

## ⚙️ Feature Engineering

* **Temporal Features:** Year-based trends.
* **Categorical Interaction:** Relationships between specific crops and regions.
* **Feature Importance:** Identified 'Item' and 'Area' as the strongest predictors of yield.

## 🤖 Models Implemented

### 1️⃣ Linear Regression (Baseline)

* Used to establish a performance floor.
* Struggles with the non-linear complexity of global agriculture.

### 2️⃣ Artificial Neural Network (ANN)

* **Layers:** Input → Dense (64) → Dense (32) → Output.
* **Activation:** ReLU.
* **Optimizer:** Adam.
* Captures deep non-linear patterns in the data.

### 3️⃣ Optimized Random Forest (🏆 Winner)

* **RMSE:** ~8,945.
* **:** 0.945.
* **Strengths:** Highly accurate, handles outliers, and provides clear feature importance.

## 📈 Model Comparison

| Model |  |  Score |
| --- | --- | --- |
| Linear Regression | High | Low |
| ANN (Neural Network) | ~30,000 | ~0.80 |
| **Random Forest** | **8,945** | **0.945** |

`<img width="618" height="238" alt="image" src="https://github.com/user-attachments/assets/ab833030-7c1a-4426-8c78-1f38baef7a0b" />
<img width="1062" height="318" alt="image" src="https://github.com/user-attachments/assets/b1a91fa2-4fdf-4fdf-a868-49c744c4acb5" />

<img width="1531" height="264" alt="image" src="https://github.com/user-attachments/assets/6d4f24a7-a4f0-45c0-bd6b-ee16419715d3" />


`

## 🎯 Predictions

Outputs:

* Predicted Yield in **hg/ha**.
* Conversion to **tons/ha** for practical use.
* Residual analysis to ensure prediction reliability.

## 🖥 Streamlit GUI

**Features:**
✔ **Interactive EDA:** Dynamic charts and dataset statistics.

✔ **Single Prediction:** Input Area, Item, and Year for instant results.

✔ **Batch Prediction:** Upload a CSV file and download the predicted yield results.

✔ **Visual Metrics:** Live performance visualization.

**Run App:**

```bash
streamlit run app.py

```

## 💪 Strengths

* **High Precision:** 0.945  ensures reliable forecasting.
* **Scalability:** Supports batch processing for large-scale data.
* **Interpretability:** Clear insights into which factors drive yield.

## ⚠️ Limitations

* **Dynamic Factors:** Does not currently account for real-time weather anomalies (e.g., droughts).
* **Data Scope:** Forecasts are limited to the regions and crops present in the FAO dataset.

## 📌 Conclusion

This system demonstrates that **Random Forest models**, when combined with proper feature encoding, provide a powerful tool for global crop forecasting. The Streamlit integration makes these complex insights accessible for practical agricultural planning.

## 🙌 Contributors

* **Hasana Zahid** – CIIT/SP24-BAI-060
* **Dur-e-Shahwar** – CIIT/SP24-BAI-013
* **Instructor:** Mr. Umar Nouman & Ms. Hilal Jan
* **Institution:** COMSATS University Islamabad (CUI)

⭐ **Future Improvements:** Integrate real-time weather APIs and implement LSTM for time-series sequence modeling.
