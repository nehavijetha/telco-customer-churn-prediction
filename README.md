# Telco Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn. The project covers the complete ML lifecycle, including data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment using Streamlit.

---

## Project Highlights

- Built an end-to-end Customer Churn Prediction system using Machine Learning.
- Performed data cleaning, exploratory data analysis (EDA), and feature engineering.
- Compared Logistic Regression, Random Forest, and XGBoost models.
- Achieved **84.14% ROC-AUC** using Logistic Regression.
- Developed an interactive Streamlit dashboard with real-time churn prediction, risk analysis, feature importance, and business recommendations.

---

## Project Structure

```text
telco-churn/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── logistic_regression.pkl
│   └── feature_importance.csv
│
├── outputs/
│   └── figures/
│       ├── churn_distribution.png
│       ├── contract_vs_churn.png
│       ├── tenure_vs_churn.png
│       └── monthlycharges_vs_churn.png
│
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   └── train.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  Dataset

**Source:** IBM Telco Customer Churn Dataset

### Dataset Information

- Total Customers: 7,043
- Features: 21
- Target Variable: Churn (Yes/No)
- Domain: Telecommunications

### Feature Categories

- Customer Demographics
- Account Information
- Internet Services
- Support Services
- Billing Information
- Contract Details

---

##  Exploratory Data Analysis

Performed:

- Dataset shape and structure analysis
- Missing value detection and treatment
- Duplicate record removal
- Target distribution analysis
- Customer behavior analysis

### Visualizations

- Customer Churn Distribution
- Contract Type vs Churn
- Tenure vs Churn
- Monthly Charges vs Churn

### Key Findings

- Month-to-month customers show significantly higher churn.
- Customers with low tenure are more likely to churn.
- Higher monthly charges increase churn probability.
- Lack of technical support and online security increases churn risk.

---

## Data Cleaning

### Cleaning Steps

- Converted `TotalCharges` to numeric format.
- Handled missing values using median imputation.
- Removed unnecessary customer identifiers.
- Removed duplicate records.

### Dataset Shape

| Stage            | Rows |
| ---------------- | ---- |
| Original Dataset | 7043 |
| After Cleaning   | 7021 |

---

## Feature Engineering

### Features Created

#### AvgMonthlySpend

```python
TotalCharges / (tenure + 1)
```

Represents average customer spending behavior.

#### HasSupport

```python
OnlineSecurity OR TechSupport
```

Captures whether the customer has support-related services.

---

## Models Trained

### Logistic Regression

- Accuracy: 80.21%
- Precision: 66.43%
- Recall: 51.08%
- F1 Score: 57.75%
- ROC-AUC: 84.14%

### Random Forest

- Accuracy: 77.86%
- Precision: 61.01%
- Recall: 45.43%
- F1 Score: 52.08%
- ROC-AUC: 81.54%

### XGBoost

- Accuracy: 80.00%
- Precision: 65.42%
- Recall: 51.88%
- F1 Score: 57.87%
- ROC-AUC: 83.96%

---

## Best Model

### Logistic Regression

Selected based on:

- Highest ROC-AUC Score
- Strong overall performance
- Faster inference
- Better interpretability

---

## Feature Importance

Top churn-driving features identified:

1. Contract Type
2. Tenure
3. Monthly Charges
4. Tech Support
5. Online Security
6. Payment Method

These features play the most significant role in customer churn prediction.

---

## Streamlit Dashboard

The project includes an interactive dashboard with the following pages:

### Home

- Project overview
- Dataset information
- Model summary

### Prediction

- Real-time churn prediction
- Customer health score
- Risk analysis dashboard
- Churn probability gauge
- Risk distribution donut chart
- Business recommendations

### EDA Dashboard

- Churn visualizations
- Customer behavior insights

### 🏆 Model Performance

- Comparison of all trained models
- Performance metrics

### Feature Importance

- Top churn drivers
- Business insights

---

## Business Impact

The solution helps telecom companies:

- Identify customers likely to churn.
- Design targeted retention campaigns.
- Improve customer satisfaction.
- Reduce customer acquisition costs.
- Increase long-term customer retention.

---

## Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-Learn
- XGBoost

### Deployment

- Streamlit

### Model Serialization

- Joblib

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run EDA

```bash
python src/eda.py
```

### Train Models

```bash
python src/train.py
```

### Launch Dashboard

```bash
streamlit run app.py
```

---

## Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit
plotly
joblib
```

---

## Author

**Neha Vijetha Dasari**

Graduate, IIT Hyderabad

### Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Model Evaluation
- Data Visualization
- Streamlit Dashboard Development
- Business Analytics
