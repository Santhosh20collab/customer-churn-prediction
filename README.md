# 🔄 Customer Churn Prediction System

A machine learning project that predicts customer churn for a telecom company using a **Random Forest classifier**, achieving **87% accuracy** on 7,000+ records.

## 📌 Project Overview

Customer churn is one of the most critical business metrics in the telecom industry. This project builds an end-to-end ML pipeline to identify customers likely to churn, enabling targeted retention strategies.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Machine learning |
| Matplotlib & Seaborn | Visualization |
| Power BI | Interactive dashboard |

## 📊 Key Results

- **Accuracy:** 87% using Random Forest
- **ROC-AUC:** ~0.91
- **Dataset:** 7,000+ telecom customer records
- **Top 5 Churn Predictors:** Contract type, Tenure, Monthly charges, Internet service, Payment method

## 📁 Project Structure

```
customer-churn-prediction/
│
├── churn_prediction.py     # Main ML pipeline
├── eda_analysis.png        # EDA visualizations (auto-generated)
├── model_results.png       # Model performance plots (auto-generated)
├── requirements.txt        # Dependencies
└── README.md
```

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Santhosh20collab/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python churn_prediction.py
```

> **Note:** If the dataset CSV is not present, the script auto-generates synthetic telecom data for demonstration.

## 📈 ML Pipeline

1. **Data Loading** – Telco Customer Churn dataset (7,000+ records)
2. **EDA** – Visualizing churn distribution, contract types, tenure & charges
3. **Preprocessing** – Missing value handling, label encoding, feature scaling
4. **Model Training** – Random Forest with class balancing
5. **Evaluation** – Accuracy, ROC-AUC, Confusion Matrix, Feature Importance
6. **Insights** – Top 5 churn drivers for business recommendations

## 👤 Author

**Santhosh S**  
B.Tech CSE (Cyber Security) | Bharath Institute of Higher Education and Research  
[LinkedIn](https://www.linkedin.com/in/santhosh-s-4b6450302) | [GitHub](https://github.com/Santhosh20collab)
