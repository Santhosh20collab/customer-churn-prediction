"""
Customer Churn Prediction System
Author: Santhosh S
Description: Predicts customer churn using Random Forest classifier on telecom dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
def load_data(filepath="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Load the Telco Customer Churn dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Dataset not found. Generating synthetic data for demo...")
        df = generate_synthetic_data()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def generate_synthetic_data(n=7000, seed=42):
    """Generate synthetic telecom churn data for demonstration."""
    np.random.seed(seed)
    tenure       = np.random.randint(0, 72, n)
    monthly      = np.round(np.random.uniform(18, 120, n), 2)
    total        = np.round(tenure * monthly + np.random.normal(0, 50, n), 2)
    contract     = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet     = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment      = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    senior       = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner      = np.random.choice(["Yes", "No"], n)
    dependents   = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    paperless    = np.random.choice(["Yes", "No"], n)
    tech_support = np.random.choice(["Yes", "No", "No internet service"], n)
    online_sec   = np.random.choice(["Yes", "No", "No internet service"], n)

    # Churn probability influenced by contract type, tenure, monthly charges
    churn_prob = (
        0.05
        + (contract == "Month-to-month") * 0.30
        + (monthly > 70) * 0.15
        + (tenure < 12) * 0.20
        + (internet == "Fiber optic") * 0.10
        + senior * 0.08
        - (tenure > 48) * 0.15
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.where(np.random.random(n) < churn_prob, "Yes", "No")

    return pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "TechSupport": tech_support,
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })


# ── 2. Data Preprocessing ─────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # Fix TotalCharges if it's string
    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df


# ── 3. EDA Visualizations ─────────────────────────────────────────────────────
def plot_eda(df_raw):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Customer Churn – Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # Churn distribution
    churn_counts = df_raw["Churn"].value_counts()
    axes[0, 0].pie(churn_counts, labels=["No Churn", "Churn"],
                   autopct="%1.1f%%", colors=["#2196F3", "#F44336"],
                   startangle=140, explode=(0, 0.05))
    axes[0, 0].set_title("Churn Distribution")

    # Churn by Contract Type
    ct = df_raw.groupby("Contract")["Churn"].value_counts(normalize=True).unstack()
    ct["Yes"].plot(kind="bar", ax=axes[0, 1], color="#F44336", alpha=0.8)
    axes[0, 1].set_title("Churn Rate by Contract Type")
    axes[0, 1].set_ylabel("Churn Rate")
    axes[0, 1].tick_params(axis="x", rotation=15)

    # Monthly Charges vs Churn
    sns.boxplot(data=df_raw, x="Churn", y="MonthlyCharges",
                palette={"No": "#2196F3", "Yes": "#F44336"}, ax=axes[1, 0])
    axes[1, 0].set_title("Monthly Charges vs Churn")

    # Tenure vs Churn
    sns.histplot(data=df_raw, x="tenure", hue="Churn",
                 palette={"No": "#2196F3", "Yes": "#F44336"},
                 bins=30, ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_title("Tenure Distribution by Churn")

    plt.tight_layout()
    plt.savefig("eda_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: eda_analysis.png")


# ── 4. Model Training ─────────────────────────────────────────────────────────
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   random_state=42, class_weight="balanced")
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  Model Performance")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return model, X_train, X_test, y_test, y_pred, y_prob, scaler


# ── 5. Result Visualizations ──────────────────────────────────────────────────
def plot_results(model, X_train, X_test, y_test, y_pred, y_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Results – Customer Churn Prediction", fontsize=15, fontweight="bold")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    # Feature Importance (Top 10)
    feat_imp = pd.Series(model.feature_importances_,
                         index=X_train.columns).sort_values(ascending=True).tail(10)
    feat_imp.plot(kind="barh", ax=axes[2], color="#2196F3", alpha=0.85)
    axes[2].set_title("Top 10 Feature Importances")
    axes[2].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: model_results.png")


# ── 6. Top Churn Predictors ───────────────────────────────────────────────────
def top_predictors(model, feature_names):
    feat_imp = pd.Series(model.feature_importances_, index=feature_names)
    top5 = feat_imp.sort_values(ascending=False).head(5)
    print("\nTop 5 Churn Predictors:")
    print("─" * 35)
    for i, (feat, score) in enumerate(top5.items(), 1):
        print(f"  {i}. {feat:<25} {score:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Customer Churn Prediction System")
    print("=" * 50)

    df_raw = load_data()
    plot_eda(df_raw)

    df = preprocess(df_raw)
    model, X_train, X_test, y_test, y_pred, y_prob, scaler = train_model(df)
    plot_results(model, X_train, X_test, y_test, y_pred, y_prob)
    top_predictors(model, X_train.columns)

    print("\n✓ All outputs saved. Run complete!")
