import pandas as pd

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Churned customers
churned = df[df["Churn"] == "Yes"]

# Revenue at risk
total_risk = churned["TotalCharges"].sum()
monthly_risk = churned["MonthlyCharges"].sum()

print("=" * 50)
print("REVENUE AT RISK ANALYSIS")
print("=" * 50)

print(f"\nTotal Revenue At Risk: ${total_risk:,.2f}")
print(f"Monthly Revenue At Risk: ${monthly_risk:,.2f}")

# Contract analysis
print("\nRevenue At Risk By Contract")
print(churned.groupby("Contract")["MonthlyCharges"].sum().sort_values(ascending=False))

# Internet analysis
print("\nRevenue At Risk By Internet Service")
print(
    churned.groupby("InternetService")["MonthlyCharges"]
    .sum()
    .sort_values(ascending=False)
)

# Payment analysis
print("\nRevenue At Risk By Payment Method")
print(
    churned.groupby("PaymentMethod")["MonthlyCharges"]
    .sum()
    .sort_values(ascending=False)
)
