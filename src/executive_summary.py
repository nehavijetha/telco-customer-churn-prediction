import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

total_customers = len(df)

churn_rate = (df["Churn"] == "Yes").mean() * 100

monthly_revenue_at_risk = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()

historical_spend = df[df["Churn"] == "Yes"]["TotalCharges"].sum()

print("=" * 50)
print("EXECUTIVE SUMMARY")
print("=" * 50)

print(f"Total Customers: {total_customers:,}")

print(f"Overall Churn Rate: {churn_rate:.2f}%")

print(f"Monthly Revenue At Risk: " f"${monthly_revenue_at_risk:,.2f}")

print(f"Historical Spend of Churned Customers: " f"${historical_spend:,.2f}")

print("\nTop Churn Drivers")

print("""
1. Month-to-Month Contract
2. Low Tenure
3. High Monthly Charges
4. No Tech Support
5. No Online Security
""")
