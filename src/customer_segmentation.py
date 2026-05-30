import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# CLTV
df["CLTV"] = df["MonthlyCharges"] * df["tenure"]

# Segments
q1 = df["CLTV"].quantile(0.25)
q3 = df["CLTV"].quantile(0.75)


def segment(cltv):
    if cltv >= q3:
        return "High Value"
    elif cltv >= q1:
        return "Medium Value"
    else:
        return "Low Value"


df["CustomerSegment"] = df["CLTV"].apply(segment)

print("=" * 50)
print("CUSTOMER SEGMENTATION")
print("=" * 50)

print(df["CustomerSegment"].value_counts())

print("\nChurn Rate by Segment")

segment_churn = df.groupby("CustomerSegment")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print(segment_churn)

# Save segment counts for Streamlit dashboard

segment_counts = df["CustomerSegment"].value_counts().reset_index()
segment_counts.columns = ["Segment", "Customers"]

segment_counts.to_csv("models/customer_segments.csv", index=False)

# Save churn rate by segment

segment_churn_df = segment_churn.reset_index()
segment_churn_df.columns = ["Segment", "ChurnRate"]

segment_churn_df.to_csv("models/customer_segment_churn.csv", index=False)

print("\nFiles saved successfully.")
