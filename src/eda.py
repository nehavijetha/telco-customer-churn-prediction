import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────

os.makedirs("outputs/figures", exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update(
    {"figure.dpi": 120, "axes.titlesize": 13, "axes.titleweight": "bold"}
)

CHURN_PALETTE = {"No": "#22c55e", "Yes": "#ef4444"}


# ── Load & clean ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# TotalCharges has whitespace strings instead of NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.drop(columns=["customerID"], inplace=True)
df.drop_duplicates(inplace=True)


# ── Quick report ──────────────────────────────────────────────────────────────

print("=" * 50)
print(f"Shape          : {df.shape}")
print(f"Duplicates     : {df.duplicated().sum()}")
print(f"Missing values : {df.isnull().sum().sum()}")
print("=" * 50)
print("\nChurn counts:")
print(df["Churn"].value_counts())
print("\nChurn %:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# ── Helper ────────────────────────────────────────────────────────────────────


def save(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", bbox_inches="tight")
    plt.close()
    print(f"Saved → outputs/figures/{filename}")


# ── 1. Churn distribution ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Churn", data=df, palette=CHURN_PALETTE, ax=ax)
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Count")

for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 40,
        f"{int(bar.get_height()):,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

save("churn_distribution.png")


# ── 2. Contract type vs churn ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn", palette=CHURN_PALETTE, ax=ax)
ax.set_title("Contract Type vs Churn")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Churn")
save("contract_vs_churn.png")


# ── 3. Tenure vs churn ────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure", palette=CHURN_PALETTE, ax=ax)
ax.set_title("Tenure vs Churn")
ax.set_xlabel("Churn")
ax.set_ylabel("Tenure (months)")
save("tenure_vs_churn.png")


# ── 4. Monthly charges vs churn ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=CHURN_PALETTE, ax=ax)
ax.set_title("Monthly Charges vs Churn")
ax.set_xlabel("Churn")
ax.set_ylabel("Monthly Charges ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
save("monthlycharges_vs_churn.png")


# ── 5. Internet service vs churn ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="InternetService", hue="Churn", palette=CHURN_PALETTE, ax=ax)
ax.set_title("Internet Service vs Churn")
ax.set_xlabel("Internet Service")
ax.set_ylabel("Count")
ax.legend(title="Churn")
save("internet_service_vs_churn.png")


# ── 6. Payment method vs churn ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn", palette=CHURN_PALETTE, ax=ax)
ax.set_title("Payment Method vs Churn")
ax.set_xlabel("Payment Method")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25)
ax.legend(title="Churn")
save("payment_method_vs_churn.png")


# ── 7. Senior citizen vs churn ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn",
    palette=CHURN_PALETTE,
    ax=ax,
)
ax.set_title("Senior Citizen vs Churn")
ax.set_xlabel("Senior Citizen (0 = No, 1 = Yes)")
ax.set_ylabel("Count")
ax.legend(title="Churn")
save("senior_citizen_vs_churn.png")


print("\n✅ All figures saved to outputs/figures/")
