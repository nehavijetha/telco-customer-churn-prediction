import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f"Original shape : {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Clean
# ══════════════════════════════════════════════════════════════════════════════

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.drop(columns=["customerID"], inplace=True)
df.drop_duplicates(inplace=True)

print(f"After cleaning  : {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

df["HasSupport"] = (
    (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
).astype(int)

print(f"\nEngineered features sample:")
print(df[["tenure", "TotalCharges", "AvgMonthlySpend", "HasSupport"]].head(3))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Encode target
# ══════════════════════════════════════════════════════════════════════════════

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print(f"\nTarget distribution:")
print(df["Churn"].value_counts())
print(f"Churn rate: {df['Churn'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Split features & target
# ══════════════════════════════════════════════════════════════════════════════

X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"\nX : {X.shape}   y : {y.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Identify column types
# ══════════════════════════════════════════════════════════════════════════════

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumerical   ({len(numerical_cols)})  : {numerical_cols}")
print(f"Categorical ({len(categorical_cols)}) : {categorical_cols}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Train / test split
# ══════════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # keeps churn ratio equal in both splits
)

print(f"\nTrain : {X_train.shape}  |  Test : {X_test.shape}")
print(
    f"Train churn rate : {y_train.mean():.1%}  |  Test churn rate : {y_test.mean():.1%}"
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Preprocessor  (scale numerics · one-hot encode categoricals)
# ══════════════════════════════════════════════════════════════════════════════

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
    ],
    remainder="drop",
)

# Fit on train only — never on test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Recover feature names for downstream use (feature importance, SHAP, etc.)
ohe_feature_names = (
    preprocessor.named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
)
all_feature_names = numerical_cols + ohe_feature_names

print(f"\nProcessed X_train : {X_train_processed.shape}")
print(f"Processed X_test  : {X_test_processed.shape}")
print(f"Total features    : {len(all_feature_names)}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Export for model training
# ══════════════════════════════════════════════════════════════════════════════

import joblib, os

os.makedirs("models", exist_ok=True)

joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(X_train_processed, "models/X_train.pkl")
joblib.dump(X_test_processed, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")
joblib.dump(all_feature_names, "models/feature_names.pkl")

print("\n✅ Preprocessor and splits saved to models/")
