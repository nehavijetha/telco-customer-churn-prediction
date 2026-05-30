import os
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & clean
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.drop(columns=["customerID"], inplace=True)
df.drop_duplicates(inplace=True)

print(f"Loaded  : {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["HasSupport"] = (
    (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
).astype(int)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# ══════════════════════════════════════════════════════════════════════════════
# 3. Split
# ══════════════════════════════════════════════════════════════════════════════

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train   : {X_train.shape}  |  Test : {X_test.shape}")
print(f"Churn % : {y_train.mean():.1%} train  |  {y_test.mean():.1%} test")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluate helper
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(name: str, y_true, y_pred, y_prob) -> dict:
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    print(f"\n{'─' * 42}")
    print(f"  {name}")
    print(f"{'─' * 42}")
    for k, v in list(metrics.items())[1:]:
        print(f"  {k:<12}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Stay", "Churn"]))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 6. Train models
# ══════════════════════════════════════════════════════════════════════════════

results = []

# ── Logistic Regression ───────────────────────────────────────────────────────

lr_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]

results.append(evaluate("Logistic Regression", y_test, lr_pred, lr_prob))

# ── Random Forest ─────────────────────────────────────────────────────────────

rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]

results.append(evaluate("Random Forest", y_test, rf_pred, rf_prob))

# ── XGBoost ───────────────────────────────────────────────────────────────────

xgb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
        ),
    ]
)

xgb_pipeline.fit(X_train, y_train)
xgb_pred = xgb_pipeline.predict(X_test)
xgb_prob = xgb_pipeline.predict_proba(X_test)[:, 1]

results.append(evaluate("XGBoost", y_test, xgb_pred, xgb_prob))

# ══════════════════════════════════════════════════════════════════════════════
# 7. Summary table
# ══════════════════════════════════════════════════════════════════════════════

summary = pd.DataFrame(results).set_index("Model")
summary_pct = (summary * 100).round(2)

print("\n" + "═" * 60)
print("  MODEL COMPARISON SUMMARY (%)")
print("═" * 60)
print(summary_pct.to_string())

best_model = summary["ROC-AUC"].idxmax()
print(
    f"\n🏆 Best model (ROC-AUC): {best_model}  →  {summary.loc[best_model, 'ROC-AUC']:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Feature importance  (from best model — Logistic Regression coefficients)
# ══════════════════════════════════════════════════════════════════════════════

feature_names = lr_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

coefficients = lr_pipeline.named_steps["model"].coef_[0]

importance_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": abs(coefficients)})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

importance_df.to_csv("models/feature_importance.csv", index=False)
print(f"\nTop 5 features:\n{importance_df.head().to_string(index=False)}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Save best model
# ══════════════════════════════════════════════════════════════════════════════

joblib.dump(lr_pipeline, "models/logistic_regression.pkl")
print("\n✅ Model saved → models/logistic_regression.pkl")
