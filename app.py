import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
[data-testid="stSidebar"] { min-width: 200px; }
div[data-testid="metric-container"] {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Model ─────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    return joblib.load("models/logistic_regression.pkl")


model = load_model()

# ── Navigation ────────────────────────────────────────────────────────────────

page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Prediction", "Model Performance", "EDA Dashboard", "Feature Importance"],
)


# ════════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════════

if page == "Home":

    st.title("📊 Telco Customer Churn Prediction")
    st.caption("Predict customer retention risk using machine learning")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", "7,043")
    c2.metric("Model Accuracy", "80.2%")
    c3.metric("ROC-AUC", "84.1%")

    st.markdown("---")
    st.markdown("""
    ### About this project

    This dashboard predicts whether a telecom customer is likely to churn,
    using demographic, service, contract, and billing data.

    | | |
    |---|---|
    | **Dataset** | 7,043 customers · 21 features |
    | **Best model** | Logistic Regression |
    | **Models trained** | Logistic Regression · Random Forest · XGBoost |
    """)


# ════════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Prediction":

    st.title("🤖 Customer Churn Prediction")

    form_col, result_col = st.columns([1, 1.35], gap="large")

    # ── LEFT: Input form ──────────────────────────────────────────────────────

    with form_col:

        st.markdown("##### Customer profile")

        r1c1, r1c2 = st.columns(2)
        gender = r1c1.selectbox("Gender", ["Male", "Female"])
        senior = r1c2.selectbox(
            "Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

        r2c1, r2c2 = st.columns(2)
        partner = r2c1.selectbox("Partner", ["Yes", "No"])
        dependents = r2c2.selectbox("Dependents", ["Yes", "No"])

        tenure = st.slider("Tenure (months)", 0, 72, 12)

        st.markdown("##### Contract & billing")

        contract = st.selectbox(
            "Contract type",
            ["Month-to-month", "One year", "Two year"],
        )

        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        b1, b2 = st.columns(2)
        paperless_billing = b1.selectbox("Paperless billing", ["Yes", "No"])
        monthly_charges = b2.number_input(
            "Monthly charges ($)", min_value=0.0, value=70.0
        )

        with st.expander("🌐 Internet & support services"):

            p1, p2 = st.columns(2)
            phone_service = p1.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = p2.selectbox(
                "Multiple lines", ["Yes", "No", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )

            s1, s2 = st.columns(2)
            online_security = s1.selectbox(
                "Online security", ["Yes", "No", "No internet service"]
            )
            tech_support = s2.selectbox(
                "Tech support", ["Yes", "No", "No internet service"]
            )

            s3, s4 = st.columns(2)
            online_backup = s3.selectbox(
                "Online backup", ["Yes", "No", "No internet service"]
            )
            device_protection = s4.selectbox(
                "Device protection", ["Yes", "No", "No internet service"]
            )

            s5, s6 = st.columns(2)
            streaming_tv = s5.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = s6.selectbox(
                "Streaming movies", ["Yes", "No", "No internet service"]
            )

        predict_btn = st.button(
            "🔮 Predict churn", use_container_width=True, type="primary"
        )

    # ── RIGHT: Results panel ──────────────────────────────────────────────────

    with result_col:

        if not predict_btn:
            st.markdown("<br>" * 4, unsafe_allow_html=True)
            st.info("👈  Fill in the form and click **Predict churn** to see results.")

        else:

            # ── Feature engineering ───────────────────────────────────────────

            total_charges = tenure * monthly_charges
            avg_monthly_spend = total_charges / (tenure + 1)
            has_support = int(online_security == "Yes" or tech_support == "Yes")

            input_df = pd.DataFrame(
                {
                    "gender": [gender],
                    "SeniorCitizen": [senior],
                    "Partner": [partner],
                    "Dependents": [dependents],
                    "tenure": [tenure],
                    "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines],
                    "InternetService": [internet_service],
                    "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup],
                    "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support],
                    "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies],
                    "Contract": [contract],
                    "PaperlessBilling": [paperless_billing],
                    "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges],
                    "AvgMonthlySpend": [avg_monthly_spend],
                    "HasSupport": [has_support],
                }
            )

            probability = model.predict_proba(input_df)[0][1]
            health_score = int((1 - probability) * 100)

            # ── Customer snapshot (FIX 1: short labels, no truncation) ────────

            contract_labels = {
                "Month-to-month": "Monthly",
                "One year": "1 Year",
                "Two year": "2 Year",
            }

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Contract", contract_labels[contract])
            m2.metric("Tenure", f"{tenure} mo")
            m3.metric("Monthly", f"${monthly_charges:.0f}")
            m4.metric("Payment", payment_method.split()[0])

            st.markdown("---")

            # ── KPI row ───────────────────────────────────────────────────────

            k1, k2 = st.columns(2)
            k1.metric("⚠️ Churn risk", f"{probability:.1%}")
            k2.metric("💚 Retention chance", f"{1 - probability:.1%}")

            # ── Health score bar (FIX 2: color-coded bar) ─────────────────────

            bar_color = (
                "#22c55e"
                if health_score >= 80
                else "#f59e0b" if health_score >= 60 else "#ef4444"
            )

            st.markdown(f"**Health score — {health_score}/100**")
            st.markdown(
                f"""
<div style="background:#2a2a3e; border-radius:6px; height:10px; overflow:hidden;">
  <div style="width:{health_score}%; height:100%; background:{bar_color}; border-radius:6px;"></div>
</div>
<div style="margin-bottom:12px"></div>
""",
                unsafe_allow_html=True,
            )

            # ── Gauge chart (FIX 3: transparent bg, lighter step colors) ──────

            gauge_color = (
                "#ef4444"
                if probability >= 0.60
                else "#f59e0b" if probability >= 0.30 else "#22c55e"
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(probability * 100, 1),
                    number={"suffix": "%", "font": {"size": 28}},
                    title={"text": "Churn risk", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": gauge_color},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(34,197,94,0.25)"},
                            {"range": [30, 60], "color": "rgba(245,158,11,0.25)"},
                            {"range": [60, 100], "color": "rgba(239,68,68,0.25)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.75,
                            "value": round(probability * 100, 1),
                        },
                    },
                )
            )

            fig_gauge.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Risk badge ────────────────────────────────────────────────────

            if probability >= 0.60:
                st.error("🔴 High risk customer")
            elif probability >= 0.30:
                st.warning("🟡 Medium risk customer")
            else:
                st.success("🟢 Low risk customer")

            # ── Key risk drivers ──────────────────────────────────────────────

            st.markdown("#### 🔍 Key risk drivers")

            risk_factors = []
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract")
            if tenure < 12:
                risk_factors.append(f"Low tenure ({tenure} months)")
            if tech_support == "No":
                risk_factors.append("No tech support")
            if online_security == "No":
                risk_factors.append("No online security")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment")
            if monthly_charges > 80:
                risk_factors.append(f"High monthly charges (${monthly_charges:.0f})")

            if not risk_factors:
                st.success("✅ No major risk factors detected")
            else:
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")

            # ── Retention strategy ────────────────────────────────────────────

            st.markdown("#### 🎯 Retention strategy")

            if probability >= 0.60:
                st.markdown("""
- 🏷️ Offer annual contract discount
- 📞 Proactive outreach within 48 hrs
- 🛡️ Provide premium support package
- 🎁 Run targeted retention campaign
                """)
            elif probability >= 0.30:
                st.markdown("""
- 🎁 Send loyalty rewards offer
- 📈 Encourage long-term contract upgrade
- 🔧 Promote bundled service plans
- 💬 Personalized check-in message
                """)
            else:
                st.markdown("""
- 💬 Maintain regular engagement
- ⭐ Promote premium add-ons
- 📊 Monitor usage for upsell opportunities
                """)


# ════════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Model Performance":

    st.title("🏆 Model Performance")

    metrics = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Accuracy (%)": [80.21, 77.86, 80.00],
            "Precision (%)": [66.43, 61.01, 65.42],
            "Recall (%)": [51.08, 45.43, 51.88],
            "F1 Score (%)": [57.75, 52.08, 57.87],
            "ROC-AUC (%)": [84.14, 81.54, 83.96],
        }
    )

    st.dataframe(
        metrics.style.highlight_max(
            subset=[
                "Accuracy (%)",
                "Precision (%)",
                "Recall (%)",
                "F1 Score (%)",
                "ROC-AUC (%)",
            ],
            color="#1a472a",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.success("🏆 Best model: Logistic Regression")


# ════════════════════════════════════════════════════════════════════════════════
# EDA DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

elif page == "EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Churn distribution")
        st.image("outputs/figures/churn_distribution.png", use_container_width=True)

        st.subheader("Tenure vs churn")
        st.image("outputs/figures/tenure_vs_churn.png", use_container_width=True)

    with c2:
        st.subheader("Contract type vs churn")
        st.image("outputs/figures/contract_vs_churn.png", use_container_width=True)

        st.subheader("Monthly charges vs churn")
        st.image(
            "outputs/figures/monthlycharges_vs_churn.png", use_container_width=True
        )


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Feature Importance":

    import plotly.express as px

    st.title("🔍 Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Contract",
                "Tenure",
                "Monthly Charges",
                "Tech Support",
                "Online Security",
                "Payment Method",
                "Internet Service",
                "Paperless Billing",
            ],
            "Importance": [0.30, 0.22, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04],
        }
    ).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top drivers of customer churn",
        color="Importance",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
    )

    fig.update_layout(
        height=400,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
| Driver | Insight |
|---|---|
| **Contract** | Month-to-month customers churn significantly more |
| **Tenure** | First 12 months are the highest risk window |
| **Monthly charges** | Higher bills correlate with increased churn |
| **Tech support** | Absence of support doubles churn likelihood |
| **Online security** | Unprotected customers feel less locked in |
""")
