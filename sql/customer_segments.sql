-- High-value churned customers

SELECT
    Contract,
    tenure,
    MonthlyCharges,
    TotalCharges,
    PaymentMethod,
    InternetService
FROM telco
WHERE Churn = 'Yes'
  AND TotalCharges > (
        SELECT PERCENTILE_CONT(0.75)
        WITHIN GROUP (ORDER BY TotalCharges)
        FROM telco
    )
ORDER BY TotalCharges DESC;