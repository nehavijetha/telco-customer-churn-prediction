-- Monthly revenue at risk by segment

SELECT
    Contract,
    InternetService,
    COUNT(*) AS churned_customers,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_lost,
    ROUND(SUM(TotalCharges), 2) AS total_revenue_lost
FROM telco
WHERE Churn = 'Yes'
GROUP BY Contract, InternetService
ORDER BY monthly_revenue_lost DESC;