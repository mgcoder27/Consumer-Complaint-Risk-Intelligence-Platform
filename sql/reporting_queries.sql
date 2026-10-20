-- 1. Complaint volume by product and calendar month.
-- This shows how demand for support changes over time for each product.
SELECT
    pi.product_id AS product,
    d.year,
    d.month,
    d.month_name,
    COUNT(*) AS complaint_count
FROM fact_complaints AS f
JOIN dim_product_issue AS pi
    ON pi.product_id = f.product_id
   AND pi.issue_id = f.issue_id
JOIN dim_date AS d
    ON d.date_id = f.date_received_id
GROUP BY pi.product_id, d.year, d.month, d.month_name
ORDER BY d.year, d.month, complaint_count DESC, product;

-- 2. Monetary-relief rate by company.
-- The denominator is all complaints for the company. A response is counted as
-- relief when its CFPB response text contains "monetary relief".
SELECT
    c.company_name,
    COUNT(*) AS complaint_count,
    SUM(
        CASE
            WHEN LOWER(COALESCE(f.company_response, '')) LIKE '%monetary relief%'
                THEN 1
            ELSE 0
        END
    ) AS monetary_relief_count,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN LOWER(COALESCE(f.company_response, '')) LIKE '%monetary relief%'
                    THEN 1
                ELSE 0
            END
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS monetary_relief_rate_pct
FROM fact_complaints AS f
JOIN dim_company AS c
    ON c.company_id = f.company_id
GROUP BY c.company_id, c.company_name
ORDER BY monetary_relief_rate_pct DESC, complaint_count DESC, c.company_name;

-- 3. Five most common issues in each state.
-- ROW_NUMBER ranks issue counts separately within each state.
WITH issue_counts AS (
    SELECT
        s.state_code,
        f.issue_id AS issue,
        COUNT(*) AS complaint_count
    FROM fact_complaints AS f
    JOIN dim_state AS s
        ON s.state_id = f.state_id
    GROUP BY s.state_code, f.issue_id
),
ranked_issues AS (
    SELECT
        state_code,
        issue,
        complaint_count,
        ROW_NUMBER() OVER (
            PARTITION BY state_code
            ORDER BY complaint_count DESC, issue
        ) AS issue_rank
    FROM issue_counts
)
SELECT state_code, issue, complaint_count, issue_rank
FROM ranked_issues
WHERE issue_rank <= 5
ORDER BY state_code, issue_rank;

-- 4. Overall month-over-month complaint trend.
-- A recursive month spine includes months with zero complaints, so LAG always
-- compares with the immediately preceding calendar month.
WITH RECURSIVE date_bounds AS (
    SELECT
        CAST(DATE_FORMAT(MIN(d.date_value), '%Y-%m-01') AS DATE) AS first_month,
        CAST(DATE_FORMAT(MAX(d.date_value), '%Y-%m-01') AS DATE) AS last_month
    FROM fact_complaints AS f
    JOIN dim_date AS d
        ON d.date_id = f.date_received_id
),
month_spine AS (
    SELECT first_month AS month_start
    FROM date_bounds
    WHERE first_month IS NOT NULL
    UNION ALL
    SELECT DATE_ADD(month_spine.month_start, INTERVAL 1 MONTH)
    FROM month_spine
    CROSS JOIN date_bounds
    WHERE month_spine.month_start < date_bounds.last_month
),
monthly_counts AS (
    SELECT
        YEAR(d.date_value) AS year,
        MONTH(d.date_value) AS month,
        COUNT(*) AS complaint_count
    FROM fact_complaints AS f
    JOIN dim_date AS d
        ON d.date_id = f.date_received_id
    GROUP BY YEAR(d.date_value), MONTH(d.date_value)
),
monthly_series AS (
    SELECT
        YEAR(spine.month_start) AS year,
        MONTH(spine.month_start) AS month,
        COALESCE(counts.complaint_count, 0) AS complaint_count
    FROM month_spine AS spine
    LEFT JOIN monthly_counts AS counts
        ON counts.year = YEAR(spine.month_start)
       AND counts.month = MONTH(spine.month_start)
),
monthly_changes AS (
    SELECT
        year,
        month,
        complaint_count,
        LAG(complaint_count) OVER (ORDER BY year, month) AS previous_month_count
    FROM monthly_series
)
SELECT
    year,
    month,
    complaint_count,
    previous_month_count,
    complaint_count - previous_month_count AS month_over_month_change,
    ROUND(
        100.0 * (complaint_count - previous_month_count)
            / NULLIF(previous_month_count, 0),
        2
    ) AS month_over_month_change_pct
FROM monthly_changes
ORDER BY year, month;

-- 5. Ten companies with the largest complaint volume.
-- Use this list to prioritize high-volume company relationships.
SELECT
    c.company_name,
    COUNT(*) AS complaint_count
FROM fact_complaints AS f
JOIN dim_company AS c
    ON c.company_id = f.company_id
GROUP BY c.company_id, c.company_name
ORDER BY complaint_count DESC, c.company_name
LIMIT 10;

-- 6. Timely-response rate by company.
-- Unknown response values remain NULL and are excluded from the rate denominator.
SELECT
    c.company_name,
    COUNT(*) AS complaint_count,
    SUM(CASE WHEN f.timely_response = 1 THEN 1 ELSE 0 END) AS timely_count,
    SUM(CASE WHEN f.timely_response = 0 THEN 1 ELSE 0 END) AS late_count,
    ROUND(
        100.0 * SUM(CASE WHEN f.timely_response = 1 THEN 1 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN f.timely_response IN (0, 1) THEN 1 ELSE 0 END), 0),
        2
    ) AS timely_response_rate_pct
FROM fact_complaints AS f
JOIN dim_company AS c
    ON c.company_id = f.company_id
GROUP BY c.company_id, c.company_name
ORDER BY timely_response_rate_pct DESC, complaint_count DESC, c.company_name;
