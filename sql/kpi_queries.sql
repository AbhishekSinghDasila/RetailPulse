-- ============================================================
-- RetailPulse — SQL Analytics Queries (Phase 2 Preview)
-- ============================================================
-- Run against: data/retailpulse.db (SQLite)
-- These queries demonstrate window functions, CTEs, cohort
-- analysis, and RFM scoring — all common interview topics.
-- ============================================================


-- ── Q1: Monthly Revenue Trend ────────────────────────────────
-- Shows MoM growth rate using LAG() window function
SELECT
    strftime('%Y-%m', order_date)                              AS month,
    COUNT(*)                                                   AS total_orders,
    ROUND(SUM(total_amount), 2)                                AS gross_revenue,
    ROUND(SUM(CASE WHEN status='Delivered' THEN total_amount END), 2) AS delivered_revenue,
    ROUND(
        100.0 * (SUM(total_amount) - LAG(SUM(total_amount)) OVER (ORDER BY strftime('%Y-%m', order_date)))
        / LAG(SUM(total_amount)) OVER (ORDER BY strftime('%Y-%m', order_date)),
        1
    )                                                          AS mom_growth_pct
FROM orders
GROUP BY month
ORDER BY month;


-- ── Q2: Revenue by Category with % Share ────────────────────
SELECT
    p.category,
    COUNT(DISTINCT o.order_id)                                 AS orders,
    ROUND(SUM(oi.line_total), 2)                               AS revenue,
    ROUND(100.0 * SUM(oi.line_total) / SUM(SUM(oi.line_total)) OVER (), 1) AS pct_of_total,
    ROUND(AVG(p.price), 0)                                     AS avg_price
FROM order_items oi
JOIN orders   o ON oi.order_id   = o.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.status = 'Delivered'
GROUP BY p.category
ORDER BY revenue DESC;


-- ── Q3: Customer Cohort Retention (monthly) ──────────────────
-- First-purchase cohort vs repeat purchase months
WITH first_orders AS (
    SELECT
        customer_id,
        MIN(strftime('%Y-%m', order_date)) AS cohort_month
    FROM orders
    WHERE status != 'Cancelled'
    GROUP BY customer_id
),
customer_activity AS (
    SELECT
        o.customer_id,
        fo.cohort_month,
        strftime('%Y-%m', o.order_date)    AS activity_month
    FROM orders o
    JOIN first_orders fo ON o.customer_id = fo.customer_id
    WHERE o.status != 'Cancelled'
)
SELECT
    cohort_month,
    activity_month,
    COUNT(DISTINCT customer_id)            AS active_customers
FROM customer_activity
GROUP BY cohort_month, activity_month
ORDER BY cohort_month, activity_month;


-- ── Q4: RFM Scoring ──────────────────────────────────────────
-- Recency / Frequency / Monetary — classic segmentation model
WITH rfm_base AS (
    SELECT
        customer_id,
        CAST(julianday('2024-06-30') - julianday(MAX(order_date)) AS INTEGER) AS recency_days,
        COUNT(*)                                                               AS frequency,
        ROUND(SUM(total_amount), 2)                                           AS monetary
    FROM orders
    WHERE status = 'Delivered'
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,  -- lower recency = better
        NTILE(5) OVER (ORDER BY frequency    DESC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary     DESC) AS m_score
    FROM rfm_base
)
SELECT
    customer_id,
    recency_days,
    frequency,
    monetary,
    r_score, f_score, m_score,
    (r_score + f_score + m_score)           AS rfm_total,
    CASE
        WHEN (r_score + f_score + m_score) >= 13 THEN 'Champions'
        WHEN (r_score + f_score + m_score) >= 10 THEN 'Loyal'
        WHEN r_score >= 4 AND f_score <= 2        THEN 'New Customer'
        WHEN r_score <= 2 AND f_score >= 3        THEN 'At-Risk'
        ELSE 'Needs Attention'
    END                                     AS rfm_segment
FROM rfm_scored
ORDER BY rfm_total DESC;


-- ── Q5: Sales Funnel Analysis ────────────────────────────────
SELECT
    exit_page,
    COUNT(*)                                                       AS sessions_exited,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)            AS pct_of_exits,
    ROUND(100.0 * SUM(converted) / COUNT(*), 1)                   AS conversion_rate
FROM sessions
GROUP BY exit_page
ORDER BY
    CASE exit_page
        WHEN 'home'         THEN 1
        WHEN 'category'     THEN 2
        WHEN 'product'      THEN 3
        WHEN 'cart'         THEN 4
        WHEN 'checkout'     THEN 5
        WHEN 'confirmation' THEN 6
    END;


-- ── Q6: Top 10 Products by Revenue ──────────────────────────
SELECT
    p.product_id,
    p.name,
    p.category,
    p.price,
    p.rating,
    SUM(oi.quantity)                     AS units_sold,
    ROUND(SUM(oi.line_total), 2)         AS total_revenue,
    RANK() OVER (ORDER BY SUM(oi.line_total) DESC) AS revenue_rank
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders   o ON oi.order_id   = o.order_id
WHERE o.status = 'Delivered'
GROUP BY p.product_id
ORDER BY total_revenue DESC
LIMIT 10;


-- ── Q7: Average Order Value by Acquisition Channel ───────────
SELECT
    c.acquisition_channel,
    COUNT(DISTINCT o.order_id)           AS total_orders,
    COUNT(DISTINCT o.customer_id)        AS unique_customers,
    ROUND(AVG(o.total_amount), 2)        AS avg_order_value,
    ROUND(SUM(o.total_amount), 2)        AS total_revenue
FROM orders     o
JOIN customers  c ON o.customer_id = c.customer_id
WHERE o.status = 'Delivered'
GROUP BY c.acquisition_channel
ORDER BY total_revenue DESC;


-- ── Q8: Customer Lifetime Value (LTV) ────────────────────────
WITH clv AS (
    SELECT
        o.customer_id,
        c.segment,
        c.city,
        c.acquisition_channel,
        COUNT(DISTINCT o.order_id)        AS orders_placed,
        ROUND(SUM(o.total_amount), 2)     AS total_spent,
        ROUND(AVG(o.total_amount), 2)     AS avg_order_value,
        MIN(o.order_date)                 AS first_order,
        MAX(o.order_date)                 AS last_order
    FROM orders    o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'Delivered'
    GROUP BY o.customer_id
)
SELECT *,
    ROUND(
        total_spent / NULLIF(
            CAST(julianday(last_order) - julianday(first_order) AS REAL) / 30.0,
            0
        ),
        2
    )                                     AS monthly_revenue_rate,
    NTILE(4) OVER (ORDER BY total_spent)  AS ltv_quartile
FROM clv
ORDER BY total_spent DESC;


-- ── Q9: Payment Method Preference by City ────────────────────
SELECT
    c.city,
    o.payment_method,
    COUNT(*)                              AS orders,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY c.city), 1) AS city_share_pct
FROM orders    o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.city, o.payment_method
ORDER BY c.city, orders DESC;


-- ── Q10: Weekly Sales with 4-Week Rolling Average ────────────
SELECT
    strftime('%Y-W%W', order_date)        AS week,
    ROUND(SUM(total_amount), 2)           AS weekly_revenue,
    ROUND(
        AVG(SUM(total_amount)) OVER (
            ORDER BY strftime('%Y-W%W', order_date)
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ),
        2
    )                                     AS rolling_4w_avg
FROM orders
WHERE status = 'Delivered'
GROUP BY week
ORDER BY week;


-- ── Q11: Cancellation Rate by Category ───────────────────────
SELECT
    p.category,
    COUNT(DISTINCT o.order_id)            AS total_orders,
    SUM(CASE WHEN o.status='Cancelled' THEN 1 ELSE 0 END) AS cancelled,
    ROUND(
        100.0 * SUM(CASE WHEN o.status='Cancelled' THEN 1 ELSE 0 END) / COUNT(*),
        1
    )                                     AS cancel_rate_pct
FROM orders    o
JOIN order_items oi ON o.order_id    = oi.order_id
JOIN products    p  ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY cancel_rate_pct DESC;


-- ── Q12: Device-wise Conversion Funnel ───────────────────────
SELECT
    device,
    COUNT(*)                              AS total_sessions,
    SUM(converted)                        AS conversions,
    ROUND(100.0 * SUM(converted)/COUNT(*),2) AS conversion_rate_pct,
    ROUND(AVG(duration_seconds)/60.0, 1) AS avg_duration_min
FROM sessions
GROUP BY device
ORDER BY conversion_rate_pct DESC;
