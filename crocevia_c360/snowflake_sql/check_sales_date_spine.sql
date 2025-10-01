-- crocevia_c360/snowflake_sql/check_sales_date_spine.sql
-- Validates sale date completeness for CROCEVIA sales tables before running Snowpark ML notebook.

USE DATABASE CROCEVIA_DB;

-- Parameters
SET INPUT_TABLE = 'BRONZE_DATA.CROCEVIA_SALES_20PCT_STORES';
SET START_DATE = DATEADD('day', -365, CURRENT_DATE());
SET END_DATE = CURRENT_DATE();

-- Note: GENERATOR rowcount must be constant, so we cap at 400 days (covers >1 year).
WITH filtered_sales AS (
    SELECT SALE_DATE
    FROM IDENTIFIER($INPUT_TABLE)
    WHERE SALE_DATE BETWEEN $START_DATE AND $END_DATE
      AND CUSTOMER_ID IS NOT NULL
),
expected_spine AS (
    SELECT DATEADD('day', seq4(), $START_DATE) AS spine_date
    FROM TABLE(GENERATOR(ROWCOUNT => 400))
    WHERE spine_date BETWEEN $START_DATE AND $END_DATE
),
missing_dates AS (
    SELECT s.spine_date
    FROM expected_spine s
    LEFT JOIN filtered_sales f
        ON s.spine_date = f.SALE_DATE
    WHERE f.SALE_DATE IS NULL
),
consecutive_sequences AS (
    SELECT
        spine_date,
        DATEADD('day', -ROW_NUMBER() OVER (ORDER BY spine_date), spine_date) AS grp
    FROM missing_dates
),
ranked_gaps AS (
    SELECT
        MIN(spine_date) AS gap_start,
        MAX(spine_date) AS gap_end,
        COUNT(*) AS gap_length
    FROM consecutive_sequences
    GROUP BY grp
),
observed_days AS (
    SELECT COUNT(DISTINCT SALE_DATE) AS observed_day_count
    FROM filtered_sales
)
SELECT
    $INPUT_TABLE AS table_name,
    $START_DATE AS window_start,
    $END_DATE AS window_end,
    observed_day_count,
    DATEDIFF('day', $START_DATE, $END_DATE) + 1 AS expected_day_count,
    (DATEDIFF('day', $START_DATE, $END_DATE) + 1) - observed_day_count AS missing_day_count,
    gap_start,
    gap_end,
    gap_length
FROM observed_days
LEFT JOIN ranked_gaps
ORDER BY gap_length DESC;
