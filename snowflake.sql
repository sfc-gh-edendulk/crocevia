-- ================================
-- SUMMIT SPORTS CRM EXPORT SETUP
-- ================================
USE DATABASE SS_RAW;
CREATE OR REPLACE SCHEMA SS_BRONZE;

CREATE STAGE IF NOT EXISTS SS_RAW.SS_BRONZE.SS_BRONZE_STAGE;

CREATE FILE FORMAT IF NOT EXISTS SS_RAW.SS_BRONZE.PARQUET_EXPORT_FORMAT
  TYPE = PARQUET
  COMPRESSION = SNAPPY;

SHOW STAGES;

COPY INTO @SS_RAW.SS_BRONZE.SS_BRONZE_STAGE/summit_sports_crm.parquet
FROM SS_101.RAW_CUSTOMER.CUSTOMER_LOYALTY
FILE_FORMAT = (FORMAT_NAME = 'SS_RAW.SS_BRONZE.PARQUET_EXPORT_FORMAT')
SINGLE = TRUE
OVERWRITE = TRUE
MAX_FILE_SIZE = 5000000000;

-- Download to local (requires SnowSQL)
GET @SS_RAW.SS_BRONZE.SS_BRONZE_STAGE/summit_sports_crm.parquet file:///Users/edendulk/code/crocevia/data;

-- ================================
-- CROCEVIA CRM SETUP
-- ================================
-- Create Crocevia database and schema
CREATE DATABASE IF NOT EXISTS CROCEVIA;
CREATE SCHEMA IF NOT EXISTS CROCEVIA.RAW_DATA;

USE DATABASE CROCEVIA;
USE SCHEMA RAW_DATA;

-- Test: Run Snowpark CRM generator (execute snowpark/generate_crocevia_crm.py)
-- This will create: CROCEVIA.RAW_DATA.CROCEVIA_CRM

-- Validate the generated data
SELECT 
    overlap_type,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM CROCEVIA.RAW_DATA.CROCEVIA_CRM
GROUP BY overlap_type
ORDER BY count DESC;

-- Sample the results
SELECT * FROM CROCEVIA.RAW_DATA.CROCEVIA_CRM LIMIT 10; 

