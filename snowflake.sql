-- ================================
-- SUMMIT SPORTS CRM EXPORT SETUP
-- ================================
USE DATABASE SS_RAW;
CREATE OR REPLACE SCHEMA SS_BRONZE;

CREATE STAGE IF NOT EXISTS CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE;

CREATE FILE FORMAT IF NOT EXISTS CROCEVIA_DB.RAW_DATA.PARQUET_EXPORT_FORMAT
  TYPE = PARQUET
  COMPRESSION = SNAPPY;

SHOW STAGES;

COPY INTO @CROCEVIA.RAW_DATA.CR_BRONZE_STAGE/crocevia_crm.parquet
FROM CROCEVIA.RAW_DATA.CROCEVIA_CRM
FILE_FORMAT = (FORMAT_NAME = 'CROCEVIA.RAW_DATA.PARQUET_EXPORT_FORMAT')
--SINGLE = TRUE
OVERWRITE = TRUE
--MAX_FILE_SIZE = 5000000000
;

-- Download to local (requires SnowSQL)
GET @CROCEVIA.RAW_DATA.CR_BRONZE_STAGE/crocevia_crm.parquet file:///Users/edendulk/code/crocevia/data;

put file:///Users/edendulk/code/crocevia/data/adresses-france.csv.gz @CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE;

put file:///Users/edendulk/code/crocevia/data/lieux-dits-beta-france.csv.gz @CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE;

put file:///Users/edendulk/code/crocevia/data/summit_sports_crm.parquet @CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE;

-- ================================
-- CROCEVIA CRM SETUP
-- ================================
-- Create Crocevia database and schema
CREATE DATABASE IF NOT EXISTS CROCEVIA;
CREATE SCHEMA IF NOT EXISTS CROCEVIA.RAW_DATA;

show API integrations;
CREATE OR REPLACE GIT REPOSITORY crocevia_repo
  API_INTEGRATION = git_api_integration
  GIT_CREDENTIALS = git_secret
  ORIGIN = 'https://github.com/my-account/snowflake-extensions.git';

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

