# Crocevia C360 Operational Runbook

## Purpose
Guide analysts through executing the Crocevia C360 segmentation pipeline and Streamlit product recommender in Snowflake.

## Contacts
- Product Owner: TBD
- Data Engineering: TBD
- Analytics Engineering: TBD

## Pre-flight Checklist
1. Warehouse `CROCEVIA_WH` set to adequate size (recommend `XSMALL` or larger).
2. Confirm data freshness in Bronze (`CROCEVIA_SALES_20PCT_STORES`, `CROCEVIA_PRODUCTS`) and Raw CRM (`CROCEVIA_CRM`).
3. Ensure stage `CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE` has latest model code (if custom handlers used).
4. Validate append-only policy still enforced (no pending overwrite requests).

## Execution Steps
1. Open `notebooks/crocevia_c360_audience_segmentation.ipynb` in Snowsight.
2. Set analysis window if running for historical backfill (defaults to last 365 days).
3. Run all cells. On success, verify:
   - Segment table updated (`C360_CUSTOMER_SEGMENTS`).
   - Segment product recommendations table refreshed (`C360_SEGMENT_PRODUCT_RECS`).
   - Lookalike scores table refreshed (`C360_LOOKALIKE_SCORES`).
   - Audit table appended with new run key (`C360_MODEL_RUN_AUDIT`).
4. Launch Streamlit app `/streamlit/customer_recommender.py` in Snowsight to validate recommendations.

## Monitoring & Alerting
- Audit table row counts triggered via downstream Looker/BI dashboards.
- Add optional notification by scheduling a Snowflake task to evaluate anomalies (future enhancement).

## Failure Modes & Recovery
- **Missing Dates**: Notebook raises hard failure on date spine gaps. Investigate Bronze ingestion jobs.
- **Model Save Failure**: Ensure RAW stage permissions and storage quota.
- **Streamlit Errors**: Confirm `SNOWFLAKE_CONNECTIONS_C360` env variable for local runs or run inside Snowsight.

## Change Management
- Use Git PRs for notebook/app updates; tag releases using semantic versioning (e.g., `v1.0.0`).
- Document changes in `docs/changelog.md`.
- After validation, merge `feature/crocevia-snowflake-demo` to `main` with approval.
