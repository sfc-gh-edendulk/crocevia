# Crocevia C360 Demo

End-to-end Snowflake demo showcasing Snowpark ML segmentation and a Streamlit product recommender on Crocevia data.

## Components

- **Notebook**: `notebooks/crocevia_c360_audience_segmentation.ipynb` – builds RFM features, trains Snowpark ML models, writes results to GOLD tables.
- **Streamlit App**: `streamlit/customer_recommender.py` – customer picker, segment insights, product recommendations, lookalike audience.
- **Snowflake SQL**: creation scripts for tasks, stages, file formats, and orchestration.

## Prerequisites

1. Snowflake objects available:
   - Database: `CROCEVIA_DB`
   - Warehouse: `CROCEVIA_WH`
   - Stage: `CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE`
   - Bronze tables: `CROCEVIA_SALES_20PCT_STORES`, `CROCEVIA_PRODUCTS`
   - Raw CRM table: `CROCEVIA_CRM`
2. Python packages installed in environment:
   - `snowflake-snowpark-python`
   - `snowflake-ml-python`
   - `streamlit`
3. Ensure append-only writes remain enabled; set `write_mode = "append"` unless explicitly approved to overwrite.

## Run Book


1. **Prerequisite check**
   - Run `snowflake_sql/check_sales_date_spine.sql` in Snowsight Worksheets.
   - Review missing date ranges; if evenly spaced gaps or large intervals appear, address Bronze ingestion before notebook execution.

### Handling date spine gaps

The notebook automatically checks `CROCEVIA_DB.BRONZE_DATA.CROCEVIA_SALES_20PCT_STORES` for missing days. If gaps exist, it falls back to the longest contiguous window and prints the adjusted start/end dates. If the reduced window is not acceptable, investigate upstream Bronze loaders before rerunning the notebook.
1. **Notebook execution**
   - Open in Snowsight and run sections sequentially.
   - Confirms data completeness via date spine check.
   - Persists outputs:
     - `CROCEVIA_DB.GOLD_ANALYTICS.C360_CUSTOMER_SEGMENTS`
     - `C360_SEGMENT_PRODUCT_RECS`
     - `C360_LOOKALIKE_SCORES`
     - `C360_MODEL_RUN_AUDIT`
   - Model artifacts stored in stage `@CROCEVIA_DB.RAW_DATA.CR_BRONZE_STAGE/c360/models/{run_key}`.
2. **Streamlit deployment**
   - Upload `streamlit/customer_recommender.py` in Snowsight Streamlit.
   - Set environment variable `SNOWFLAKE_CONNECTIONS_C360` when running locally.
   - App requires tables from the notebook run.
3. **Governance**
   - Audit table monitors counts per run key.
   - Update runbook in `docs/` for policy changes.

## CI/CD Notes

- Add lint/type checks covering notebook Python if exported as `.py` using `nbconvert`.
- Use semantic versioning for handlers; update `docs/changelog.md` when promoting.
- PR review required for stored procedure changes. Keep append-only policy documented.

