import json
import os

import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSessionException

SEGMENT_TABLE = "CROCEVIA_DB.GOLD_ANALYTICS.C360_CUSTOMER_SEGMENTS"
PRODUCT_REC_TABLE = "CROCEVIA_DB.GOLD_ANALYTICS.C360_SEGMENT_PRODUCT_RECS"
LOOKALIKE_TABLE = "CROCEVIA_DB.GOLD_ANALYTICS.C360_LOOKALIKE_SCORES"
CRM_TABLE = "CROCEVIA_DB.RAW_DATA.CROCEVIA_CRM"
SALES_TABLE = "CROCEVIA_DB.BRONZE_DATA.CROCEVIA_SALES_20PCT_STORES"


@st.cache_resource(show_spinner=False)
def get_session() -> Session:
    """Return active Snowpark session for Streamlit in Snowflake or local testing."""
    try:
        return get_active_session()
    except (SnowparkSessionException, NameError):
        config_env = os.environ.get("SNOWFLAKE_CONNECTIONS_C360")
        if not config_env:
            raise RuntimeError(
                "Snowpark session not found. Run inside Snowsight Streamlit or set SNOWFLAKE_CONNECTIONS_C360."
            )
        connection_parameters = json.loads(config_env)
        return Session.builder.configs(connection_parameters).create()


@st.cache_data(show_spinner=False)
def load_segment_summary(session: Session):
    return session.table(SEGMENT_TABLE).group_by("SEGMENT_LABEL").agg(
        F.count("CUSTOMER_ID").alias("CUSTOMER_COUNT"),
        F.avg("MONETARY_VALUE").alias("AVG_MONETARY"),
        F.avg("PURCHASE_FREQUENCY").alias("AVG_FREQUENCY"),
        F.avg("RECENCY_DAYS").alias("AVG_RECENCY")
    ).sort(F.col("AVG_MONETARY").desc()).to_pandas()


@st.cache_data(show_spinner=False)
def load_customer_profile(session: Session, customer_id: str):
    crm = session.table(CRM_TABLE).filter(F.col("CUSTOMER_ID") == customer_id).to_pandas()
    segments = session.table(SEGMENT_TABLE).filter(F.col("CUSTOMER_ID") == customer_id).to_pandas()
    recent_orders = (
        session.table(SALES_TABLE)
        .filter(F.col("CUSTOMER_ID") == customer_id)
        .order_by(F.col("SALE_DATE").desc())
        .limit(10)
        .to_pandas()
    )
    return crm, segments, recent_orders


@st.cache_data(show_spinner=False)
def load_product_recs(session: Session, segment_label: int):
    return session.table(PRODUCT_REC_TABLE).filter(F.col("SEGMENT_LABEL") == segment_label).order_by("PRODUCT_RANK").to_pandas()


@st.cache_data(show_spinner=False)
def load_lookalikes(session: Session, customer_id: str):
    return (
        session.table(LOOKALIKE_TABLE)
        .filter(F.col("CUSTOMER_ID") == customer_id)
        .order_by("LOOKALIKE_RANK")
        .limit(25)
        .to_pandas()
    )


def main():
    st.set_page_config(page_title="Crocevia C360 Recommender", layout="wide")
    st.title("Crocevia C360 Product Recommender")
    st.caption("Powered by Snowpark ML audience segmentation and lookalike scoring.")

    session = get_session()
    session.use_database("CROCEVIA_DB")
    session.use_schema("GOLD_ANALYTICS")
    session.use_warehouse("CROCEVIA_WH")

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Customer Selection")
        customer_options = session.table(SEGMENT_TABLE).select("CUSTOMER_ID").distinct().order_by("CUSTOMER_ID").limit(1000).to_pandas()["CUSTOMER_ID"].tolist()
        customer_id = st.selectbox("Choose customer id", customer_options)

    with cols[1]:
        st.subheader("Segment Insights")
        segment_summary = load_segment_summary(session)
        st.dataframe(segment_summary, hide_index=True, use_container_width=True)

    if not customer_id:
        st.stop()

    crm_df, segment_df, orders_df = load_customer_profile(session, customer_id)

    st.markdown("---")
    st.subheader("Customer 360 Snapshot")

    snapshot_cols = st.columns(3)
    if not segment_df.empty:
        segment_label = int(segment_df.iloc[0]["SEGMENT_LABEL"])
        snapshot_cols[0].metric("Segment", segment_label)
        snapshot_cols[1].metric("Monetary", f"€{segment_df.iloc[0]['MONETARY_VALUE']:.2f}")
        snapshot_cols[2].metric("Frequency", int(segment_df.iloc[0]["PURCHASE_FREQUENCY"]))
    if not crm_df.empty:
        st.write("**CRM Profile**")
        st.dataframe(crm_df, hide_index=True, use_container_width=True)
    if not orders_df.empty:
        st.write("**Recent Orders**")
        st.dataframe(orders_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Recommended Products")
    if not segment_df.empty:
        recs = load_product_recs(session, int(segment_df.iloc[0]["SEGMENT_LABEL"]))
        if recs.empty:
            st.info("No recommendations available for this segment yet.")
        else:
            st.dataframe(recs[["PRODUCT_NAME", "REVENUE", "PRODUCT_RANK"]], hide_index=True, use_container_width=True)
    else:
        st.warning("Customer not assigned to a segment. Run the Snowpark model first.")

    st.markdown("---")
    st.subheader("Lookalike Audience")
    lookalike_df = load_lookalikes(session, customer_id)
    if lookalike_df.empty:
        st.info("No lookalike scores yet.")
    else:
        st.dataframe(lookalike_df[["CUSTOMER_ID", "DISTANCE_TO_TOP_BUYERS", "LOOKALIKE_RANK"]], hide_index=True, use_container_width=True)

    st.markdown("---")
    st.caption("For governance: model run metadata at CROCEVIA_DB.GOLD_ANALYTICS.C360_MODEL_RUN_AUDIT")


if __name__ == "__main__":
    main()
