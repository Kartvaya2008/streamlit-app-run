# pages/Dashboard.py
import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.analytics import (
    compute_kpis,
    compute_growth,
    compute_best_worst,
    compute_region_performance,
    compute_moving_average,
    compute_cumulative,
)
from utils.charts import (
    revenue_by_product_chart,
    revenue_over_time_chart,
    region_revenue_heatmap,
    units_sold_histogram,
    top_n_products_chart,
    contribution_chart,
)
from utils.ui import (
    render_header,
    render_footer,
    render_css,
    render_section_divider,
    render_kpi_cards,
)

if "auth" not in st.session_state or not st.session_state.auth:
    st.write("Please login first.")
    st.stop()

render_css()
render_header()

# Data Loading and Filtering
st.sidebar.header("Filters")
with st.sidebar.expander("Date Range", expanded=True):
    min_date = st.session_state.df["Date"].min().date()
    max_date = st.session_state.df["Date"].max().date()
    date_range = st.date_input(
        "Select Date Range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

with st.sidebar.expander("Regions", expanded=True):
    all_regions = st.checkbox("All Regions", value=True)
    if not all_regions:
        region_filter = st.multiselect("Select Regions", st.session_state.df["Region"].unique())
    else:
        region_filter = st.session_state.df["Region"].unique()

with st.sidebar.expander("Products", expanded=True):
    product_search = st.text_input("Search Product")
    product_filter = st.multiselect(
        "Select Products",
        st.session_state.df["Product"].unique(),
        default=st.session_state.df["Product"].unique(),
    )

aggregation = st.sidebar.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])
reset = st.sidebar.button("Reset Filters")

if reset:
    # Reset to defaults
    pass  # Handled by session state absence

# Filter Data
df = st.session_state.df.copy()
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

df = df[df["Region"].isin(region_filter)]
if product_search:
    df = df[df["Product"].str.contains(product_search, case=False)]
df = df[df["Product"].isin(product_filter)]

# Aggregate
if aggregation == "Weekly":
    df = df.resample("W", on="Date").sum(numeric_only=True).reset_index()
elif aggregation == "Monthly":
    df = df.resample("M", on="Date").sum(numeric_only=True).reset_index()

# Refresh Button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.session_state.df = load_data()  # Reload

render_section_divider("Key Performance Indicators")

kpis = compute_kpis(df)
prev_kpis = compute_kpis(df)  # Placeholder, compute actual prev
growth = compute_growth(kpis, prev_kpis)
render_kpi_cards(kpis, growth)

render_section_divider("Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(revenue_by_product_chart(df), use_container_width=True)
    st.plotly_chart(units_sold_histogram(df), use_container_width=True)

with col2:
    st.plotly_chart(revenue_over_time_chart(df), use_container_width=True)
    st.plotly_chart(top_n_products_chart(df, 5), use_container_width=True)

st.plotly_chart(region_revenue_heatmap(df), use_container_width=True)
st.plotly_chart(contribution_chart(df), use_container_width=True)

render_section_divider("Data Table")
st.dataframe(df, use_container_width=True)

# Downloads
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data", csv, "filtered_data.csv", "text/csv")

render_footer()
