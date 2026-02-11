# pages/Insights.py
import streamlit as st
from utils.data_loader import load_data
from utils.analytics import (
    generate_insights,
    detect_anomalies,
    forecast_sales,
    compute_trends,
)
from utils.ui import render_header, render_footer, render_css, render_section_divider
from fpdf import FPDF

if "auth" not in st.session_state or not st.session_state.auth:
    st.write("Please login first.")
    st.stop()

render_css()
render_header()

if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df  # Use filtered from session if possible, but simplify

render_section_divider("Auto-Generated Insights")

insights = generate_insights(df)
st.markdown(insights)

anomalies = detect_anomalies(df)
if anomalies:
    st.warning("Anomalies Detected: " + ", ".join(anomalies))

trends = compute_trends(df)
st.info(f"Trend Analysis: {trends}")

forecast_days = st.radio("Forecast Period", [14, 30])
forecast = forecast_sales(df, forecast_days)
st.subheader("Sales Forecast")
st.line_chart(forecast)  # Simple, use Plotly if needed

# Download Insights
text_insights = insights + "\n" + trends + "\nAnomalies: " + ", ".join(anomalies)
st.download_button("Download Insights Text", text_insights, "insights.txt")

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, text_insights)
pdf_output = pdf.output(dest="S").encode("latin1")
st.download_button("Download Insights PDF", pdf_output, "insights.pdf", "application/pdf")

render_footer()
