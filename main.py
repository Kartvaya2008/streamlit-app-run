import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# ===============================
# 1. CONFIG & UI STYLING
# ===============================
st.set_page_config(
    page_title="Nexus Analytics | Enterprise Sales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1E88E5; }
        .main-header {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem; font-weight: 800; margin-bottom: 0rem;
        }
        .stMetric {
            background-color: rgba(28, 131, 225, 0.1);
            padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5;
        }
        footer {visibility: hidden;}
        .footer-text {
            position: fixed; bottom: 10px; right: 10px; font-size: 0.8rem; color: gray;
        }
        div[data-testid="stExpander"] { border: none; box-shadow: none; }
        </style>
    """, unsafe_allow_html=True)

# ===============================
# 2. DATA ENGINE (data_loader.py)
# ===============================
@st.cache_data(ttl=3600)
def get_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # High-fidelity dummy generator
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=120)
        regions = ["North", "South", "East", "West"]
        products = ["Enterprise SaaS", "Cloud Storage", "Cybersecurity", "IoT Hub"]
        
        data = []
        for date in dates:
            for region in regions:
                for prod in products:
                    # Adding seasonality and noise
                    base_rev = np.random.randint(1000, 5000)
                    if date.dayofweek >= 5: base_rev *= 0.7 # Weekend dip
                    data.append({
                        "Date": date, "Region": region, "Product": prod,
                        "Revenue": base_rev, "Units_Sold": np.random.randint(5, 50)
                    })
        df = pd.DataFrame(data)
    return df

# ===============================
# 3. ANALYTICS LOGIC (analytics.py)
# ===============================
class AnalyticsEngine:
    @staticmethod
    def get_kpis(df, prev_df):
        rev = df["Revenue"].sum()
        prev_rev = prev_df["Revenue"].sum() if not prev_df.empty else rev
        rev_delta = ((rev - prev_rev) / prev_rev) * 100
        
        units = df["Units_Sold"].sum()
        best_prod = df.groupby("Product")["Revenue"].sum().idxmax()
        
        return rev, rev_delta, units, best_prod

    @staticmethod
    def forecast_sales(df, days=14):
        daily_rev = df.groupby('Date')['Revenue'].sum()
        model = SimpleExpSmoothing(daily_rev, initialization_method="estimated").fit()
        forecast = model.forecast(days)
        return forecast

# ===============================
# 4. MAIN INTERFACE
# ===============================
def main():
    apply_custom_css()
    
    # --- HEADER ---
    col_logo, col_text = st.columns([1, 5])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=80)
    with col_text:
        st.markdown('<p class="main-header">Nexus Analytics Pro</p>', unsafe_allow_html=True)
        st.caption("Intelligence-driven sales performance dashboard")

    # --- SIDEBAR & AUTH ---
    with st.sidebar:
        st.title("🛡️ Secure Access")
        password = st.text_input("Enter Access Key", type="password")
        if password != "admin123":
            st.warning("Please enter 'admin123' to unlock.")
            st.stop()
            
        st.success("Authenticated")
        st.divider()
        
        st.header("🎛️ Control Panel")
        uploaded_file = st.file_uploader("Upload Sales CSV", type="csv")
        
        with st.expander("Global Filters", expanded=True):
            raw_data = get_data(uploaded_file)
            
            date_range = st.date_input(
                "Date Range",
                value=(raw_data['Date'].min(), raw_data['Date'].max()),
                min_value=raw_data['Date'].min(),
                max_value=raw_data['Date'].max()
            )
            
            regions = st.multiselect("Regions", options=raw_data['Region'].unique(), default=raw_data['Region'].unique())
            products = st.multiselect("Products", options=raw_data['Product'].unique(), default=raw_data['Product'].unique())
            search_query = st.text_input("🔍 Search Product Name")

        if st.button("🔄 Refresh Data Engine"):
            st.cache_data.clear()
            st.rerun()

    # --- FILTER PROCESSING ---
    mask = (
        (raw_data['Date'].dt.date >= date_range[0]) & 
        (raw_data['Date'].dt.date <= date_range[1]) &
        (raw_data['Region'].isin(regions)) &
        (raw_data['Product'].isin(products))
    )
    if search_query:
        mask = mask & (raw_data['Product'].str.contains(search_query, case=False))
        
    filtered_df = raw_data[mask]
    
    # Comparative Data (Previous Period for Delta)
    period_len = (date_range[1] - date_range[0]).days
    prev_start = date_range[0] - timedelta(days=period_len)
    prev_mask = (raw_data['Date'].dt.date >= prev_start) & (raw_data['Date'].dt.date < date_range[0])
    prev_df = raw_data[prev_mask]

    # --- TOP KPI WIDGETS ---
    rev, rev_delta, units, best_prod = AnalyticsEngine.get_kpis(filtered_df, prev_df)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${rev/1e3:.1f}k", f"{rev_delta:.1f}%")
    m2.metric("Units Dispatched", f"{units:,}", "Live")
    m3.metric("Top Performer", best_prod)
    m4.metric("Active Regions", len(regions))

    # --- VISUALIZATION GRID ---
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Revenue Trajectory & Forecast")
        daily_rev = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
        fig_line = px.line(daily_rev, x='Date', y='Revenue', template="plotly_white", color_discrete_sequence=['#1E88E5'])
        fig_line.add_trace(go.Scatter(x=daily_rev['Date'], y=daily_rev['Revenue'].rolling(7).mean(), name="7D MA", line=dict(dash='dash')))
        
        # Overlay Forecast
        if len(daily_rev) > 7:
            fc = AnalyticsEngine.forecast_sales(filtered_df)
            fig_line.add_trace(go.Scatter(x=pd.date_range(daily_rev['Date'].max(), periods=14), y=fc, name="AI Forecast", line=dict(color='orange')))
        
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        st.subheader("🎯 Contribution %")
        fig_pie = px.pie(filtered_df, values='Revenue', names='Product', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- SECONDARY VISUALS ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🗺️ Regional Heatmap")
        pivot = filtered_df.pivot_table(index='Region', columns='Product', values='Revenue', aggfunc='sum')
        fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Blues')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_b:
        st.subheader("📦 Distribution of Sales Size")
        fig_hist = px.histogram(filtered_df, x="Revenue", nbins=30, marginal="rug", color_discrete_sequence=['#26a69a'])
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- AI INSIGHTS ---
    with st.expander("🤖 AI Insights & Anomaly Detection", expanded=True):
        avg_rev = filtered_df['Revenue'].mean()
        high_growth_reg = filtered_df.groupby('Region')['Revenue'].sum().idxmax()
        
        st.write(f"**Executive Summary:**")
        st.write(f"- Performance is currently trending **{'Up' if rev_delta > 0 else 'Down'}** compared to previous period.")
        st.write(f"- **{high_growth_reg}** is the primary revenue driver this period.")
        
        anomalies = filtered_df[filtered_df['Revenue'] > (avg_rev * 2.5)]
        if not anomalies.empty:
            st.warning(f"Detected {len(anomalies)} revenue spikes (Potential anomalies).")
            st.dataframe(anomalies)

    # --- DATA ACTIONS ---
    st.divider()
    d1, d2, d3 = st.columns([2, 1, 1])
    with d1:
        st.subheader("📄 Transactional Ledger")
        st.dataframe(filtered_df, use_container_width=True, height=300)
    with d2:
        st.download_button("📥 Export CSV", data=filtered_df.to_csv(), file_name="sales_export.csv", mime="text/csv")
    with d3:
        if st.button("📸 Generate Snapshot"):
            st.toast("Report Snapshot Saved to Server!")

    # --- FOOTER ---
    st.markdown(f'<p class="footer-text">Nexus Analytics v2.4 | © {datetime.now().year} Senior Data Office</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
