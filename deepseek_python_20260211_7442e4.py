import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import Config
from data_loader import DataLoader
from analytics import SalesAnalytics
from charts import ChartBuilder
from ui import UIComponents
from utils import Utils

# Page configuration
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
UIComponents.apply_custom_css()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'kpis' not in st.session_state:
    st.session_state.kpis = None

class SalesDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.analytics = SalesAnalytics()
        self.charts = ChartBuilder()
        self.ui = UIComponents()
        self.utils = Utils()
        
    def run(self):
        """Run the dashboard application"""
        # Header
        self.ui.create_header()
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        self.create_main_content()
        
        # Footer
        self.ui.create_footer()
    
    def create_sidebar(self):
        """Create sidebar with filters"""
        with st.sidebar:
            st.image(
                "https://img.freepik.com/free-vector/indian-chai-tea-glass-promo-poster_1017-60340.jpg?semt=ais_wordcount_boost&w=740&q=80",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("### 📊 Dashboard Controls")
            
            # Data Source Selection
            with st.expander("📂 Data Source", expanded=True):
                data_source = st.radio(
                    "Select Data Source",
                    ["Sample Data", "Upload CSV/Excel"],
                    key="data_source"
                )
                
                if data_source == "Upload CSV/Excel":
                    uploaded_file = st.file_uploader(
                        "Choose a file",
                        type=['csv', 'xlsx', 'xls'],
                        key="file_uploader"
                    )
                    
                    if uploaded_file:
                        try:
                            st.session_state.data = self.data_loader.load_uploaded_data(uploaded_file)
                            st.success(f"✅ Loaded {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error loading file: {e}")
                    else:
                        # Fallback to sample data
                        st.session_state.data = self.data_loader.generate_sample_data()
                else:
                    # Use sample data
                    st.session_state.data = self.data_loader.generate_sample_data()
            
            # Filters
            with st.expander("🔍 Filters", expanded=True):
                # Date Range
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=Config.DEFAULT_START_DATE,
                        key="start_date_filter"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=Config.DEFAULT_END_DATE,
                        key="end_date_filter"
                    )
                
                # Region Filter
                if st.session_state.data is not None:
                    all_regions = sorted(st.session_state.data['region'].unique())
                    selected_regions = st.multiselect(
                        "Select Regions",
                        options=all_regions,
                        default=all_regions,
                        key="region_filter"
                    )
                    
                    # Product Filter with Search
                    product_search = st.text_input(
                        "🔍 Search Products",
                        placeholder="Type to search...",
                        key="product_search"
                    )
                    
                    all_products = sorted(st.session_state.data['product'].unique())
                    if product_search:
                        filtered_products = [p for p in all_products 
                                          if product_search.lower() in p.lower()]
                    else:
                        filtered_products = all_products
                    
                    selected_products = st.multiselect(
                        "Select Products",
                        options=filtered_products,
                        default=filtered_products[:5] if len(filtered_products) > 5 else filtered_products,
                        key="product_filter"
                    )
            
            # Action