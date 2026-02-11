"""
Sales Analytics Pro - Production Dashboard
Author: Data Science Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import logging
from pathlib import Path
import hashlib
import hmac
import json
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.salesanalyticspro.com/help',
        'Report a bug': 'https://www.salesanalyticspro.com/bug',
        'About': "# Sales Analytics Pro\nVersion 2.0.0\nEnterprise-Grade Sales Analytics Dashboard"
    }
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration settings"""
    
    # Application Info
    APP_NAME = "Sales Analytics Pro"
    APP_VERSION = "2.0.0"
    APP_DESCRIPTION = "Enterprise-Grade Sales Analytics Dashboard"
    APP_AUTHOR = "Data Science Team"
    COPYRIGHT_YEAR = datetime.now().year
    
    # Authentication
    REQUIRE_AUTHENTICATION = True
    
    # Data Settings
    SAMPLE_SIZE = 1000
    CACHE_TTL = 3600
    
    # Colors
    PRIMARY_COLOR = "#FF4B4B"
    SECONDARY_COLOR = "#0068C9"
    SUCCESS_COLOR = "#00CC96"
    WARNING_COLOR = "#FFB800"
    DANGER_COLOR = "#DC3545"
    INFO_COLOR = "#17A2B8"
    
    # Chart Colors
    CHART_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]
    
    # Date Configuration
    DEFAULT_START_DATE = datetime(2023, 1, 1)
    DEFAULT_END_DATE = datetime(2024, 1, 1)

# ============================================================================
# AUTHENTICATION
# ============================================================================

class Authenticator:
    """Handle user authentication"""
    
    @staticmethod
    def hash_password(password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def authenticate(username, password):
        """Authenticate user credentials"""
        # Demo credentials
        valid_users = {
            "admin": Authenticator.hash_password("admin123"),
            "demo": Authenticator.hash_password("demo123")
        }
        
        if username in valid_users:
            if hmac.compare_digest(
                Authenticator.hash_password(password),
                valid_users[username]
            ):
                st.session_state.user = {
                    "username": username,
                    "name": "Administrator" if username == "admin" else "Demo User",
                    "role": "admin" if username == "admin" else "viewer",
                    "authenticated_at": datetime.now().isoformat()
                }
                return True
        return False

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Handle data loading, generation, and caching"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def generate_sample_data():
        """Generate realistic sales data"""
        np.random.seed(42)
        
        products = [
            'Chai Tea Premium', 'Masala Chai', 'Green Tea Deluxe',
            'Herbal Infusion', 'Oolong Supreme', 'Black Tea Classic',
            'Matcha Zen', 'Earl Grey', 'Jasmine Pearl', 'Darjeeling Gold'
        ]
        
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        dates = pd.date_range(
            start=Config.DEFAULT_START_DATE,
            end=Config.DEFAULT_END_DATE,
            freq='D'
        )[:Config.SAMPLE_SIZE]
        
        data = []
        for date in dates:
            for region in regions:
                for product in np.random.choice(products, 3, replace=False):
                    base_price = np.random.uniform(5, 25)
                    units = np.random.poisson(15)
                    
                    # Add seasonality
                    if date.month in [11, 12]:
                        units *= np.random.uniform(1.3, 1.8)
                    if date.weekday() >= 5:
                        units *= np.random.uniform(1.1, 1.4)
                    
                    revenue = base_price * units
                    
                    # Add anomalies
                    if np.random.random() < 0.02:
                        units *= np.random.uniform(2, 4)
                        revenue = base_price * units
                    
                    data.append({
                        'date': date,
                        'region': region,
                        'product': product,
                        'units_sold': int(units),
                        'unit_price': round(base_price, 2),
                        'revenue': round(revenue, 2),
                        'customer_rating': round(np.random.uniform(3.5, 5.0), 1)
                    })
        
        df = pd.DataFrame(data)
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.day_name()
        
        return df
    
    @staticmethod
    def load_uploaded_data(uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel file.")
                return None
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Convert date column
            date_cols = ['date', 'order_date', 'sale_date', 'transaction_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.rename(columns={col: 'date'}, inplace=True)
                    break
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def filter_data(df, start_date, end_date, regions, products):
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Date filter
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
        
        # Region filter
        if regions:
            filtered_df = filtered_df[filtered_df['region'].isin(regions)]
        
        # Product filter
        if products:
            filtered_df = filtered_df[filtered_df['product'].isin(products)]
        
        return filtered_df

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class SalesAnalytics:
    """Advanced analytics and insights generation"""
    
    @staticmethod
    def calculate_kpis(df):
        """Calculate key performance indicators"""
        total_revenue = df['revenue'].sum()
        total_units = df['units_sold'].sum()
        avg_order_value = total_revenue / len(df) if len(df) > 0 else 0
        
        # Growth vs previous period
        current_month = df[df['date'] >= (datetime.now() - timedelta(days=30))]['revenue'].sum()
        previous_month = df[
            (df['date'] >= (datetime.now() - timedelta(days=60))) &
            (df['date'] < (datetime.now() - timedelta(days=30)))
        ]['revenue'].sum()
        
        revenue_growth = ((current_month - previous_month) / previous_month * 100) if previous_month > 0 else 0
        
        # Product performance
        product_performance = df.groupby('product')['revenue'].sum().reset_index()
        best_product = product_performance.loc[product_performance['revenue'].idxmax()] if len(product_performance) > 0 else None
        worst_product = product_performance.loc[product_performance['revenue'].idxmin()] if len(product_performance) > 0 else None
        
        return {
            'total_revenue': round(total_revenue, 2),
            'total_units': int(total_units),
            'avg_order_value': round(avg_order_value, 2),
            'revenue_growth': round(revenue_growth, 1),
            'total_orders': len(df),
            'unique_products': df['product'].nunique(),
            'best_product': best_product['product'] if best_product is not None else 'N/A',
            'best_product_revenue': round(best_product['revenue'], 2) if best_product is not None else 0,
            'worst_product': worst_product['product'] if worst_product is not None else 'N/A',
            'worst_product_revenue': round(worst_product['revenue'], 2) if worst_product is not None else 0,
        }
    
    @staticmethod
    def generate_insights(df):
        """Generate automated insights from data"""
        insights = []
        
        if len(df) == 0:
            return insights
        
        # Weekly trend
        weekly_sales = df.groupby(pd.Grouper(key='date', freq='W'))['revenue'].sum()
        if len(weekly_sales) > 1:
            weekly_growth = (weekly_sales.iloc[-1] - weekly_sales.iloc[-2]) / weekly_sales.iloc[-2] * 100
            
            if weekly_growth > 10:
                insights.append({
                    'type': 'success',
                    'icon': '📈',
                    'title': 'Strong Weekly Growth',
                    'description': f'Weekly revenue grew by {weekly_growth:.1f}% compared to previous week'
                })
            elif weekly_growth < -5:
                insights.append({
                    'type': 'warning',
                    'icon': '⚠️',
                    'title': 'Revenue Decline Alert',
                    'description': f'Weekly revenue decreased by {abs(weekly_growth):.1f}%'
                })
        
        # Regional performance
        region_performance = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
        if len(region_performance) > 0:
            best_region = region_performance.index[0]
            region_share = (region_performance.iloc[0] / region_performance.sum() * 100)
            insights.append({
                'type': 'info',
                'icon': '📍',
                'title': 'Regional Leader',
                'description': f'{best_region} leads with {region_share:.1f}% of total revenue'
            })
        
        # Top products
        top_products = df.groupby('product')['revenue'].sum().nlargest(3)
        if len(top_products) > 0:
            top_share = (top_products.sum() / df['revenue'].sum() * 100)
            insights.append({
                'type': 'success',
                'icon': '🏆',
                'title': 'Product Concentration',
                'description': f'Top 3 products contribute {top_share:.1f}% of revenue'
            })
        
        return insights
    
    @staticmethod
    def detect_anomalies(df):
        """Detect anomalous sales patterns"""
        df['is_anomaly'] = False
        
        if len(df) > 0:
            # Simple statistical anomaly detection
            mean_revenue = df['revenue'].mean()
            std_revenue = df['revenue'].std()
            threshold = mean_revenue + 2 * std_revenue
            
            df['is_anomaly'] = df['revenue'] > threshold
        
        return df
    
    @staticmethod
    def forecast_revenue(df, days=30):
        """Simple revenue forecasting"""
        try:
            daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
            
            # Use last 7 days average for forecast
            last_7_avg = daily_revenue['revenue'].tail(7).mean()
            
            forecast_dates = pd.date_range(
                start=daily_revenue['date'].max() + timedelta(days=1),
                periods=days,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'revenue': last_7_avg,
                'is_forecast': True
            })
            
            historical_df = pd.DataFrame({
                'date': daily_revenue['date'],
                'revenue': daily_revenue['revenue'],
                'is_forecast': False
            })
            
            return pd.concat([historical_df, forecast_df], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_moving_average(df, window=7):
        """Calculate moving average"""
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        daily_revenue = daily_revenue.set_index('date')
        return daily_revenue['revenue'].rolling(window=window).mean()

# ============================================================================
# CHART BUILDER
# ============================================================================

class ChartBuilder:
    """Create interactive Plotly charts"""
    
    @staticmethod
    def create_revenue_trend_chart(df):
        """Create revenue over time chart"""
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color=Config.PRIMARY_COLOR, width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.1)'
        ))
        
        # Add moving average
        ma = daily_revenue['revenue'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=ma,
            mode='lines',
            name='7-day MA',
            line=dict(color=Config.SUCCESS_COLOR, width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Revenue Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_product_performance_chart(df):
        """Create product revenue bar chart"""
        product_revenue = df.groupby('product')['revenue'].sum().reset_index()
        product_revenue = product_revenue.sort_values('revenue', ascending=True)
        
        fig = px.bar(
            product_revenue,
            y='product',
            x='revenue',
            orientation='h',
            title='Product Revenue Distribution',
            color='revenue',
            color_continuous_scale='Viridis',
            text=product_revenue['revenue'].apply(lambda x: f'${x:,.0f}')
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Revenue ($)',
            yaxis_title='',
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        fig.update_traces(textposition='outside')
        
        return fig
    
    @staticmethod
    def create_region_heatmap(df):
        """Create region vs product heatmap"""
        pivot_table = pd.pivot_table(
            df,
            values='revenue',
            index='region',
            columns='product',
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='YlOrRd',
            hoverongaps=False,
            text=np.round(pivot_table.values, 0),
            texttemplate='$%{text:,.0f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Region vs Product Revenue Matrix',
            xaxis_title='Product',
            yaxis_title='Region',
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def create_distribution_chart(df):
        """Create units sold distribution"""
        fig = px.histogram(
            df,
            x='units_sold',
            nbins=30,
            title='Units Sold Distribution',
            color_discrete_sequence=[Config.SECONDARY_COLOR],
            marginal='box'
        )
        
        fig.update_layout(
            xaxis_title='Units Sold',
            yaxis_title='Frequency',
            height=350,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_cumulative_revenue_chart(df):
        """Create cumulative revenue chart"""
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        daily_revenue['cumulative'] = daily_revenue['revenue'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['cumulative'],
            mode='lines',
            name='Cumulative Revenue',
            line=dict(color=Config.SUCCESS_COLOR, width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.1)'
        ))
        
        fig.update_layout(
            title='Cumulative Revenue',
            xaxis_title='Date',
            yaxis_title='Cumulative Revenue ($)',
            hovermode='x',
            height=350
        )
        
        return fig
    
    @staticmethod
    def create_top_products_chart(df, n=10):
        """Create top N products chart"""
        product_revenue = df.groupby('product')['revenue'].sum().nlargest(n).reset_index()
        
        fig = px.bar(
            product_revenue,
            x='product',
            y='revenue',
            title=f'Top {n} Products by Revenue',
            color='revenue',
            color_continuous_scale='Viridis',
            text=product_revenue['revenue'].apply(lambda x: f'${x:,.0f}')
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Revenue ($)',
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        
        fig.update_traces(textposition='outside')
        
        return fig
    
    @staticmethod
    def create_region_performance_chart(df):
        """Create region performance visualization"""
        region_stats = df.groupby('region').agg({
            'revenue': 'sum',
            'units_sold': 'sum',
            'customer_rating': 'mean'
        }).round(2).reset_index()
        
        fig = px.scatter(
            region_stats,
            x='units_sold',
            y='revenue',
            size='revenue',
            color='region',
            text='region',
            title='Regional Performance Matrix',
            hover_data=['customer_rating'],
            size_max=60
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(line=dict(width=2, color='DarkSlateGrey'))
        )
        
        fig.update_layout(
            xaxis_title='Units Sold',
            yaxis_title='Revenue ($)',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_contribution_chart(df, dimension='region'):
        """Create contribution pie chart"""
        contribution = df.groupby(dimension)['revenue'].sum().reset_index()
        
        fig = px.pie(
            contribution,
            values='revenue',
            names=dimension,
            title=f'Revenue Contribution by {dimension.title()}',
            hole=0.4,
            color_discrete_sequence=Config.CHART_COLORS
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}<br>Revenue: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    @staticmethod
    def create_anomaly_chart(df):
        """Create anomaly detection chart"""
        fig = go.Figure()
        
        normal = df[df['is_anomaly'] == False]
        fig.add_trace(go.Scatter(
            x=normal['date'],
            y=normal['revenue'],
            mode='markers',
            name='Normal',
            marker=dict(color=Config.SUCCESS_COLOR, size=8, opacity=0.6),
            hovertemplate='Date: %{x}<br>Revenue: $%{y:.2f}<br>Status: Normal<extra></extra>'
        ))
        
        anomalies = df[df['is_anomaly'] == True]
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['revenue'],
                mode='markers',
                name='Anomaly',
                marker=dict(color=Config.DANGER_COLOR, size=12, symbol='x'),
                hovertemplate='Date: %{x}<br>Revenue: $%{y:.2f}<br>⚠️ Anomaly Detected<extra></extra>'
            ))
        
        fig.update_layout(
            title='Anomaly Detection in Revenue',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='closest',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_forecast_chart(historical_df, forecast_df):
        """Create forecast visualization"""
        fig = go.Figure()
        
        # Historical data
        hist = historical_df[historical_df['is_forecast'] == False]
        fig.add_trace(go.Scatter(
            x=hist['date'],
            y=hist['revenue'],
            mode='lines',
            name='Historical',
            line=dict(color=Config.PRIMARY_COLOR, width=2)
        ))
        
        # Forecast data
        if not forecast_df.empty:
            forecast = forecast_df[forecast_df['is_forecast'] == True]
            if not forecast.empty:
                fig.add_trace(go.Scatter(
                    x=forecast['date'],
                    y=forecast['revenue'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color=Config.SUCCESS_COLOR, width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                    y=(forecast['revenue'] * 1.2).tolist() + (forecast['revenue'] * 0.8).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 204, 150, 0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    showlegend=True
                ))
        
        fig.update_layout(
            title='Revenue Forecast (30-Day)',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_moving_average_chart(df, ma_data, window):
        """Create moving average chart"""
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['revenue'],
            mode='lines',
            name='Daily Revenue',
            line=dict(color=Config.PRIMARY_COLOR, width=1.5),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'][window-1:],
            y=ma_data.dropna(),
            mode='lines',
            name=f'{window}-Day MA',
            line=dict(color=Config.SUCCESS_COLOR, width=3)
        ))
        
        fig.update_layout(
            title=f'{window}-Day Moving Average',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            height=350
        )
        
        return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """UI components and styling"""
    
    @staticmethod
    def apply_custom_css(theme='light'):
        """Apply custom CSS styling"""
        
        if theme == 'dark':
            bg_color = "#1e1e1e"
            text_color = "#ffffff"
            card_bg = "#2d2d2d"
            border_color = "#404040"
        else:
            bg_color = "#ffffff"
            text_color = "#000000"
            card_bg = "#ffffff"
            border_color = "#e9ecef"
        
        st.markdown(f"""
        <style>
            /* Global Styles */
            .stApp {{
                background-color: {bg_color};
                color: {text_color};
            }}
            
            /* Header with Gradient */
            .gradient-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            
            /* KPI Cards */
            .kpi-card {{
                background: {card_bg};
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border-left: 4px solid #FF4B4B;
                transition: all 0.3s ease;
                margin-bottom: 1rem;
                border: 1px solid {border_color};
            }}
            
            .kpi-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0,0,0,0.1);
                border-left-width: 6px;
            }}
            
            .kpi-value {{
                font-size: 2.2rem;
                font-weight: 700;
                color: #FF4B4B;
                margin: 0.5rem 0;
            }}
            
            .kpi-label {{
                font-size: 0.9rem;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 0.5rem;
            }}
            
            /* Insight Cards */
            .insight-card {{
                padding: 1.2rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 4px solid;
                transition: all 0.3s ease;
            }}
            
            .insight-success {{
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-left-color: #28a745;
            }}
            
            .insight-warning {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                border-left-color: #ffc107;
            }}
            
            .insight-info {{
                background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
                border-left-color: #17a2b8;
            }}
            
            /* Section Headers */
            .section-header {{
                padding: 1rem 0;
                margin: 2rem 0 1rem 0;
                border-bottom: 2px solid #e9ecef;
            }}
            
            .section-title {{
                font-size: 1.5rem;
                font-weight: 600;
                color: {text_color};
            }}
            
            /* Footer */
            .footer {{
                text-align: center;
                padding: 2rem;
                margin-top: 3rem;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
            }}
            
            /* Metric Changes */
            .metric-positive {{
                color: #28a745;
                font-weight: 600;
            }}
            
            .metric-negative {{
                color: #dc3545;
                font-weight: 600;
            }}
            
            /* Download Button */
            .download-btn {{
                display: inline-block;
                padding: 0.5rem 1rem;
                background-color: #FF4B4B;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-weight: 600;
                transition: all 0.3s ease;
            }}
            
            .download-btn:hover {{
                background-color: #ff3333;
                color: white;
                text-decoration: none;
            }}
            
            /* Hide Streamlit Branding */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_header():
        """Create dashboard header"""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.image(
                "https://cdn-icons-png.flaticon.com/512/924/924514.png",
                width=100
            )
        
        with col2:
            st.markdown(f"""
            <div class="gradient-header">
                <h1 style="margin:0; font-size: 2.5rem;">{Config.APP_NAME}</h1>
                <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">
                    {Config.APP_DESCRIPTION} v{Config.APP_VERSION}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.get('authenticated', False):
                user = st.session_state.get('user', {})
                st.markdown(f"""
                <div style="text-align: right; padding: 1rem;">
                    <p style="margin:0; font-weight: 600;">Welcome, {user.get('name', 'User')}</p>
                    <p style="margin:0; font-size: 0.8rem; color: #666;">{user.get('role', '').title()}</p>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_kpi_card(title, value, change=None, prefix="", suffix=""):
        """Create a KPI card"""
        change_html = ""
        if change is not None:
            change_class = "metric-positive" if change >= 0 else "metric-negative"
            change_icon = "▲" if change >= 0 else "▼"
            change_html = f"""
            <div style="margin-top: 0.5rem;">
                <span class="{change_class}">{change_icon} {abs(change):.1f}% vs last period</span>
            </div>
            """
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{prefix}{value}{suffix}</div>
            {change_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_insight_card(insight):
        """Create an insight card"""
        st.markdown(f"""
        <div class="insight-card insight-{insight['type']}">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span style="font-size: 1.8rem;">{insight['icon']}</span>
                <h4 style="margin:0; color: #000;">{insight['title']}</h4>
            </div>
            <p style="margin:0; color: #333; font-size: 1rem;">
                {insight['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_section_header(title, icon=""):
        """Create a section header"""
        icon_html = f"{icon} " if icon else ""
        st.markdown(f"""
        <div class="section-header">
            <h2 class="section-title">{icon_html}{title}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_footer():
        """Create dashboard footer"""
        st.markdown(f"""
        <div class="footer">
            <p>
                © {Config.COPYRIGHT_YEAR} {Config.APP_AUTHOR} | 
                {Config.APP_NAME} v{Config.APP_VERSION} |
                Last refresh: {st.session_state.get('last_refresh', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}
            </p>
            <p style="font-size: 0.8rem; margin-top: 0.5rem;">
                Built with Streamlit • Powered by Advanced Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_date_range_filter(min_date, max_date):
        """Create date range filter"""
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
                min_value=min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
                max_value=max_date.date() if isinstance(max_date, pd.Timestamp) else max_date,
                key="start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date.date() if isinstance(max_date, pd.Timestamp) else max_date,
                min_value=min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
                max_value=max_date.date() if isinstance(max_date, pd.Timestamp) else max_date,
                key="end_date"
            )
        
        return start_date, end_date

# ============================================================================
# UTILITIES
# ============================================================================

class Utils:
    """Utility functions"""
    
    @staticmethod
    def format_currency(value):
        """Format number as currency"""
        if value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    
    @staticmethod
    def format_number(value):
        """Format large numbers"""
        if value >= 1e6:
            return f"{value/1e6:.2f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{int(value):,}"
    
    @staticmethod
    @st.cache_data
    def convert_df_to_csv(df):
        """Convert dataframe to CSV"""
        return df.to_csv(index=False).encode('utf-8')
    
   
