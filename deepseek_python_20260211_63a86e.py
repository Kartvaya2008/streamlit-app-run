import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Tuple, Dict, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Handle data loading, generation, and caching"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def generate_sample_data() -> pd.DataFrame:
        """Generate realistic sales data"""
        np.random.seed(42)
        
        # Products and regions
        products = [
            'Chai Tea Premium', 'Masala Chai', 'Green Tea Deluxe',
            'Herbal Infusion', 'Oolong Supreme', 'Black Tea Classic',
            'Matcha Zen', 'Earl Grey', 'Jasmine Pearl', 'Darjeeling Gold'
        ]
        
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        # Generate dates
        dates = pd.date_range(
            start=Config.DEFAULT_START_DATE,
            end=Config.DEFAULT_END_DATE,
            freq='D'
        )
        
        data = []
        for date in dates[:Config.SAMPLE_SIZE]:
            for region in regions:
                for product in products:
                    # Generate realistic sales patterns
                    base_price = np.random.uniform(5, 25)
                    units = np.random.poisson(20)
                    
                    # Add seasonality and trends
                    if date.month in [10, 11, 12]:  # Holiday season
                        units *= np.random.uniform(1.2, 1.5)
                    if date.weekday() in [5, 6]:  # Weekends
                        units *= np.random.uniform(1.1, 1.3)
                    
                    revenue = base_price * units
                    
                    # Add some anomalies
                    if np.random.random() < 0.01:
                        units *= np.random.uniform(2, 5)
                    
                    data.append({
                        'date': date,
                        'region': region,
                        'product': product,
                        'units_sold': int(units),
                        'unit_price': round(base_price, 2),
                        'revenue': round(revenue, 2),
                        'category': 'Beverage',
                        'customer_rating': round(np.random.uniform(3.5, 5.0), 1)
                    })
        
        df = pd.DataFrame(data)
        
        # Add derived columns
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.day_name()
        
        logger.info(f"Generated sample data with {len(df)} records")
        return df
    
    @staticmethod
    def load_uploaded_data(uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV/Excel file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Convert date column if exists
            date_cols = ['date', 'timestamp', 'order_date', 'sale_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    break
            
            logger.info(f"Loaded data from {uploaded_file.name} with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading uploaded file: {e}")
            raise
    
    @staticmethod
    def filter_data(
        df: pd.DataFrame,
        date_range: Tuple[datetime, datetime],
        regions: list,
        products: list,
        search_term: str = ""
    ) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Date filter
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) & 
            (filtered_df['date'] <= end_date)
        ]
        
        # Region filter
        if regions:
            filtered_df = filtered_df[filtered_df['region'].isin(regions)]
        
        # Product filter
        if products:
            filtered_df = filtered_df[filtered_df['product'].isin(products)]
        
        # Search filter
        if search_term:
            filtered_df = filtered_df[
                filtered_df['product'].str.contains(search_term, case=False, na=False)
            ]
        
        return filtered_df