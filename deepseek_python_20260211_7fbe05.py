import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple
import logging
import json
from datetime import datetime
import io
import base64

logger = logging.getLogger(__name__)

class Utils:
    """Utility functions"""
    
    @staticmethod
    def setup_logging():
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log'),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def format_currency(value: float) -> str:
        """Format number as currency"""
        if abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    
    @staticmethod
    def format_number(value: float) -> str:
        """Format large numbers"""
        if abs(value) >= 1e6:
            return f"{value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"
    
    @staticmethod
    def create_download_link(data, filename: str, filetype: str = "csv"):
        """Create download link for various file types"""
        if filetype == "csv":
            data = data.to_csv(index=False)
            mime_type = "text/csv"
        elif filetype == "json":
            data = json.dumps(data, indent=2)
            mime_type = "application/json"
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")
        
        b64 = base64.b64encode(data.encode()).decode()
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{filetype}">📥 Download {filetype.upper()}</a>'
    
    @staticmethod
    def get_date_range_filter() -> Tuple[datetime, datetime]:
        """Create date range picker"""
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=Config.DEFAULT_START_DATE,
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=Config.DEFAULT_END_DATE,
                key="end_date"
            )
        return start_date, end_date
    
    @staticmethod
    @st.cache_data
    def convert_df_to_csv(df: pd.DataFrame) -> bytes:
        """Convert dataframe to CSV bytes"""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def create_metric_comparison(current: float, previous: float, label: str) -> Dict[str, Any]:
        """Create metric comparison data"""
        if previous == 0:
            return {
                'label': label,
                'current': current,
                'previous': previous,
                'change': 0,
                'change_pct': 0,
                'trend': 'neutral'
            }
        
        change = current - previous
        change_pct = (change / previous) * 100
        
        return {
            'label': label,
            'current': current,
            'previous': previous,
            'change': change,
            'change_pct': change_pct,
            'trend': 'up' if change_pct > 0 else 'down' if change_pct < 0 else 'neutral'
        }