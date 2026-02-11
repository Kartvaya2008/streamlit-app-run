import os
from datetime import datetime

class Config:
    # App Configuration
    APP_NAME = "Sales Analytics Pro"
    APP_VERSION = "2.0.0"
    APP_AUTHOR = "Data Science Team"
    COPYRIGHT_YEAR = datetime.now().year
    
    # Colors
    PRIMARY_COLOR = "#FF4B4B"
    SECONDARY_COLOR = "#0068C9"
    SUCCESS_COLOR = "#00CC96"
    WARNING_COLOR = "#FFB800"
    
    # Chart Colors
    CHART_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    
    # Date Configuration
    DEFAULT_START_DATE = datetime(2023, 1, 1)
    DEFAULT_END_DATE = datetime(2024, 1, 1)
    
    # Data Settings
    SAMPLE_SIZE = 1000
    CACHE_TTL = 3600  # seconds
    
    # File Paths
    LOG_FILE = "logs/app.log"
    
    @staticmethod
    def setup_directories():
        """Create necessary directories"""
        os.makedirs("logs", exist_ok=True)
        os.makedirs("exports", exist_ok=True)