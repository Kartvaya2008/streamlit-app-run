"""
UI Components and Styling Module
Author: Data Science Team
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Any, Dict, List
import base64

class UIComponents:
    """Advanced UI components and styling"""
    
    @staticmethod
    def apply_custom_css(theme='light'):
        """Apply custom CSS with theme support"""
        
        if theme == 'dark':
            bg_color = "#1e1e1e"
            text_color = "#ffffff"
            card_bg = "#2d2d2d"
            border_color = "#404040"
        else:
            bg_color = "#ffffff"
            text_color = "#000000"
            card_bg = "#f8f9fa"
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
                animation: slideIn 0.5s ease-out;
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
            .section-header {
