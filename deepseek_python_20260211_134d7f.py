import streamlit as st
from typing import Dict, Any, List
import base64
from datetime import datetime
from config import Config

class UIComponents:
    """UI components and styling"""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling"""
        st.markdown("""
            <style>
            /* Main container */
            .main {
                padding: 0rem 1rem;
            }
            
            /* Header styling */
            .header-container {
                background: linear-gradient(135deg, %s, %s);
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                color: white;
            }
            
            /* KPI cards */
            .kpi-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 4px solid %s;
                transition: transform 0.3s ease;
            }
            
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            
            .kpi-value {
                font-size: 2rem;
                font-weight: bold;
                color: %s;
            }
            
            .kpi-label {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Insight cards */
            .insight-success {
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
                border-left: 4px solid #28a745;
            }
            
            .insight-warning {
                background: linear-gradient(135deg, #fff3cd, #ffeaa7);
                border-left: 4px solid #ffc107;
            }
            
            .insight-info {
                background: linear-gradient(135deg, #d1ecf1, #bee5eb);
                border-left: 4px solid #17a2b8;
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 1rem;
                margin-top: 3rem;
                color: #666;
                border-top: 1px solid #eee;
            }
            
            /* Section headers */
            .section-header {
                padding: 1rem 0;
                margin: 2rem 0 1rem 0;
                border-bottom: 2px solid #eee;
            }
            
            /* Metric comparison */
            .metric-change-positive {
                color: #28a745;
                font-weight: bold;
            }
            
            .metric-change-negative {
                color: #dc3545;
                font-weight: bold;
            }
            
            /* Button styling */
            .stButton > button {
                border-radius: 5px;
                font-weight: bold;
            }
            
            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
        """ % (
            Config.PRIMARY_COLOR,
            Config.SECONDARY_COLOR,
            Config.PRIMARY_COLOR,
            Config.PRIMARY_COLOR
        ), unsafe_allow_html=True)
    
    @staticmethod
    def create_header():
        """Create dashboard header"""
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.image(
                "https://cdn-icons-png.flaticon.com/512/924/924514.png",
                width=80
            )
        
        with col2:
            st.markdown(f"""
                <div class="header-container">
                    <h1 style="margin:0; font-size: 2.5rem;">{Config.APP_NAME}</h1>
                    <p style="margin:0; font-size: 1.1rem; opacity: 0.9;">
                        Advanced Sales Analytics Dashboard v{Config.APP_VERSION}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def create_kpi_card(title: str, value: Any, change: float = None, 
                       prefix: str = "", suffix: str = "") -> None:
        """Create a KPI card widget"""
        change_html = ""
        if change is not None:
            change_class = "metric-change-positive" if change >= 0 else "metric-change-negative"
            change_icon = "📈" if change >= 0 else "📉"
            change_html = f"""
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                    <span class="{change_class}">{change_icon} {abs(change):.1f}%</span>
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
    def create_insight_card(insight: Dict[str, Any]) -> None:
        """Create an insight card"""
        st.markdown(f"""
            <div class="kpi-card insight-{insight['type']}">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 1.5rem;">{insight['icon']}</span>
                    <h4 style="margin:0;">{insight['title']}</h4>
                </div>
                <p style="margin:0; color: #333;">{insight['description']}</p>
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
                    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_section_header(title: str, icon: str = ""):
        """Create a section header"""
        icon_html = f"{icon} " if icon else ""
        st.markdown(f"""
            <div class="section-header">
                <h2>{icon_html}{title}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_download_button(df: pd.DataFrame, filename: str = "data"):
        """Create download button for dataframe"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)