import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SalesAnalytics:
    """Advanced analytics and insights generation"""
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        total_revenue = df['revenue'].sum()
        total_units = df['units_sold'].sum()
        avg_order_value = total_revenue / len(df) if len(df) > 0 else 0
        
        # Growth vs previous period
        df['period'] = df['date'].dt.to_period('M')
        current_period = df[df['date'] >= (datetime.now() - timedelta(days=30))]['revenue'].sum()
        previous_period = df[
            (df['date'] >= (datetime.now() - timedelta(days=60))) &
            (df['date'] < (datetime.now() - timedelta(days=30)))
        ]['revenue'].sum()
        
        revenue_growth = ((current_period - previous_period) / previous_period * 100 
                         if previous_period > 0 else 0)
        
        # Best performing product
        product_performance = df.groupby('product')['revenue'].sum().reset_index()
        best_product = product_performance.loc[product_performance['revenue'].idxmax()]
        worst_product = product_performance.loc[product_performance['revenue'].idxmin()]
        
        return {
            'total_revenue': round(total_revenue, 2),
            'total_units': int(total_units),
            'avg_order_value': round(avg_order_value, 2),
            'revenue_growth': round(revenue_growth, 2),
            'total_orders': len(df),
            'unique_products': df['product'].nunique(),
            'best_product': best_product['product'],
            'best_product_revenue': round(best_product['revenue'], 2),
            'worst_product': worst_product['product'],
            'worst_product_revenue': round(worst_product['revenue'], 2),
        }
    
    @staticmethod
    def generate_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate automated insights from data"""
        insights = []
        
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
        
        # Best performing region
        region_performance = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
        if len(region_performance) > 0:
            best_region = region_performance.index[0]
            region_share = (region_performance.iloc[0] / region_performance.sum() * 100)
            insights.append({
                'type': 'info',
                'icon': '📍',
                'title': 'Regional Leader',
                'description': f'{best_region} contributes {region_share:.1f}% of total revenue'
            })
        
        # Product concentration
        top_products = df.groupby('product')['revenue'].sum().nlargest(3).sum()
        total_revenue = df['revenue'].sum()
        concentration = (top_products / total_revenue * 100)
        if concentration > 60:
            insights.append({
                'type': 'info',
                'icon': '🎯',
                'title': 'High Product Concentration',
                'description': f'Top 3 products account for {concentration:.1f}% of revenue'
            })
        
        # Seasonality detection
        monthly_revenue = df.groupby(df['date'].dt.month)['revenue'].sum()
        if len(monthly_revenue) >= 3:
            if monthly_revenue.idxmax() in [11, 12]:
                insights.append({
                    'type': 'success',
                    'icon': '🎄',
                    'title': 'Holiday Season Boost',
                    'description': 'Peak sales observed in holiday months'
                })
        
        return insights
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous sales patterns"""
        try:
            # Prepare features
            features = df[['units_sold', 'revenue', 'unit_price']].fillna(0)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            df['anomaly_score'] = iso_forest.fit_predict(features)
            df['is_anomaly'] = df['anomaly_score'] == -1
            
            return df
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            df['is_anomaly'] = False
            return df
    
    @staticmethod
    def forecast_revenue(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Simple revenue forecasting"""
        try:
            # Aggregate daily revenue
            daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
            daily_revenue = daily_revenue.set_index('date')
            
            # Simple moving average forecast
            forecast = pd.date_range(
                start=daily_revenue.index[-1] + timedelta(days=1),
                periods=days,
                freq='D'
            )
            
            # Use last 30 days average for simplicity
            last_30_avg = daily_revenue['revenue'].tail(30).mean()
            forecast_df = pd.DataFrame({
                'date': forecast,
                'revenue': last_30_avg,
                'is_forecast': True
            })
            
            # Prepare historical data
            historical_df = pd.DataFrame({
                'date': daily_revenue.index,
                'revenue': daily_revenue['revenue'],
                'is_forecast': False
            })
            
            return pd.concat([historical_df, forecast_df], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_moving_average(df: pd.DataFrame, window: int = 7) -> pd.Series:
        """Calculate moving average"""
        return df.groupby('date')['revenue'].sum().rolling(window=window).mean()