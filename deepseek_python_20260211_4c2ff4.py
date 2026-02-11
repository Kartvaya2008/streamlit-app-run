import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from config import Config

class ChartBuilder:
    """Create interactive Plotly charts"""
    
    @staticmethod
    def create_revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
        """Create revenue over time chart"""
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        
        fig = go.Figure()
        
        # Line chart
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color=Config.PRIMARY_COLOR, width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.1)'
        ))
        
        # Moving average
        moving_avg = daily_revenue['revenue'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=moving_avg,
            mode='lines',
            name='7-day MA',
            line=dict(color=Config.SUCCESS_COLOR, width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Revenue Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_product_performance_chart(df: pd.DataFrame) -> go.Figure:
        """Create product revenue bar chart"""
        product_revenue = df.groupby('product')['revenue'].sum().reset_index()
        product_revenue = product_revenue.sort_values('revenue', ascending=True)
        
        fig = px.bar(
            product_revenue,
            y='product',
            x='revenue',
            orientation='h',
            color='revenue',
            color_continuous_scale='Viridis',
            title='Product Revenue Contribution'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Revenue ($)',
            yaxis_title='Product',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_region_heatmap(df: pd.DataFrame) -> go.Figure:
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
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Region vs Product Revenue Heatmap',
            xaxis_title='Product',
            yaxis_title='Region',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_distribution_chart(df: pd.DataFrame) -> go.Figure:
        """Create units sold distribution chart"""
        fig = px.histogram(
            df,
            x='units_sold',
            nbins=50,
            title='Units Sold Distribution',
            color_discrete_sequence=[Config.SECONDARY_COLOR]
        )
        
        fig.update_layout(
            xaxis_title='Units Sold',
            yaxis_title='Count',
            height=350
        )
        
        return fig
    
    @staticmethod
    def create_cumulative_revenue_chart(df: pd.DataFrame) -> go.Figure:
        """Create cumulative revenue chart"""
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        daily_revenue['cumulative_revenue'] = daily_revenue['revenue'].cumsum()
        daily_revenue = daily_revenue.sort_values('date')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['cumulative_revenue'],
            mode='lines',
            name='Cumulative Revenue',
            line=dict(color=Config.SUCCESS_COLOR, width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Cumulative Revenue Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Revenue ($)',
            height=350
        )
        
        return fig
    
    @staticmethod
    def create_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
        """Create forecast visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['revenue'],
            mode='lines',
            name='Historical',
            line=dict(color=Config.PRIMARY_COLOR, width=2)
        ))
        
        # Forecast data
        if not forecast_df.empty:
            forecast_actual = forecast_df[forecast_df['is_forecast'] == False]
            forecast_future = forecast_df[forecast_df['is_forecast'] == True]
            
            if not forecast_actual.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_actual['date'],
                    y=forecast_actual['revenue'],
                    mode='lines',
                    name='Actual',
                    line=dict(color=Config.PRIMARY_COLOR, width=2)
                ))
            
            if not forecast_future.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_future['date'],
                    y=forecast_future['revenue'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color=Config.SUCCESS_COLOR, width=2, dash='dash')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_future['date'],
                    y=forecast_future['revenue'] * 1.2,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_future['date'],
                    y=forecast_future['revenue'] * 0.8,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 204, 150, 0.2)',
                    name='Confidence Interval'
                ))
        
        fig.update_layout(
            title='Revenue Forecast (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            height=400
        )
        
        return fig