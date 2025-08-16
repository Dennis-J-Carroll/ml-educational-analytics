"""
Visualization Components for Educational Analytics Dashboard

This module provides reusable visualization components for displaying
analytics data in an intuitive and professional manner.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from ..config.settings import config

logger = logging.getLogger(__name__)

class DashboardComponents:
    """
    Reusable dashboard components for educational analytics
    
    This class provides standardized visualization components that
    maintain consistent styling and professional appearance.
    """
    
    def __init__(self):
        """Initialize dashboard components with theme configuration"""
        self.theme_colors = {
            'primary': config.ui.theme_color,
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        logger.debug("DashboardComponents initialized")
    
    def render_metric_card(self, title: str, value: Any, delta: Optional[float] = None,
                          delta_color: str = "normal", help_text: Optional[str] = None):
        """
        Render a professional metric card with optional delta indicator
        
        Args:
            title: Card title
            value: Main metric value
            delta: Change indicator (optional)
            delta_color: Color for delta ("normal", "inverse", or "off")
            help_text: Tooltip help text (optional)
        """
        try:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )
            
        except Exception as e:
            logger.error(f"Error rendering metric card '{title}': {e}")
            st.error(f"Error displaying metric: {title}")
    
    def render_performance_overview(self, analytics_summary: Dict[str, Any]):
        """
        Render the main performance overview dashboard
        
        Args:
            analytics_summary: Dictionary containing summary metrics
        """
        try:
            st.subheader("📊 Performance Overview")
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self.render_metric_card(
                    "Total Students",
                    analytics_summary.get('total_students', 0),
                    help_text="Total number of students tracked in the system"
                )
            
            with col2:
                at_risk = analytics_summary.get('at_risk_students', 0)
                total = analytics_summary.get('total_students', 1)
                risk_percentage = (at_risk / total * 100) if total > 0 else 0
                
                self.render_metric_card(
                    "At-Risk Students",
                    at_risk,
                    delta=f"{risk_percentage:.1f}%",
                    delta_color="inverse",
                    help_text="Students identified as at risk of poor performance"
                )
            
            with col3:
                avg_performance = analytics_summary.get('average_performance', 0)
                self.render_metric_card(
                    "Average Performance",
                    f"{avg_performance:.1%}",
                    help_text="Overall accuracy rate across all students"
                )
            
            with col4:
                active_today = analytics_summary.get('active_today', 0)
                self.render_metric_card(
                    "Active Today",
                    active_today,
                    help_text="Students who had sessions today"
                )
                
        except Exception as e:
            logger.error(f"Error rendering performance overview: {e}")
            st.error("Error displaying performance overview")
    
    def render_performance_trends_chart(self, sessions_df: pd.DataFrame) -> go.Figure:
        """
        Create performance trends visualization
        
        Args:
            sessions_df: DataFrame containing session data
            
        Returns:
            Plotly figure object
        """
        try:
            if sessions_df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for trend analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Process data for trends
            sessions_df['date'] = pd.to_datetime(sessions_df['start_time']).dt.date
            daily_trends = sessions_df.groupby('date').agg({
                'correct_answers': 'sum',
                'questions_attempted': 'sum',
                'engagement_score': 'mean',
                'user_id': 'nunique'
            }).reset_index()
            
            daily_trends['accuracy'] = (
                daily_trends['correct_answers'] / 
                daily_trends['questions_attempted'].replace(0, np.nan)
            ).fillna(0)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Accuracy Trend', 'Engagement Trend', 
                              'Daily Active Students', 'Questions Attempted'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Accuracy trend
            fig.add_trace(
                go.Scatter(
                    x=daily_trends['date'],
                    y=daily_trends['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color=self.theme_colors['primary'])
                ),
                row=1, col=1
            )
            
            # Engagement trend
            fig.add_trace(
                go.Scatter(
                    x=daily_trends['date'],
                    y=daily_trends['engagement_score'],
                    mode='lines+markers',
                    name='Engagement',
                    line=dict(color=self.theme_colors['success'])
                ),
                row=1, col=2
            )
            
            # Active students
            fig.add_trace(
                go.Bar(
                    x=daily_trends['date'],
                    y=daily_trends['user_id'],
                    name='Active Students',
                    marker_color=self.theme_colors['info']
                ),
                row=2, col=1
            )
            
            # Questions attempted
            fig.add_trace(
                go.Bar(
                    x=daily_trends['date'],
                    y=daily_trends['questions_attempted'],
                    name='Questions',
                    marker_color=self.theme_colors['warning']
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Performance Trends Analysis"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance trends chart: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def render_risk_analysis_chart(self, risk_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create risk analysis visualization
        
        Args:
            risk_data: List of risk prediction data
            
        Returns:
            Plotly figure object
        """
        try:
            if not risk_data:
                fig = go.Figure()
                fig.add_annotation(
                    text="No risk data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            risk_df = pd.DataFrame(risk_data)
            
            # Create risk distribution pie chart
            risk_counts = risk_df['risk_level'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker_colors=[
                        self.theme_colors['success'] if level == 'low' else self.theme_colors['danger']
                        for level in risk_counts.index
                    ]
                )
            ])
            
            fig.update_layout(
                title="Student Risk Distribution",
                annotations=[dict(text='Risk<br>Level', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk analysis chart: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def render_student_details_table(self, sessions_df: pd.DataFrame, risk_data: List[Dict[str, Any]]):
        """
        Render detailed student information table
        
        Args:
            sessions_df: DataFrame containing session data
            risk_data: List of risk prediction data
        """
        try:
            if sessions_df.empty:
                st.info("No student data available")
                return
            
            # Aggregate student data
            student_summary = sessions_df.groupby('user_id').agg({
                'questions_attempted': 'sum',
                'correct_answers': 'sum',
                'engagement_score': 'mean',
                'session_id': 'count',
                'start_time': 'max'
            }).reset_index()
            
            student_summary['accuracy'] = (
                student_summary['correct_answers'] / 
                student_summary['questions_attempted'].replace(0, np.nan)
            ).fillna(0)
            
            # Merge with risk data
            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                student_summary = student_summary.merge(
                    risk_df[['user_id', 'risk_level', 'risk_score']], 
                    on='user_id', 
                    how='left'
                )
                student_summary['risk_level'] = student_summary['risk_level'].fillna('unknown')
                student_summary['risk_score'] = student_summary['risk_score'].fillna(0)
            
            # Format for display
            display_df = student_summary.copy()
            display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.1%}")
            display_df['engagement_score'] = display_df['engagement_score'].apply(lambda x: f"{x:.2f}")
            display_df['last_activity'] = pd.to_datetime(display_df['start_time']).dt.strftime('%Y-%m-%d')
            
            # Select and rename columns for display
            columns_to_show = {
                'user_id': 'Student ID',
                'session_id': 'Sessions',
                'questions_attempted': 'Questions',
                'accuracy': 'Accuracy',
                'engagement_score': 'Engagement',
                'last_activity': 'Last Activity'
            }
            
            if 'risk_level' in display_df.columns:
                columns_to_show['risk_level'] = 'Risk Level'
            
            display_df = display_df[list(columns_to_show.keys())].rename(columns=columns_to_show)
            
            # Display with styling
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            logger.error(f"Error rendering student details table: {e}")
            st.error("Error displaying student details")
    
    def render_insights_panel(self, insights: List[Dict[str, Any]]):
        """
        Render insights and recommendations panel
        
        Args:
            insights: List of insight dictionaries
        """
        try:
            st.subheader("💡 Key Insights")
            
            if not insights:
                st.info("No insights available. Collect more data to generate insights.")
                return
            
            for insight in insights:
                insight_type = insight.get('type', 'general')
                description = insight.get('description', 'No description available')
                value = insight.get('value', '')
                
                # Choose appropriate icon and color based on insight type
                if insight_type == 'trend_analysis':
                    icon = "📈" if 'improving' in str(value) else "📉" if 'declining' in str(value) else "📊"
                    color = "success" if 'improving' in str(value) else "warning" if 'declining' in str(value) else "info"
                elif insight_type == 'overall_performance':
                    icon = "🎯"
                    color = "info"
                elif insight_type == 'activity_pattern':
                    icon = "⏰"
                    color = "secondary"
                else:
                    icon = "💡"
                    color = "primary"
                
                # Display insight with appropriate styling
                if color == "warning":
                    st.warning(f"{icon} {description}")
                elif color == "success":
                    st.success(f"{icon} {description}")
                else:
                    st.info(f"{icon} {description}")
                    
        except Exception as e:
            logger.error(f"Error rendering insights panel: {e}")
            st.error("Error displaying insights")

class ChartFactory:
    """
    Factory class for creating standardized charts with consistent styling
    """
    
    @staticmethod
    def create_empty_chart(message: str = "No data available") -> go.Figure:
        """
        Create an empty chart with a message
        
        Args:
            message: Message to display
            
        Returns:
            Empty Plotly figure with message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        return fig
    
    @staticmethod
    def apply_theme(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
        """
        Apply consistent theme styling to a Plotly figure
        
        Args:
            fig: Plotly figure to style
            title: Optional chart title
            
        Returns:
            Styled figure
        """
        fig.update_layout(
            title=title,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
        fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
        
        return fig