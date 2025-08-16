"""
Educational Analytics Dashboard - Main Application

This is the main Streamlit application file for the Educational Analytics Dashboard.
It uses a modular architecture with separation of concerns for better maintainability
and professional code organization.

Features:
- Predictive risk analysis using machine learning
- Performance trend analysis and insights
- Real-time student engagement tracking
- Comprehensive analytics dashboard
- Professional data visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import modular components
from src.config import config
from src.database import DatabaseManager, DatabaseError
from src.analytics import AnalyticsEngine, AnalyticsError
from src.visualization import DashboardComponents, ChartFactory
from src.utils import (
    DataValidator, DateTimeHelper, SecurityHelper, 
    FormatHelper, FileHelper
)

# Configure logging
logger = logging.getLogger(__name__)

class EducationalAnalyticsDashboard:
    """
    Main dashboard application class
    
    This class orchestrates all components of the educational analytics
    dashboard and provides the main user interface.
    """
    
    def __init__(self):
        """Initialize the dashboard application"""
        # Configure Streamlit page
        st.set_page_config(
            page_title=config.ui.page_title,
            page_icon=config.ui.page_icon,
            layout=config.ui.layout,
            initial_sidebar_state=config.ui.sidebar_state
        )
        
        # Initialize core components
        try:
            self.db_manager = DatabaseManager()
            self.analytics_engine = AnalyticsEngine(self.db_manager)
            self.dashboard_components = DashboardComponents()
            
            # Validate configuration
            if not config.validate_config():
                st.error("Configuration validation failed. Please check your settings.")
                st.stop()
                
            logger.info("Educational Analytics Dashboard initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            st.error(f"Application initialization failed: {e}")
            st.stop()
    
    def render_header(self):
        """Render the main dashboard header"""
        st.title("🎓 Educational Analytics Dashboard")
        st.markdown("*Transform student data into actionable insights with AI-powered analytics*")
        
        # Show configuration info in debug mode
        if config.debug_mode:
            with st.expander("🔧 Configuration Info", expanded=False):
                st.json(config.to_dict())
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        with st.sidebar:
            st.header("📊 Analytics Controls")
            
            # Date range selector
            date_range = st.selectbox(
                "Analysis Period",
                options=["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                index=1
            )
            
            # Convert to days
            if date_range == "Last 7 days":
                days_back = 7
            elif date_range == "Last 30 days":
                days_back = 30
            elif date_range == "Last 90 days":
                days_back = 90
            else:
                days_back = None
            
            st.session_state['analysis_days'] = days_back
            
            # Student filter
            st.subheader("🎯 Student Filter")
            selected_student = st.selectbox(
                "Select Student",
                options=["All Students"] + self._get_student_list(),
                index=0
            )
            
            st.session_state['selected_student'] = None if selected_student == "All Students" else selected_student
            
            # Analytics actions
            st.subheader("🤖 AI Actions")
            
            if st.button("🔄 Retrain Models", type="secondary"):
                self._retrain_models()
            
            if st.button("📊 Generate Report", type="secondary"):
                self._generate_report()
            
            # System info
            st.subheader("ℹ️ System Info")
            st.info(f"Environment: {'Production' if config.is_production else 'Development'}")
            st.info(f"Debug Mode: {'Enabled' if config.debug_mode else 'Disabled'}")
    
    def render_main_dashboard(self):
        """Render the main dashboard content"""
        try:
            # Get data based on filters
            days_back = st.session_state.get('analysis_days')
            selected_student = st.session_state.get('selected_student')
            
            if days_back:
                start_date, end_date = DateTimeHelper.get_date_range(days_back)
                sessions_df = self.db_manager.get_student_sessions(
                    user_id=selected_student,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                sessions_df = self.db_manager.get_student_sessions(user_id=selected_student)
            
            # Get analytics summary
            analytics_summary = self.db_manager.get_analytics_summary()
            
            # Render performance overview
            self.dashboard_components.render_performance_overview(analytics_summary)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Trends", "⚠️ Risk Analysis", "👥 Student Details", "💡 Insights"])
            
            with tab1:
                self._render_performance_tab(sessions_df)
            
            with tab2:
                self._render_risk_analysis_tab(sessions_df)
            
            with tab3:
                self._render_student_details_tab(sessions_df)
            
            with tab4:
                self._render_insights_tab(sessions_df)
                
        except Exception as e:
            logger.error(f"Error rendering main dashboard: {e}")
            st.error(f"Dashboard rendering failed: {e}")
    
    def _render_performance_tab(self, sessions_df: pd.DataFrame):
        """Render the performance trends tab"""
        st.subheader("📈 Performance Trends Analysis")
        
        if sessions_df.empty:
            st.info("No performance data available for the selected period.")
            return
        
        # Create performance trends chart
        trends_chart = self.dashboard_components.render_performance_trends_chart(sessions_df)
        st.plotly_chart(trends_chart, use_container_width=True)
        
        # Performance statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Statistics")
            
            total_questions = sessions_df['questions_attempted'].sum()
            total_correct = sessions_df['correct_answers'].sum()
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            avg_engagement = sessions_df['engagement_score'].mean()
            
            st.metric("Overall Accuracy", FormatHelper.format_percentage(overall_accuracy))
            st.metric("Average Engagement", f"{avg_engagement:.2f}/1.0")
            st.metric("Total Questions", FormatHelper.format_large_number(total_questions))
            st.metric("Active Students", sessions_df['user_id'].nunique())
        
        with col2:
            st.subheader("🎯 Performance Distribution")
            
            # Create accuracy distribution chart
            sessions_df['accuracy'] = sessions_df['correct_answers'] / sessions_df['questions_attempted'].replace(0, np.nan)
            accuracy_bins = pd.cut(sessions_df['accuracy'].dropna(), bins=[0, 0.5, 0.7, 0.9, 1.0], labels=['Low', 'Medium', 'High', 'Excellent'])
            
            distribution_data = accuracy_bins.value_counts()
            if not distribution_data.empty:
                st.bar_chart(distribution_data)
            else:
                st.info("No accuracy data available")
    
    def _render_risk_analysis_tab(self, sessions_df: pd.DataFrame):
        """Render the risk analysis tab"""
        st.subheader("⚠️ Student Risk Analysis")
        
        if sessions_df.empty:
            st.info("No data available for risk analysis.")
            return
        
        try:
            # Perform risk analysis
            with st.spinner("Analyzing student risk factors..."):
                risk_analysis = self.analytics_engine.analyze_student_risk()
            
            if risk_analysis['status'] == 'success':
                # Display risk overview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Students", 
                        risk_analysis['total_students']
                    )
                
                with col2:
                    st.metric(
                        "High Risk Students", 
                        risk_analysis['high_risk_count'],
                        delta=f"{risk_analysis['risk_percentage']:.1f}%"
                    )
                
                with col3:
                    low_risk_count = risk_analysis['total_students'] - risk_analysis['high_risk_count']
                    st.metric("Low Risk Students", low_risk_count)
                
                # Risk analysis chart
                risk_chart = self.dashboard_components.render_risk_analysis_chart(risk_analysis['predictions'])
                st.plotly_chart(risk_chart, use_container_width=True)
                
                # Detailed risk predictions table
                if risk_analysis['predictions']:
                    st.subheader("📋 Detailed Risk Predictions")
                    risk_df = pd.DataFrame(risk_analysis['predictions'])
                    
                    # Format for display
                    display_df = risk_df.copy()
                    display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.2f}")
                    display_df = display_df.rename(columns={
                        'user_id': 'Student ID',
                        'risk_score': 'Risk Score',
                        'risk_level': 'Risk Level'
                    })
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            else:
                st.warning(f"Risk analysis failed: {risk_analysis['message']}")
                
        except AnalyticsError as e:
            logger.error(f"Risk analysis error: {e}")
            st.error(f"Risk analysis failed: {e}")
    
    def _render_student_details_tab(self, sessions_df: pd.DataFrame):
        """Render the student details tab"""
        st.subheader("👥 Student Details")
        
        if sessions_df.empty:
            st.info("No student data available.")
            return
        
        # Get risk data for the table
        try:
            risk_analysis = self.analytics_engine.analyze_student_risk()
            risk_data = risk_analysis.get('predictions', []) if risk_analysis['status'] == 'success' else []
        except:
            risk_data = []
        
        # Render student details table
        self.dashboard_components.render_student_details_table(sessions_df, risk_data)
        
        # Individual student analysis
        st.subheader("🔍 Individual Student Analysis")
        
        student_list = sessions_df['user_id'].unique().tolist()
        if student_list:
            selected_student = st.selectbox(
                "Select student for detailed analysis:",
                options=student_list,
                key="detailed_student_select"
            )
            
            if selected_student:
                self._render_individual_student_analysis(selected_student, sessions_df)
    
    def _render_individual_student_analysis(self, user_id: str, sessions_df: pd.DataFrame):
        """Render detailed analysis for individual student"""
        student_sessions = sessions_df[sessions_df['user_id'] == user_id]
        
        if student_sessions.empty:
            st.warning(f"No data available for student {user_id}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 Performance Summary - {SecurityHelper.anonymize_user_id(user_id)}")
            
            total_sessions = len(student_sessions)
            total_questions = student_sessions['questions_attempted'].sum()
            total_correct = student_sessions['correct_answers'].sum()
            accuracy = total_correct / total_questions if total_questions > 0 else 0
            avg_engagement = student_sessions['engagement_score'].mean()
            
            st.metric("Total Sessions", total_sessions)
            st.metric("Accuracy Rate", FormatHelper.format_percentage(accuracy))
            st.metric("Average Engagement", f"{avg_engagement:.2f}/1.0")
            st.metric("Questions Attempted", total_questions)
        
        with col2:
            st.subheader("📈 Progress Over Time")
            
            # Create individual progress chart
            student_sessions['session_date'] = pd.to_datetime(student_sessions['start_time']).dt.date
            daily_progress = student_sessions.groupby('session_date').agg({
                'correct_answers': 'sum',
                'questions_attempted': 'sum'
            }).reset_index()
            
            daily_progress['accuracy'] = daily_progress['correct_answers'] / daily_progress['questions_attempted']
            
            if len(daily_progress) > 1:
                st.line_chart(daily_progress.set_index('session_date')['accuracy'])
            else:
                st.info("Need more session data to show progress trend")
    
    def _render_insights_tab(self, sessions_df: pd.DataFrame):
        """Render the insights and recommendations tab"""
        st.subheader("💡 AI-Powered Insights")
        
        if sessions_df.empty:
            st.info("No data available for insights generation.")
            return
        
        try:
            # Generate insights
            with st.spinner("Generating AI insights..."):
                insights_result = self.analytics_engine.generate_performance_insights()
            
            if insights_result['status'] == 'success':
                # Render insights panel
                self.dashboard_components.render_insights_panel(insights_result['insights'])
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                summary = insights_result['summary']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sessions", summary['total_sessions'])
                with col2:
                    st.metric("Average Accuracy", FormatHelper.format_percentage(summary['avg_accuracy']))
                with col3:
                    st.metric("Average Engagement", f"{summary['avg_engagement']:.2f}/1.0")
            
            else:
                st.warning(f"Insights generation failed: {insights_result['message']}")
                
        except AnalyticsError as e:
            logger.error(f"Insights generation error: {e}")
            st.error(f"Failed to generate insights: {e}")
    
    def _get_student_list(self) -> List[str]:
        """Get list of available students"""
        try:
            sessions_df = self.db_manager.get_student_sessions()
            return sessions_df['user_id'].unique().tolist() if not sessions_df.empty else []
        except:
            return []
    
    def _retrain_models(self):
        """Retrain machine learning models"""
        try:
            with st.spinner("Retraining AI models..."):
                result = self.analytics_engine.train_models()
            
            if result['status'] == 'success':
                st.success(f"Models retrained successfully! Accuracy: {result['training_metrics'].get('accuracy', 0):.3f}")
            else:
                st.warning(f"Model training failed: {result['message']}")
                
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            st.error(f"Failed to retrain models: {e}")
    
    def _generate_report(self):
        """Generate and download analytics report"""
        try:
            # This would generate a comprehensive report
            # For now, just show a success message
            st.success("Report generation functionality will be implemented in a future version.")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            st.error(f"Failed to generate report: {e}")
    
    def run(self):
        """Run the main dashboard application"""
        try:
            # Initialize session state
            if 'analysis_days' not in st.session_state:
                st.session_state['analysis_days'] = 30
            if 'selected_student' not in st.session_state:
                st.session_state['selected_student'] = None
            
            # Render application components
            self.render_header()
            self.render_sidebar()
            self.render_main_dashboard()
            
            # Footer
            st.markdown("---")
            st.markdown(
                "*Educational Analytics Dashboard - Powered by AI and Machine Learning*  \n"
                f"*Version: {__import__('src').__version__} | "
                f"Environment: {'Production' if config.is_production else 'Development'}*"
            )
            
        except Exception as e:
            logger.error(f"Application runtime error: {e}")
            st.error(f"Application error: {e}")

def main():
    """Main application entry point"""
    try:
        # Create and run dashboard
        dashboard = EducationalAnalyticsDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Critical application error: {e}")
        st.error("Critical application error. Please check logs and restart.")

if __name__ == "__main__":
    main()