"""
Analytics package for Educational Analytics Dashboard

This package provides comprehensive analytics capabilities including:
- Risk prediction using machine learning models
- Performance trend analysis and insights generation
- Engagement scoring algorithms
- Statistical analysis and pattern recognition
- Predictive modeling for student outcomes
"""

from .engine import AnalyticsEngine, RiskPredictor, PerformanceAnalyzer, AnalyticsError

__all__ = ['AnalyticsEngine', 'RiskPredictor', 'PerformanceAnalyzer', 'AnalyticsError']