"""
Educational Analytics Dashboard - Modular Architecture

This package provides a comprehensive analytics platform for educational institutions
to track student performance, predict at-risk students, and optimize learning outcomes.

Modules:
    - config: Configuration management and environment variables
    - database: Database operations and data management
    - analytics: Core analytics engine and ML models
    - visualization: Data visualization and dashboard components
    - utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Dennis J. Carroll"
__email__ = "your-email@domain.com"

# Import main classes for easy access
from .analytics.engine import AnalyticsEngine
from .database.manager import DatabaseManager
from .config.settings import Config

__all__ = ['AnalyticsEngine', 'DatabaseManager', 'Config']