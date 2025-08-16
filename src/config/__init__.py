"""
Configuration package for Educational Analytics Dashboard

This package provides comprehensive configuration management including:
- Environment variable handling
- Database configuration
- Analytics model settings
- UI customization options
- Security and privacy settings
"""

from .settings import Config, DatabaseConfig, AnalyticsConfig, UIConfig, SecurityConfig, config

__all__ = ['Config', 'DatabaseConfig', 'AnalyticsConfig', 'UIConfig', 'SecurityConfig', 'config']