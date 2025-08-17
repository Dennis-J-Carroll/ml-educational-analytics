"""
Configuration Management for Educational Analytics Dashboard

This module handles all configuration settings, environment variables,
and application constants for the educational analytics platform.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    path: str = "learning_analytics.db"
    timeout: int = 30
    check_same_thread: bool = False
    
    @property
    def connection_string(self) -> str:
        """Generate SQLite connection string"""
        return f"sqlite:///{self.path}"

@dataclass
class AnalyticsConfig:
    """Analytics and ML model configuration"""
    risk_prediction_threshold: float = 0.65
    engagement_score_weights: Dict[str, float] = None
    model_retrain_interval_days: int = 7
    min_data_points_for_prediction: int = 5
    
    def __post_init__(self):
        if self.engagement_score_weights is None:
            self.engagement_score_weights = {
                'questions_attempted': 0.3,
                'correct_answers': 0.4,
                'time_spent': 0.2,
                'session_frequency': 0.1
            }

@dataclass
class UIConfig:
    """User interface configuration"""
    page_title: str = "Educational Analytics Dashboard"
    page_icon: str = "🎓"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    theme_color: str = "#1f77b4"
    
@dataclass
class SecurityConfig:
    """Security and privacy settings"""
    enable_data_anonymization: bool = True
    session_timeout_minutes: int = 30
    max_file_upload_size_mb: int = 10
    allowed_file_types: list = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['csv', 'xlsx', 'json']

class Config:
    """
    Main configuration class that consolidates all settings
    
    This class provides a centralized way to access all configuration
    settings while supporting environment variable overrides.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration with optional environment file
        
        Args:
            env_file: Path to .env file for environment variables
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.env_file = env_file or self.project_root / ".env"
        
        # Load environment variables if file exists
        if self.env_file.exists():
            self._load_env_file()
        
        # Initialize configuration sections
        self.database = DatabaseConfig(
            path=os.getenv('DB_PATH', 'learning_analytics.db'),
            timeout=int(os.getenv('DB_TIMEOUT', '30'))
        )
        
        self.analytics = AnalyticsConfig(
            risk_prediction_threshold=float(os.getenv('RISK_THRESHOLD', '0.65')),
            model_retrain_interval_days=int(os.getenv('MODEL_RETRAIN_DAYS', '7'))
        )
        
        self.ui = UIConfig(
            page_title=os.getenv('PAGE_TITLE', 'Educational Analytics Dashboard'),
            layout=os.getenv('UI_LAYOUT', 'wide')
        )
        
        self.security = SecurityConfig(
            enable_data_anonymization=os.getenv('ANONYMIZE_DATA', 'true').lower() == 'true',
            session_timeout_minutes=int(os.getenv('SESSION_TIMEOUT', '30'))
        )
        
        # Setup logging
        self.setup_logging()
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"Could not load .env file: {e}")
    
    def setup_logging(self):
        """Configure application logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_format = os.getenv('LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create logs directory if it doesn't exist
        logs_dir = self.project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logging handlers
        handlers = [logging.StreamHandler()]
        
        # Only add file handler if we can write to the logs directory
        try:
            log_file = logs_dir / 'app.log'
            handlers.append(logging.FileHandler(log_file))
        except (PermissionError, OSError):
            # Fall back to console logging only (useful for cloud deployments)
            pass
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=handlers
        )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'
    
    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return os.getenv('DEBUG', 'false').lower() == 'true'
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Safely retrieve sensitive configuration values
        
        Args:
            key: Environment variable key
            default: Default value if key not found
            
        Returns:
            Secret value or default
        """
        return os.getenv(key, default)
    
    def validate_config(self) -> bool:
        """
        Validate all configuration settings
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate database path is writable
            db_dir = Path(self.database.path).parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate analytics thresholds
            assert 0 <= self.analytics.risk_prediction_threshold <= 1
            assert self.analytics.model_retrain_interval_days > 0
            
            # Validate security settings
            assert self.security.session_timeout_minutes > 0
            assert self.security.max_file_upload_size_mb > 0
            
            return True
            
        except (AssertionError, PermissionError, OSError) as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary representation of all configuration
        """
        return {
            'database': {
                'path': self.database.path,
                'timeout': self.database.timeout
            },
            'analytics': {
                'risk_threshold': self.analytics.risk_prediction_threshold,
                'retrain_interval': self.analytics.model_retrain_interval_days
            },
            'ui': {
                'title': self.ui.page_title,
                'layout': self.ui.layout
            },
            'security': {
                'anonymization': self.security.enable_data_anonymization,
                'session_timeout': self.security.session_timeout_minutes
            }
        }

# Global configuration instance
config = Config()