"""
Unit tests for configuration management

This module tests the configuration system including environment variable
handling, validation, and default value assignment.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Config, DatabaseConfig, AnalyticsConfig, UIConfig, SecurityConfig

class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = DatabaseConfig()
        assert config.path == "learning_analytics.db"
        assert config.timeout == 30
        assert config.check_same_thread == False
    
    def test_connection_string(self):
        """Test connection string generation"""
        config = DatabaseConfig(path="test.db")
        assert config.connection_string == "sqlite:///test.db"

class TestAnalyticsConfig:
    """Test analytics configuration"""
    
    def test_default_values(self):
        """Test default analytics settings"""
        config = AnalyticsConfig()
        assert config.risk_prediction_threshold == 0.65
        assert config.model_retrain_interval_days == 7
        assert config.min_data_points_for_prediction == 5
        assert len(config.engagement_score_weights) == 4
    
    def test_engagement_weights_sum(self):
        """Test engagement score weights"""
        config = AnalyticsConfig()
        weights_sum = sum(config.engagement_score_weights.values())
        assert abs(weights_sum - 1.0) < 0.01  # Should sum to approximately 1.0

class TestSecurityConfig:
    """Test security configuration"""
    
    def test_default_values(self):
        """Test default security settings"""
        config = SecurityConfig()
        assert config.enable_data_anonymization == True
        assert config.session_timeout_minutes == 30
        assert config.max_file_upload_size_mb == 10
        assert 'csv' in config.allowed_file_types

class TestMainConfig:
    """Test main configuration class"""
    
    def test_initialization(self):
        """Test configuration initialization"""
        config = Config()
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.analytics, AnalyticsConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.security, SecurityConfig)
    
    def test_environment_variable_override(self):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {'DB_PATH': 'custom.db', 'RISK_THRESHOLD': '0.8'}):
            config = Config()
            assert config.database.path == 'custom.db'
            assert config.analytics.risk_prediction_threshold == 0.8
    
    def test_env_file_loading(self):
        """Test loading configuration from .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("DB_PATH=env_test.db\n")
            f.write("DEBUG=true\n")
            env_file_path = f.name
        
        try:
            config = Config(env_file=env_file_path)
            assert config.database.path == 'env_test.db'
            assert config.debug_mode == True
        finally:
            os.unlink(env_file_path)
    
    def test_is_production(self):
        """Test production environment detection"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config = Config()
            assert config.is_production == True
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            config = Config()
            assert config.is_production == False
    
    def test_debug_mode(self):
        """Test debug mode detection"""
        with patch.dict(os.environ, {'DEBUG': 'true'}):
            config = Config()
            assert config.debug_mode == True
        
        with patch.dict(os.environ, {'DEBUG': 'false'}):
            config = Config()
            assert config.debug_mode == False
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        assert config.validate_config() == True
        
        # Test invalid configuration
        config.analytics.risk_prediction_threshold = 2.0  # Invalid value
        assert config.validate_config() == False
    
    def test_to_dict(self):
        """Test configuration serialization"""
        config = Config()
        config_dict = config.to_dict()
        
        assert 'database' in config_dict
        assert 'analytics' in config_dict
        assert 'ui' in config_dict
        assert 'security' in config_dict
        
        assert config_dict['database']['path'] == config.database.path
        assert config_dict['analytics']['risk_threshold'] == config.analytics.risk_prediction_threshold

if __name__ == "__main__":
    pytest.main([__file__])