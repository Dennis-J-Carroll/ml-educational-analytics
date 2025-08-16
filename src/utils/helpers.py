"""
Utility Functions for Educational Analytics Dashboard

This module provides common utility functions and helpers used throughout
the application for data processing, validation, and formatting.
"""

import logging
import pandas as pd
import numpy as np
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import re
from pathlib import Path

from ..config.settings import config

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation utilities
    
    This class provides various validation methods to ensure
    data integrity and security throughout the application.
    """
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """
        Validate user ID format and content
        
        Args:
            user_id: User identifier string
            
        Returns:
            True if valid, False otherwise
        """
        if not user_id or not isinstance(user_id, str):
            return False
        
        # Check length (reasonable limits)
        if len(user_id) < 3 or len(user_id) > 50:
            return False
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return False
        
        return True
    
    @staticmethod
    def validate_session_data(session_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate session data structure and values
        
        Args:
            session_data: Dictionary containing session information
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = ['user_id', 'session_id', 'start_time']
        for field in required_fields:
            if field not in session_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate user_id
        if 'user_id' in session_data:
            if not DataValidator.validate_user_id(session_data['user_id']):
                errors.append("Invalid user_id format")
        
        # Validate numeric fields
        numeric_fields = {
            'questions_attempted': (0, 1000),
            'correct_answers': (0, 1000),
            'time_spent_seconds': (0, 86400),  # Max 24 hours
            'difficulty_level': (1, 10)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in session_data:
                try:
                    value = float(session_data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(f"{field} out of valid range ({min_val}-{max_val})")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Validate engagement score
        if 'engagement_score' in session_data:
            try:
                score = float(session_data['engagement_score'])
                if not (0 <= score <= 1):
                    errors.append("engagement_score must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("engagement_score must be a valid number")
        
        # Logical validation
        if ('correct_answers' in session_data and 'questions_attempted' in session_data):
            try:
                correct = int(session_data['correct_answers'])
                attempted = int(session_data['questions_attempted'])
                if correct > attempted:
                    errors.append("correct_answers cannot exceed questions_attempted")
            except (ValueError, TypeError):
                pass  # Already caught above
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """
        Sanitize string input for security and consistency
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Remove potentially dangerous characters
        value = re.sub(r'[<>"\';]', '', value)
        
        # Trim whitespace and limit length
        value = value.strip()[:max_length]
        
        return value

class DateTimeHelper:
    """
    Date and time utility functions
    """
    
    @staticmethod
    def get_date_range(days_back: int = 30) -> Tuple[str, str]:
        """
        Get date range for analytics queries
        
        Args:
            days_back: Number of days to go back from today
            
        Returns:
            Tuple of (start_date, end_date) in ISO format
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return start_date.isoformat(), end_date.isoformat()
    
    @staticmethod
    def format_duration(seconds: int) -> str:
        """
        Format duration in seconds to human-readable string
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def is_recent_activity(timestamp: str, hours_threshold: int = 24) -> bool:
        """
        Check if timestamp represents recent activity
        
        Args:
            timestamp: ISO format timestamp string
            hours_threshold: Hours to consider as recent
            
        Returns:
            True if recent, False otherwise
        """
        try:
            activity_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            threshold_time = datetime.now() - timedelta(hours=hours_threshold)
            return activity_time > threshold_time
        except (ValueError, AttributeError):
            return False

class DataProcessor:
    """
    Data processing and transformation utilities
    """
    
    @staticmethod
    def calculate_percentile_rank(value: float, data_series: pd.Series) -> float:
        """
        Calculate percentile rank of a value within a series
        
        Args:
            value: Value to rank
            data_series: Series of values for comparison
            
        Returns:
            Percentile rank (0-100)
        """
        if data_series.empty:
            return 50.0  # Default to median if no data
        
        rank = (data_series < value).sum() / len(data_series) * 100
        return round(rank, 1)
    
    @staticmethod
    def smooth_time_series(series: pd.Series, window: int = 7) -> pd.Series:
        """
        Apply smoothing to time series data
        
        Args:
            series: Time series data
            window: Window size for moving average
            
        Returns:
            Smoothed series
        """
        if len(series) < window:
            return series
        
        return series.rolling(window=window, center=True).mean().fillna(series)
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a data series
        
        Args:
            series: Data series to analyze
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > 3
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

class SecurityHelper:
    """
    Security and privacy utility functions
    """
    
    @staticmethod
    def anonymize_user_id(user_id: str) -> str:
        """
        Anonymize user ID for privacy protection
        
        Args:
            user_id: Original user identifier
            
        Returns:
            Anonymized identifier
        """
        if not config.security.enable_data_anonymization:
            return user_id
        
        # Create consistent hash of user_id
        hash_object = hashlib.sha256(user_id.encode())
        hash_hex = hash_object.hexdigest()
        
        # Return first 8 characters as anonymous ID
        return f"user_{hash_hex[:8]}"
    
    @staticmethod
    def generate_session_id() -> str:
        """
        Generate secure session identifier
        
        Returns:
            Unique session ID
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], 
                           sensitive_fields: List[str] = None) -> Dict[str, Any]:
        """
        Mask sensitive fields in data dictionary
        
        Args:
            data: Data dictionary to process
            sensitive_fields: List of field names to mask
            
        Returns:
            Data with sensitive fields masked
        """
        if sensitive_fields is None:
            sensitive_fields = ['email', 'phone', 'address', 'ssn']
        
        masked_data = data.copy()
        
        for field in sensitive_fields:
            if field in masked_data:
                value = str(masked_data[field])
                if len(value) > 4:
                    masked_data[field] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked_data[field] = '*' * len(value)
        
        return masked_data

class FileHelper:
    """
    File handling and processing utilities
    """
    
    @staticmethod
    def validate_upload_file(file_path: Path, allowed_extensions: List[str] = None) -> Tuple[bool, str]:
        """
        Validate uploaded file for security and format
        
        Args:
            file_path: Path to uploaded file
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if allowed_extensions is None:
            allowed_extensions = config.security.allowed_file_types
        
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file extension
        extension = file_path.suffix.lower().lstrip('.')
        if extension not in allowed_extensions:
            return False, f"File type '{extension}' not allowed"
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.security.max_file_upload_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({config.security.max_file_upload_size_mb}MB)"
        
        return True, "File is valid"
    
    @staticmethod
    def safe_file_read(file_path: Path, encoding: str = 'utf-8') -> Tuple[bool, Union[str, pd.DataFrame]]:
        """
        Safely read file content with error handling
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            Tuple of (success, content_or_error_message)
        """
        try:
            extension = file_path.suffix.lower()
            
            if extension == '.csv':
                return True, pd.read_csv(file_path, encoding=encoding)
            elif extension in ['.xlsx', '.xls']:
                return True, pd.read_excel(file_path)
            elif extension == '.json':
                with open(file_path, 'r', encoding=encoding) as f:
                    return True, json.load(f)
            else:
                with open(file_path, 'r', encoding=encoding) as f:
                    return True, f.read()
                    
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return False, str(e)

class FormatHelper:
    """
    Data formatting and display utilities
    """
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """
        Format number as percentage string
        
        Args:
            value: Decimal value (0.75 for 75%)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def format_large_number(value: int) -> str:
        """
        Format large numbers with appropriate suffixes
        
        Args:
            value: Number to format
            
        Returns:
            Formatted number string
        """
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.1f}K"
        else:
            return str(value)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50) -> str:
        """
        Truncate text with ellipsis if too long
        
        Args:
            text: Text to truncate
            max_length: Maximum length before truncation
            
        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."