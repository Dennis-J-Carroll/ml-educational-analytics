"""
Database package for Educational Analytics Dashboard

This package provides comprehensive database management including:
- Connection management with automatic retries
- Data validation and sanitization  
- Transaction handling with rollback support
- Performance optimization with indexing
- Error logging and recovery mechanisms
"""

from .manager import DatabaseManager, DatabaseError

__all__ = ['DatabaseManager', 'DatabaseError']