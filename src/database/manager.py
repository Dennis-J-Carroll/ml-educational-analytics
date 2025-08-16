"""
Database Management for Educational Analytics Dashboard

This module provides comprehensive database operations including:
- Connection management with automatic retries
- Data validation and sanitization
- Transaction handling
- Performance optimization
- Error logging and recovery
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import threading
from contextlib import contextmanager

from ..config.settings import config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass

class DatabaseManager:
    """
    Comprehensive database manager for educational analytics data
    
    This class handles all database operations with automatic connection
    management, error handling, and performance optimization.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager with connection pooling
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.database.path
        self.timeout = config.database.timeout
        self._local = threading.local()
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self.init_database()
        
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    @property
    def connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection with automatic management
        
        Returns:
            SQLite connection object
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            try:
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    timeout=self.timeout,
                    check_same_thread=False
                )
                self._local.connection.row_factory = sqlite3.Row
                # Enable foreign key constraints
                self._local.connection.execute("PRAGMA foreign_keys = ON")
                logger.debug("New database connection established")
                
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                raise DatabaseError(f"Database connection failed: {e}")
        
        return self._local.connection
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions with automatic rollback
        
        Usage:
            with db_manager.transaction():
                db_manager.execute("INSERT ...")
                db_manager.execute("UPDATE ...")
        """
        conn = self.connection
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    def init_database(self):
        """
        Initialize database schema with all required tables
        
        Creates all necessary tables for student analytics with proper
        indexes and constraints for optimal performance.
        """
        try:
            with self.transaction() as conn:
                # Student sessions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS student_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL UNIQUE,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        questions_attempted INTEGER DEFAULT 0,
                        correct_answers INTEGER DEFAULT 0,
                        difficulty_level INTEGER DEFAULT 1,
                        time_spent_seconds INTEGER DEFAULT 0,
                        engagement_score REAL DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        CHECK (questions_attempted >= 0),
                        CHECK (correct_answers >= 0),
                        CHECK (correct_answers <= questions_attempted),
                        CHECK (difficulty_level BETWEEN 1 AND 10),
                        CHECK (engagement_score BETWEEN 0 AND 1)
                    )
                ''')
                
                # Question attempts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS question_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        question_type TEXT NOT NULL,
                        difficulty INTEGER NOT NULL,
                        time_taken_seconds REAL NOT NULL,
                        is_correct BOOLEAN NOT NULL,
                        hint_used BOOLEAN DEFAULT FALSE,
                        attempt_number INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES student_sessions (session_id),
                        CHECK (difficulty BETWEEN 1 AND 10),
                        CHECK (time_taken_seconds > 0),
                        CHECK (attempt_number > 0)
                    )
                ''')
                
                # Student profiles table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS student_profiles (
                        user_id TEXT PRIMARY KEY,
                        student_name TEXT,
                        grade_level TEXT,
                        learning_style TEXT,
                        risk_score REAL DEFAULT 0,
                        total_sessions INTEGER DEFAULT 0,
                        total_time_spent INTEGER DEFAULT 0,
                        average_performance REAL DEFAULT 0,
                        last_activity TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        CHECK (risk_score BETWEEN 0 AND 1),
                        CHECK (total_sessions >= 0),
                        CHECK (average_performance BETWEEN 0 AND 1)
                    )
                ''')
                
                # Performance analytics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        date TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        context_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES student_profiles (user_id),
                        UNIQUE(user_id, date, metric_type)
                    )
                ''')
                
                # Create indexes for performance optimization
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON student_sessions(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON student_sessions(start_time)",
                    "CREATE INDEX IF NOT EXISTS idx_questions_session_id ON question_attempts(session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_analytics_user_date ON performance_analytics(user_id, date)",
                    "CREATE INDEX IF NOT EXISTS idx_profiles_risk_score ON student_profiles(risk_score)"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
            logger.info("Database schema initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise DatabaseError(f"Schema initialization failed: {e}")
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """
        Execute a SELECT query with parameters
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            
        Returns:
            List of database rows
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            logger.debug(f"Query executed successfully: {len(results)} rows returned")
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query failed: {e}")
    
    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            
        Returns:
            Number of affected rows
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                affected_rows = cursor.rowcount
                logger.debug(f"Update executed successfully: {affected_rows} rows affected")
                return affected_rows
                
        except sqlite3.Error as e:
            logger.error(f"Update execution failed: {e}")
            raise DatabaseError(f"Update failed: {e}")
    
    def get_student_sessions(self, user_id: Optional[str] = None, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve student session data with optional filtering
        
        Args:
            user_id: Specific user ID to filter
            start_date: Start date for filtering (ISO format)
            end_date: End date for filtering (ISO format)
            
        Returns:
            DataFrame containing session data
        """
        query = "SELECT * FROM student_sessions WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND start_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND start_time <= ?"
            params.append(end_date)
        
        query += " ORDER BY start_time DESC"
        
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"Failed to retrieve student sessions: {e}")
            raise DatabaseError(f"Session retrieval failed: {e}")
    
    def save_session_data(self, session_data: Dict[str, Any]) -> str:
        """
        Save student session data with validation
        
        Args:
            session_data: Dictionary containing session information
            
        Returns:
            Session ID of saved record
        """
        required_fields = ['user_id', 'session_id', 'start_time']
        
        # Validate required fields
        for field in required_fields:
            if field not in session_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Sanitize and validate data
        sanitized_data = self._sanitize_session_data(session_data)
        
        try:
            query = '''
                INSERT OR REPLACE INTO student_sessions 
                (user_id, session_id, start_time, end_time, questions_attempted, 
                 correct_answers, difficulty_level, time_spent_seconds, engagement_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                sanitized_data['user_id'],
                sanitized_data['session_id'],
                sanitized_data['start_time'],
                sanitized_data.get('end_time'),
                sanitized_data.get('questions_attempted', 0),
                sanitized_data.get('correct_answers', 0),
                sanitized_data.get('difficulty_level', 1),
                sanitized_data.get('time_spent_seconds', 0),
                sanitized_data.get('engagement_score', 0.0)
            )
            
            self.execute_update(query, params)
            logger.info(f"Session data saved for user: {sanitized_data['user_id']}")
            return sanitized_data['session_id']
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
            raise DatabaseError(f"Session save failed: {e}")
    
    def _sanitize_session_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and validate session data
        
        Args:
            data: Raw session data dictionary
            
        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        
        # String fields
        string_fields = ['user_id', 'session_id', 'start_time', 'end_time']
        for field in string_fields:
            if field in data and data[field] is not None:
                sanitized[field] = str(data[field]).strip()
        
        # Integer fields with validation
        int_fields = {
            'questions_attempted': (0, 1000),
            'correct_answers': (0, 1000),
            'difficulty_level': (1, 10),
            'time_spent_seconds': (0, 86400)  # Max 24 hours
        }
        
        for field, (min_val, max_val) in int_fields.items():
            if field in data and data[field] is not None:
                try:
                    value = int(data[field])
                    sanitized[field] = max(min_val, min(max_val, value))
                except (ValueError, TypeError):
                    sanitized[field] = min_val
        
        # Float fields with validation
        if 'engagement_score' in data and data['engagement_score'] is not None:
            try:
                score = float(data['engagement_score'])
                sanitized['engagement_score'] = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                sanitized['engagement_score'] = 0.0
        
        return sanitized
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics summary
        
        Returns:
            Dictionary containing key analytics metrics
        """
        try:
            summary = {}
            
            # Total students
            result = self.execute_query("SELECT COUNT(DISTINCT user_id) as count FROM student_sessions")
            summary['total_students'] = result[0]['count'] if result else 0
            
            # Active sessions today
            today = datetime.now().strftime('%Y-%m-%d')
            result = self.execute_query(
                "SELECT COUNT(*) as count FROM student_sessions WHERE date(start_time) = ?",
                (today,)
            )
            summary['active_today'] = result[0]['count'] if result else 0
            
            # Average performance
            result = self.execute_query("""
                SELECT AVG(CAST(correct_answers AS FLOAT) / NULLIF(questions_attempted, 0)) as avg_performance
                FROM student_sessions 
                WHERE questions_attempted > 0
            """)
            summary['average_performance'] = round(result[0]['avg_performance'] or 0, 3)
            
            # At-risk students count
            result = self.execute_query(
                "SELECT COUNT(*) as count FROM student_profiles WHERE risk_score > ?",
                (config.analytics.risk_prediction_threshold,)
            )
            summary['at_risk_students'] = result[0]['count'] if result else 0
            
            logger.debug("Analytics summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate analytics summary: {e}")
            raise DatabaseError(f"Analytics summary failed: {e}")
    
    def close_connection(self):
        """Close database connection if it exists"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("Database connection closed")
    
    def __del__(self):
        """Cleanup database connections on deletion"""
        self.close_connection()