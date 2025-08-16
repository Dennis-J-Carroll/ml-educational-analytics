"""
Analytics Engine for Educational Analytics Dashboard

This module provides comprehensive analytics capabilities including:
- Risk prediction using machine learning models
- Performance trend analysis
- Engagement scoring algorithms
- Predictive modeling for student outcomes
- Statistical analysis and insights generation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
from pathlib import Path

from ..config.settings import config
from ..database.manager import DatabaseManager

logger = logging.getLogger(__name__)

class AnalyticsError(Exception):
    """Custom exception for analytics-related errors"""
    pass

class RiskPredictor:
    """
    Machine learning model for predicting student at-risk status
    
    This class encapsulates all risk prediction logic including
    feature engineering, model training, and prediction generation.
    """
    
    def __init__(self):
        """Initialize risk prediction model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.model_path = Path("models/risk_predictor.joblib")
        
        # Ensure models directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("RiskPredictor initialized")
    
    def engineer_features(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for risk prediction from session data
        
        Args:
            sessions_df: DataFrame containing student session data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if sessions_df.empty:
                logger.warning("Empty session data provided for feature engineering")
                return pd.DataFrame()
            
            # Group by user_id to calculate per-student features
            features = sessions_df.groupby('user_id').agg({
                'questions_attempted': ['sum', 'mean', 'std'],
                'correct_answers': ['sum', 'mean'],
                'time_spent_seconds': ['sum', 'mean', 'std'],
                'engagement_score': ['mean', 'std'],
                'difficulty_level': ['mean', 'max'],
                'session_id': 'count'  # Number of sessions
            }).reset_index()
            
            # Flatten column names
            features.columns = ['user_id'] + [
                f'{col[0]}_{col[1]}' if col[1] else col[0]
                for col in features.columns[1:]
            ]
            
            # Calculate derived features
            features['accuracy_rate'] = (
                features['correct_answers_sum'] / 
                features['questions_attempted_sum'].replace(0, np.nan)
            ).fillna(0)
            
            features['avg_session_length'] = (
                features['time_spent_seconds_sum'] / 
                features['session_id_count']
            )
            
            features['consistency_score'] = 1 / (1 + features['engagement_score_std'].fillna(0))
            
            # Time-based features
            if 'start_time' in sessions_df.columns:
                latest_sessions = sessions_df.groupby('user_id')['start_time'].agg(['min', 'max'])
                features = features.merge(latest_sessions.reset_index(), on='user_id', how='left')
                
                # Days since first session
                features['days_active'] = (
                    pd.to_datetime(features['max']) - pd.to_datetime(features['min'])
                ).dt.days.fillna(0)
                
                # Days since last session
                features['days_since_last'] = (
                    datetime.now() - pd.to_datetime(features['max'])
                ).dt.days.fillna(0)
            
            # Handle missing values
            features = features.fillna(0)
            
            # Store feature columns for prediction
            self.feature_columns = [col for col in features.columns if col != 'user_id']
            
            logger.debug(f"Engineered {len(self.feature_columns)} features for {len(features)} students")
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise AnalyticsError(f"Feature engineering error: {e}")
    
    def create_risk_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Create risk labels based on performance indicators
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Series containing risk labels (0=low risk, 1=high risk)
        """
        try:
            # Define risk criteria
            low_accuracy = features_df['accuracy_rate'] < 0.6
            few_sessions = features_df['session_id_count'] < 3
            long_absence = features_df['days_since_last'] > 7
            low_engagement = features_df['engagement_score_mean'] < 0.4
            
            # Combine criteria (student is at risk if multiple criteria are met)
            risk_score = (
                low_accuracy.astype(int) +
                few_sessions.astype(int) +
                long_absence.astype(int) +
                low_engagement.astype(int)
            )
            
            # Consider at risk if 2 or more criteria are met
            risk_labels = (risk_score >= 2).astype(int)
            
            logger.debug(f"Generated risk labels: {risk_labels.sum()} high-risk students out of {len(risk_labels)}")
            return risk_labels
            
        except Exception as e:
            logger.error(f"Risk label creation failed: {e}")
            raise AnalyticsError(f"Risk labeling error: {e}")
    
    def train(self, sessions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the risk prediction model
        
        Args:
            sessions_df: DataFrame containing training data
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            if len(sessions_df) < config.analytics.min_data_points_for_prediction:
                raise AnalyticsError(
                    f"Insufficient data for training: {len(sessions_df)} < {config.analytics.min_data_points_for_prediction}"
                )
            
            # Engineer features
            features_df = self.engineer_features(sessions_df)
            if features_df.empty:
                raise AnalyticsError("No features generated from session data")
            
            # Create labels
            labels = self.create_risk_labels(features_df)
            
            # Prepare feature matrix
            X = features_df[self.feature_columns]
            y = labels
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            self.save_model()
            self.is_trained = True
            
            metrics = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'high_risk_ratio': y.mean()
            }
            
            logger.info(f"Risk prediction model trained successfully: {accuracy:.3f} accuracy")
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise AnalyticsError(f"Training error: {e}")
    
    def predict(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk scores for students
        
        Args:
            sessions_df: DataFrame containing session data
            
        Returns:
            DataFrame with user_id and risk predictions
        """
        try:
            if not self.is_trained:
                self.load_model()
            
            if not self.is_trained:
                raise AnalyticsError("Model not trained. Cannot make predictions.")
            
            # Engineer features
            features_df = self.engineer_features(sessions_df)
            if features_df.empty:
                logger.warning("No features available for prediction")
                return pd.DataFrame(columns=['user_id', 'risk_score', 'risk_level'])
            
            # Prepare features
            X = features_df[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            risk_probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of high risk
            risk_labels = self.model.predict(X_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'user_id': features_df['user_id'],
                'risk_score': risk_probabilities,
                'risk_level': ['high' if r == 1 else 'low' for r in risk_labels]
            })
            
            logger.debug(f"Generated risk predictions for {len(results)} students")
            return results
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            raise AnalyticsError(f"Prediction error: {e}")
    
    def save_model(self):
        """Save trained model and scaler to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load trained model and scaler from disk"""
        try:
            if not self.model_path.exists():
                logger.warning("No saved model found")
                return False
            
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for student analytics
    
    This class provides various statistical analysis methods for
    understanding student performance patterns and trends.
    """
    
    def __init__(self):
        """Initialize performance analyzer"""
        logger.info("PerformanceAnalyzer initialized")
    
    def calculate_engagement_score(self, session_data: Dict[str, Any]) -> float:
        """
        Calculate engagement score based on session data
        
        Args:
            session_data: Dictionary containing session metrics
            
        Returns:
            Engagement score between 0 and 1
        """
        try:
            weights = config.analytics.engagement_score_weights
            
            # Normalize metrics to 0-1 scale
            questions_score = min(session_data.get('questions_attempted', 0) / 20, 1.0)
            accuracy_score = session_data.get('correct_answers', 0) / max(session_data.get('questions_attempted', 1), 1)
            time_score = min(session_data.get('time_spent_seconds', 0) / 1800, 1.0)  # 30 minutes max
            
            # Weighted combination
            engagement_score = (
                questions_score * weights['questions_attempted'] +
                accuracy_score * weights['correct_answers'] +
                time_score * weights['time_spent']
            )
            
            return min(max(engagement_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Engagement score calculation failed: {e}")
            return 0.0
    
    def analyze_performance_trends(self, sessions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze performance trends over time
        
        Args:
            sessions_df: DataFrame containing session data
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            if sessions_df.empty:
                return {'trend': 'no_data', 'direction': 'none', 'strength': 0}
            
            # Calculate daily performance metrics
            sessions_df['date'] = pd.to_datetime(sessions_df['start_time']).dt.date
            daily_performance = sessions_df.groupby('date').agg({
                'correct_answers': 'sum',
                'questions_attempted': 'sum',
                'engagement_score': 'mean'
            }).reset_index()
            
            daily_performance['accuracy'] = (
                daily_performance['correct_answers'] / 
                daily_performance['questions_attempted'].replace(0, np.nan)
            ).fillna(0)
            
            # Calculate trend using linear regression
            if len(daily_performance) >= 3:
                x = np.arange(len(daily_performance))
                accuracy_trend = np.polyfit(x, daily_performance['accuracy'], 1)[0]
                engagement_trend = np.polyfit(x, daily_performance['engagement_score'], 1)[0]
                
                # Determine overall trend direction and strength
                overall_trend = (accuracy_trend + engagement_trend) / 2
                
                if overall_trend > 0.01:
                    direction = 'improving'
                elif overall_trend < -0.01:
                    direction = 'declining'
                else:
                    direction = 'stable'
                
                strength = min(abs(overall_trend) * 100, 1.0)
                
            else:
                direction = 'insufficient_data'
                strength = 0
            
            return {
                'direction': direction,
                'strength': strength,
                'accuracy_trend': accuracy_trend if len(daily_performance) >= 3 else 0,
                'engagement_trend': engagement_trend if len(daily_performance) >= 3 else 0,
                'data_points': len(daily_performance)
            }
            
        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
            return {'trend': 'error', 'direction': 'unknown', 'strength': 0}

class AnalyticsEngine:
    """
    Main analytics engine that coordinates all analytical operations
    
    This class provides a unified interface for all analytics functionality
    including risk prediction, performance analysis, and insights generation.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize analytics engine with database connection
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.risk_predictor = RiskPredictor()
        self.performance_analyzer = PerformanceAnalyzer()
        
        logger.info("AnalyticsEngine initialized successfully")
    
    def analyze_student_risk(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis for students
        
        Args:
            user_id: Specific user ID to analyze (optional)
            
        Returns:
            Dictionary containing risk analysis results
        """
        try:
            # Get session data
            sessions_df = self.db_manager.get_student_sessions(user_id=user_id)
            
            if sessions_df.empty:
                return {
                    'status': 'no_data',
                    'message': 'No session data available for analysis'
                }
            
            # Generate risk predictions
            risk_predictions = self.risk_predictor.predict(sessions_df)
            
            # Calculate summary statistics
            high_risk_count = len(risk_predictions[risk_predictions['risk_level'] == 'high'])
            total_students = len(risk_predictions)
            
            result = {
                'status': 'success',
                'total_students': total_students,
                'high_risk_count': high_risk_count,
                'risk_percentage': (high_risk_count / total_students * 100) if total_students > 0 else 0,
                'predictions': risk_predictions.to_dict('records')
            }
            
            logger.info(f"Risk analysis completed: {high_risk_count}/{total_students} high-risk students")
            return result
            
        except Exception as e:
            logger.error(f"Student risk analysis failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def generate_performance_insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance insights
        
        Args:
            user_id: Specific user ID to analyze (optional)
            
        Returns:
            Dictionary containing performance insights
        """
        try:
            # Get session data
            sessions_df = self.db_manager.get_student_sessions(user_id=user_id)
            
            if sessions_df.empty:
                return {
                    'status': 'no_data',
                    'insights': []
                }
            
            insights = []
            
            # Overall performance metrics
            total_sessions = len(sessions_df)
            avg_accuracy = (sessions_df['correct_answers'].sum() / 
                          sessions_df['questions_attempted'].sum()) if sessions_df['questions_attempted'].sum() > 0 else 0
            avg_engagement = sessions_df['engagement_score'].mean()
            
            insights.append({
                'type': 'overall_performance',
                'metric': 'accuracy',
                'value': avg_accuracy,
                'description': f"Overall accuracy rate: {avg_accuracy:.1%}"
            })
            
            insights.append({
                'type': 'overall_performance', 
                'metric': 'engagement',
                'value': avg_engagement,
                'description': f"Average engagement score: {avg_engagement:.2f}/1.0"
            })
            
            # Trend analysis
            trend_analysis = self.performance_analyzer.analyze_performance_trends(sessions_df)
            insights.append({
                'type': 'trend_analysis',
                'metric': 'direction',
                'value': trend_analysis['direction'],
                'description': f"Performance trend: {trend_analysis['direction']}"
            })
            
            # Activity patterns
            if 'start_time' in sessions_df.columns:
                sessions_df['hour'] = pd.to_datetime(sessions_df['start_time']).dt.hour
                peak_hour = sessions_df['hour'].mode().iloc[0] if not sessions_df['hour'].mode().empty else 12
                
                insights.append({
                    'type': 'activity_pattern',
                    'metric': 'peak_hour',
                    'value': peak_hour,
                    'description': f"Most active time: {peak_hour}:00"
                })
            
            return {
                'status': 'success',
                'insights': insights,
                'summary': {
                    'total_sessions': total_sessions,
                    'avg_accuracy': avg_accuracy,
                    'avg_engagement': avg_engagement
                }
            }
            
        except Exception as e:
            logger.error(f"Performance insight generation failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train all machine learning models with current data
        
        Returns:
            Dictionary containing training results
        """
        try:
            # Get all session data for training
            sessions_df = self.db_manager.get_student_sessions()
            
            if len(sessions_df) < config.analytics.min_data_points_for_prediction:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need at least {config.analytics.min_data_points_for_prediction} data points for training'
                }
            
            # Train risk prediction model
            training_metrics = self.risk_predictor.train(sessions_df)
            
            result = {
                'status': 'success',
                'models_trained': ['risk_predictor'],
                'training_metrics': training_metrics,
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info("Model training completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }