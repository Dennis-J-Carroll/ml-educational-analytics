# Educational Analytics Dashboard - API Documentation

## Overview

The Educational Analytics Dashboard provides a comprehensive API for managing student data, performing analytics, and generating insights. The system is built with a modular architecture that separates concerns into distinct packages.

## Architecture

```
src/
├── config/         # Configuration management
├── database/       # Database operations  
├── analytics/      # ML models and analysis
├── visualization/  # UI components
└── utils/          # Utility functions
```

## Configuration Module (`src.config`)

### Config Class

The main configuration class that consolidates all application settings.

```python
from src.config import Config

config = Config()
```

#### Methods

- `validate_config() -> bool`: Validate all configuration settings
- `to_dict() -> Dict[str, Any]`: Convert configuration to dictionary
- `get_secret(key: str, default: Optional[str] = None) -> Optional[str]`: Safely retrieve sensitive values

#### Properties

- `is_production: bool`: Check if running in production
- `debug_mode: bool`: Check if debug mode is enabled

### Configuration Sections

#### DatabaseConfig
```python
@dataclass
class DatabaseConfig:
    path: str = "learning_analytics.db"
    timeout: int = 30
    check_same_thread: bool = False
```

#### AnalyticsConfig
```python
@dataclass  
class AnalyticsConfig:
    risk_prediction_threshold: float = 0.65
    engagement_score_weights: Dict[str, float] = None
    model_retrain_interval_days: int = 7
    min_data_points_for_prediction: int = 5
```

## Database Module (`src.database`)

### DatabaseManager Class

Comprehensive database manager with connection pooling and transaction support.

```python
from src.database import DatabaseManager

db_manager = DatabaseManager()
```

#### Methods

##### Data Retrieval
- `get_student_sessions(user_id=None, start_date=None, end_date=None) -> pd.DataFrame`
- `get_analytics_summary() -> Dict[str, Any]`

##### Data Manipulation
- `save_session_data(session_data: Dict[str, Any]) -> str`
- `execute_query(query: str, params: Tuple = ()) -> List[sqlite3.Row]`
- `execute_update(query: str, params: Tuple = ()) -> int`

##### Transaction Management
```python
with db_manager.transaction():
    db_manager.execute_update("INSERT ...")
    db_manager.execute_update("UPDATE ...")
```

## Analytics Module (`src.analytics`)

### AnalyticsEngine Class

Main analytics engine that coordinates all analytical operations.

```python
from src.analytics import AnalyticsEngine

analytics = AnalyticsEngine(db_manager)
```

#### Methods

- `analyze_student_risk(user_id: Optional[str] = None) -> Dict[str, Any]`
- `generate_performance_insights(user_id: Optional[str] = None) -> Dict[str, Any]`
- `train_models() -> Dict[str, Any]`

#### Example Usage

```python
# Analyze risk for all students
risk_analysis = analytics.analyze_student_risk()

# Generate insights for specific student  
insights = analytics.generate_performance_insights(user_id="student_123")

# Retrain ML models
training_result = analytics.train_models()
```

### RiskPredictor Class

Machine learning model for predicting student at-risk status.

```python
from src.analytics import RiskPredictor

predictor = RiskPredictor()
```

#### Methods

- `train(sessions_df: pd.DataFrame) -> Dict[str, Any]`
- `predict(sessions_df: pd.DataFrame) -> pd.DataFrame`
- `save_model()` / `load_model()`

### PerformanceAnalyzer Class

Statistical analysis for student performance patterns.

```python
from src.analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
```

#### Methods

- `calculate_engagement_score(session_data: Dict[str, Any]) -> float`
- `analyze_performance_trends(sessions_df: pd.DataFrame) -> Dict[str, Any]`

## Visualization Module (`src.visualization`)

### DashboardComponents Class

Reusable dashboard components for displaying analytics data.

```python
from src.visualization import DashboardComponents

components = DashboardComponents()
```

#### Methods

- `render_metric_card(title, value, delta=None, delta_color="normal", help_text=None)`
- `render_performance_overview(analytics_summary: Dict[str, Any])`
- `render_performance_trends_chart(sessions_df: pd.DataFrame) -> go.Figure`
- `render_risk_analysis_chart(risk_data: List[Dict]) -> go.Figure`
- `render_student_details_table(sessions_df, risk_data)`
- `render_insights_panel(insights: List[Dict])`

### ChartFactory Class

Factory for creating standardized charts.

```python
from src.visualization import ChartFactory

# Create empty chart with message
fig = ChartFactory.create_empty_chart("No data available")

# Apply consistent theme
fig = ChartFactory.apply_theme(fig, title="Performance Chart")
```

## Utils Module (`src.utils`)

### DataValidator Class

Comprehensive data validation utilities.

```python
from src.utils import DataValidator

# Validate user ID
is_valid = DataValidator.validate_user_id("user_123")

# Validate session data
is_valid, errors = DataValidator.validate_session_data(session_data)

# Sanitize string input
clean_string = DataValidator.sanitize_string(user_input)
```

### DateTimeHelper Class

Date and time utility functions.

```python
from src.utils import DateTimeHelper

# Get date range
start_date, end_date = DateTimeHelper.get_date_range(days_back=30)

# Format duration
duration_str = DateTimeHelper.format_duration(3661)  # "1h 1m"

# Check recent activity
is_recent = DateTimeHelper.is_recent_activity(timestamp)
```

### SecurityHelper Class

Security and privacy utility functions.

```python
from src.utils import SecurityHelper

# Anonymize user ID
anon_id = SecurityHelper.anonymize_user_id("user_123")

# Generate session ID
session_id = SecurityHelper.generate_session_id()

# Mask sensitive data
masked_data = SecurityHelper.mask_sensitive_data(data, ['email', 'phone'])
```

### DataProcessor Class

Data processing and transformation utilities.

```python
from src.utils import DataProcessor

# Calculate percentile rank
rank = DataProcessor.calculate_percentile_rank(85, grade_series)

# Smooth time series
smoothed = DataProcessor.smooth_time_series(noisy_data, window=7)

# Detect outliers
outliers = DataProcessor.detect_outliers(data_series, method='iqr')
```

### FileHelper Class

File handling and processing utilities.

```python
from src.utils import FileHelper

# Validate uploaded file
is_valid, message = FileHelper.validate_upload_file(file_path)

# Safe file reading
success, content = FileHelper.safe_file_read(file_path)
```

### FormatHelper Class

Data formatting and display utilities.

```python
from src.utils import FormatHelper

# Format as percentage
pct_str = FormatHelper.format_percentage(0.856)  # "85.6%"

# Format large numbers
num_str = FormatHelper.format_large_number(1500000)  # "1.5M"

# Truncate text
short_text = FormatHelper.truncate_text(long_text, max_length=50)
```

## Error Handling

The system includes custom exception classes for different types of errors:

```python
from src.database import DatabaseError
from src.analytics import AnalyticsError

try:
    result = analytics.analyze_student_risk()
except AnalyticsError as e:
    logger.error(f"Analytics failed: {e}")
except DatabaseError as e:
    logger.error(f"Database error: {e}")
```

## Environment Variables

Configure the application using environment variables or `.env` file:

```bash
# Application Environment
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration  
DB_PATH=learning_analytics.db
DB_TIMEOUT=30

# Analytics Configuration
RISK_THRESHOLD=0.65
MODEL_RETRAIN_DAYS=7

# UI Configuration
PAGE_TITLE="Educational Analytics Dashboard"
UI_LAYOUT=wide

# Security Configuration
ANONYMIZE_DATA=true
SESSION_TIMEOUT=30
```

## Examples

### Complete Analytics Workflow

```python
from src.config import config
from src.database import DatabaseManager
from src.analytics import AnalyticsEngine
from src.utils import DateTimeHelper

# Initialize components
db_manager = DatabaseManager()
analytics = AnalyticsEngine(db_manager)

# Get recent session data
start_date, end_date = DateTimeHelper.get_date_range(30)
sessions_df = db_manager.get_student_sessions(
    start_date=start_date, 
    end_date=end_date
)

# Perform risk analysis
risk_analysis = analytics.analyze_student_risk()
print(f"Found {risk_analysis['high_risk_count']} high-risk students")

# Generate insights
insights = analytics.generate_performance_insights()
for insight in insights['insights']:
    print(f"💡 {insight['description']}")

# Train models if needed
if len(sessions_df) >= config.analytics.min_data_points_for_prediction:
    training_result = analytics.train_models()
    print(f"Model trained with {training_result['training_metrics']['accuracy']:.3f} accuracy")
```

### Custom Dashboard Component

```python
import streamlit as st
from src.visualization import DashboardComponents

components = DashboardComponents()

# Render metrics overview
analytics_summary = db_manager.get_analytics_summary()
components.render_performance_overview(analytics_summary)

# Create custom chart
sessions_df = db_manager.get_student_sessions()
chart = components.render_performance_trends_chart(sessions_df)
st.plotly_chart(chart, use_container_width=True)
```

This modular architecture provides a robust foundation for educational analytics while maintaining clean separation of concerns and professional code organization.