"""
Educational AI Analytics Dashboard
Transformation from: Simple Math Game → EdTech Analytics Platform
Target: Online course creators, tutoring companies
Value Prop: "Increase student engagement by 40% through personalized learning insights"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import uuid

class GameAnalytics:
    def __init__(self, db_path="learning_analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                questions_attempted INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                difficulty_level INTEGER DEFAULT 1,
                time_spent_seconds INTEGER DEFAULT 0,
                engagement_score REAL DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question_type TEXT NOT NULL,
                difficulty INTEGER NOT NULL,
                time_taken_seconds REAL NOT NULL,
                correct BOOLEAN NOT NULL,
                timestamp TEXT NOT NULL,
                hints_used INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                insight_data TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                confidence_score REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        
        # Check if we need to create sample data
        cursor.execute("SELECT COUNT(*) FROM student_sessions")
        session_count = cursor.fetchone()[0]
        
        if session_count == 0:
            self._create_sample_data(cursor)
            conn.commit()
        
        conn.close()
    
    def _create_sample_data(self, cursor):
        """Create sample data for demo purposes"""
        import random
        from datetime import datetime, timedelta
        
        # Create sample students and sessions
        students = [f"student_{i:03d}" for i in range(1, 151)]  # 150 students
        question_types = ["arithmetic", "algebra", "geometry", "statistics", "calculus"]
        
        for i, student in enumerate(students):
            # Create 1-3 sessions per student
            num_sessions = random.randint(1, 3)
            
            for session_num in range(num_sessions):
                session_id = f"session_{student}_{session_num}"
                start_time = (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                questions_attempted = random.randint(5, 25)
                correct_answers = random.randint(int(questions_attempted * 0.4), questions_attempted)
                time_spent = random.randint(300, 1800)  # 5-30 minutes
                engagement_score = random.uniform(0.3, 1.0)
                
                cursor.execute('''
                    INSERT INTO student_sessions 
                    (user_id, session_id, start_time, questions_attempted, correct_answers, 
                     difficulty_level, time_spent_seconds, engagement_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (student, session_id, start_time, questions_attempted, correct_answers,
                      random.randint(1, 5), time_spent, engagement_score))
                
                # Create individual question attempts for this session
                for q in range(questions_attempted):
                    question_type = random.choice(question_types)
                    difficulty = random.randint(1, 5)
                    time_taken = random.uniform(10, 180)  # 10 seconds to 3 minutes
                    correct = random.random() < (correct_answers / questions_attempted)
                    timestamp = (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                    hints_used = random.randint(0, 3) if not correct else 0
                    
                    cursor.execute('''
                        INSERT INTO question_attempts
                        (session_id, question_type, difficulty, time_taken_seconds, 
                         correct, timestamp, hints_used)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, question_type, difficulty, time_taken, 
                          correct, timestamp, hints_used))
    
    def generate_user_id(self):
        return f"student_{uuid.uuid4().hex[:8]}"
    
    def start_session(self, user_id):
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO student_sessions (user_id, session_id, start_time)
            VALUES (?, ?, ?)
        ''', (user_id, session_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return session_id
    
    def track_answer(self, session_id, question_type, difficulty, time_taken, correct, hints_used=0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO question_attempts 
            (session_id, question_type, difficulty, time_taken_seconds, correct, timestamp, hints_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, question_type, difficulty, time_taken, correct, 
              datetime.now().isoformat(), hints_used))
        
        # Update session statistics
        cursor.execute('''
            UPDATE student_sessions 
            SET questions_attempted = questions_attempted + 1,
                correct_answers = correct_answers + ?
            WHERE session_id = ?
        ''', (1 if correct else 0, session_id))
        
        conn.commit()
        conn.close()
    
    def generate_learning_insights(self, user_id):
        conn = sqlite3.connect(self.db_path)
        
        # Get user's question attempts
        query = '''
            SELECT qa.*, ss.user_id 
            FROM question_attempts qa
            JOIN student_sessions ss ON qa.session_id = ss.session_id
            WHERE ss.user_id = ?
            ORDER BY qa.timestamp
        '''
        attempts_df = pd.read_sql_query(query, conn, params=(user_id,))
        
        if len(attempts_df) == 0:
            conn.close()
            return {"message": "Not enough data for insights"}
        
        insights = {}
        
        # 1. Learning Pattern Analysis
        if len(attempts_df) > 10:
            accuracy_trend = attempts_df.groupby(attempts_df.index // 10)['correct'].mean()
            if len(accuracy_trend) > 1:
                slope = LinearRegression().fit(
                    np.array(range(len(accuracy_trend))).reshape(-1, 1),
                    accuracy_trend
                ).coef_[0]
                
                insights['learning_trend'] = {
                    'direction': 'improving' if slope > 0.02 else 'declining' if slope < -0.02 else 'stable',
                    'slope': slope,
                    'confidence': min(0.9, len(accuracy_trend) / 10)
                }
        
        # 2. Difficulty Optimization
        difficulty_performance = attempts_df.groupby('difficulty').agg({
            'correct': 'mean',
            'time_taken_seconds': 'mean'
        })
        
        optimal_difficulty = None
        for diff in difficulty_performance.index:
            accuracy = difficulty_performance.loc[diff, 'correct']
            if 0.6 <= accuracy <= 0.8:  # Sweet spot
                optimal_difficulty = diff
                break
        
        insights['difficulty_recommendation'] = {
            'current_optimal': optimal_difficulty,
            'performance_by_level': difficulty_performance.to_dict()
        }
        
        # 3. Time Pattern Analysis
        attempts_df['hour'] = pd.to_datetime(attempts_df['timestamp']).dt.hour
        hourly_performance = attempts_df.groupby('hour')['correct'].mean()
        best_hours = hourly_performance.nlargest(3).index.tolist()
        
        insights['optimal_study_times'] = {
            'best_hours': best_hours,
            'performance_by_hour': hourly_performance.to_dict()
        }
        
        # 4. Weakness Identification
        topic_performance = attempts_df.groupby('question_type')['correct'].mean()
        weak_topics = topic_performance[topic_performance < 0.6].index.tolist()
        strong_topics = topic_performance[topic_performance > 0.8].index.tolist()
        
        insights['topic_analysis'] = {
            'weak_areas': weak_topics,
            'strong_areas': strong_topics,
            'performance_by_topic': topic_performance.to_dict()
        }
        
        # Store insights
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO learning_insights (user_id, insight_type, insight_data, generated_at, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, 'comprehensive_analysis', json.dumps(insights), 
              datetime.now().isoformat(), 0.85))
        
        conn.commit()
        conn.close()
        
        return insights

class EducationalAnalyticsDashboard:
    def __init__(self):
        self.analytics = GameAnalytics()
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Educational AI Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def generate_demo_data(self, num_students=50):
        """Generate realistic demo data for the dashboard"""
        np.random.seed(42)
        
        for student_num in range(num_students):
            user_id = f"demo_student_{student_num:03d}"
            
            # Simulate multiple sessions per student
            sessions = np.random.randint(3, 15)
            
            for session_num in range(sessions):
                session_id = self.analytics.start_session(user_id)
                
                # Simulate learning curve
                base_accuracy = 0.4 + (session_num / sessions) * 0.4
                session_questions = np.random.randint(10, 30)
                
                for q in range(session_questions):
                    question_types = ['addition', 'subtraction', 'multiplication', 'division', 'algebra']
                    question_type = np.random.choice(question_types)
                    
                    difficulty = np.random.randint(1, 6)
                    time_taken = np.random.exponential(15) + 5  # 5-60 seconds typically
                    
                    # Accuracy influenced by difficulty and learning progress
                    accuracy_modifier = max(0.1, 1 - (difficulty - 1) * 0.15)
                    correct = np.random.random() < (base_accuracy * accuracy_modifier)
                    
                    hints_used = np.random.randint(0, 3) if not correct else 0
                    
                    self.analytics.track_answer(
                        session_id, question_type, difficulty, time_taken, correct, hints_used
                    )
    
    def create_kpi_cards(self):
        """Create KPI summary cards"""
        conn = sqlite3.connect(self.analytics.db_path)
        
        # Total students
        total_students = pd.read_sql_query(
            "SELECT COUNT(DISTINCT user_id) as count FROM student_sessions", conn
        ).iloc[0]['count']
        
        # Total questions attempted
        total_questions = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM question_attempts", conn
        ).iloc[0]['count']
        
        # Average accuracy
        accuracy_result = pd.read_sql_query(
            "SELECT AVG(CAST(correct AS FLOAT)) as accuracy FROM question_attempts", conn
        ).iloc[0]['accuracy']
        avg_accuracy = accuracy_result if accuracy_result is not None else 0.0
        
        # Active learners (last 7 days)
        active_learners = pd.read_sql_query('''
            SELECT COUNT(DISTINCT ss.user_id) as count 
            FROM student_sessions ss 
            WHERE ss.start_time > datetime('now', '-7 days')
        ''', conn).iloc[0]['count']
        
        conn.close()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📚 Total Students",
                value=f"{total_students:,}",
                delta=f"+{int(total_students * 0.1)} this week"
            )
        
        with col2:
            st.metric(
                label="❓ Questions Attempted", 
                value=f"{total_questions:,}",
                delta=f"+{int(total_questions * 0.15)} today"
            )
        
        with col3:
            st.metric(
                label="🎯 Average Accuracy",
                value=f"{avg_accuracy:.1%}",
                delta="+2.3%" if avg_accuracy > 0.65 else "-1.2%"
            )
        
        with col4:
            st.metric(
                label="🔥 Active Learners",
                value=f"{active_learners:,}",
                delta=f"+{max(1, int(active_learners * 0.2))}"
            )
    
    def learning_patterns_chart(self):
        """Create learning pattern visualization"""
        conn = sqlite3.connect(self.analytics.db_path)
        
        query = '''
            SELECT 
                DATE(qa.timestamp) as date,
                AVG(CAST(qa.correct AS FLOAT)) as accuracy,
                AVG(qa.time_taken_seconds) as avg_time,
                COUNT(*) as attempts
            FROM question_attempts qa
            WHERE qa.timestamp > datetime('now', '-30 days')
            GROUP BY DATE(qa.timestamp)
            ORDER BY date
        '''
        
        daily_stats = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(daily_stats) == 0:
            st.warning("No data available for learning patterns")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['accuracy'],
            mode='lines+markers',
            name='Daily Accuracy',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Learning Progress Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy Rate",
            yaxis=dict(tickformat='.0%'),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def difficulty_adjustment_analysis(self):
        """Analyze optimal difficulty levels"""
        conn = sqlite3.connect(self.analytics.db_path)
        
        query = '''
            SELECT 
                qa.difficulty,
                AVG(CAST(qa.correct AS FLOAT)) as accuracy,
                AVG(qa.time_taken_seconds) as avg_time,
                COUNT(*) as attempts
            FROM question_attempts qa
            GROUP BY qa.difficulty
            ORDER BY qa.difficulty
        '''
        
        difficulty_stats = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(difficulty_stats) == 0:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_accuracy = px.bar(
                difficulty_stats, 
                x='difficulty', 
                y='accuracy',
                title='Accuracy by Difficulty Level',
                color='accuracy',
                color_continuous_scale='RdYlGn'
            )
            fig_accuracy.update_layout(height=350)
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        with col2:
            fig_time = px.line(
                difficulty_stats, 
                x='difficulty', 
                y='avg_time',
                title='Average Time by Difficulty',
                markers=True
            )
            fig_time.update_layout(height=350)
            st.plotly_chart(fig_time, use_container_width=True)
        
        # AI Recommendation
        optimal_difficulty = difficulty_stats.loc[
            (difficulty_stats['accuracy'] >= 0.6) & (difficulty_stats['accuracy'] <= 0.8)
        ]['difficulty'].values
        
        if len(optimal_difficulty) > 0:
            st.info(f"🤖 **AI Recommendation**: Optimal difficulty level is **{optimal_difficulty[0]}** for maximum learning efficiency (60-80% accuracy rate)")
    
    def topic_performance_heatmap(self):
        """Create topic performance heatmap"""
        conn = sqlite3.connect(self.analytics.db_path)
        
        query = '''
            SELECT 
                qa.question_type,
                qa.difficulty,
                AVG(CAST(qa.correct AS FLOAT)) as accuracy,
                COUNT(*) as attempts
            FROM question_attempts qa
            GROUP BY qa.question_type, qa.difficulty
        '''
        
        performance_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(performance_data) == 0:
            return
        
        # Pivot for heatmap
        heatmap_data = performance_data.pivot(
            index='question_type', 
            columns='difficulty', 
            values='accuracy'
        ).fillna(0)
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Difficulty Level", y="Topic", color="Accuracy"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlGn',
            aspect="auto"
        )
        
        fig.update_layout(
            title="Topic Performance Heatmap",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def student_segmentation(self):
        """Segment students based on performance patterns"""
        conn = sqlite3.connect(self.analytics.db_path)
        
        query = '''
            SELECT 
                ss.user_id,
                AVG(CAST(qa.correct AS FLOAT)) as avg_accuracy,
                AVG(qa.time_taken_seconds) as avg_time,
                COUNT(qa.id) as total_attempts,
                COUNT(DISTINCT ss.session_id) as total_sessions
            FROM student_sessions ss
            JOIN question_attempts qa ON ss.session_id = qa.session_id
            GROUP BY ss.user_id
            HAVING total_attempts >= 10
        '''
        
        student_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(student_data) < 10:
            st.warning("Need more student data for segmentation analysis")
            return
        
        # Prepare features for clustering
        features = student_data[['avg_accuracy', 'avg_time', 'total_attempts']].values
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        student_data['segment'] = kmeans.fit_predict(features)
        
        # Label segments
        segment_labels = {
            0: 'Fast Learners',
            1: 'Struggling Students', 
            2: 'Average Performers',
            3: 'High Achievers'
        }
        
        student_data['segment_label'] = student_data['segment'].map(segment_labels)
        
        # Visualization
        fig = px.scatter(
            student_data,
            x='avg_accuracy',
            y='avg_time', 
            size='total_attempts',
            color='segment_label',
            title='Student Performance Segmentation',
            labels={'avg_accuracy': 'Average Accuracy', 'avg_time': 'Average Time (seconds)'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment summary
        segment_summary = student_data.groupby('segment_label').agg({
            'user_id': 'count',
            'avg_accuracy': 'mean',
            'avg_time': 'mean',
            'total_attempts': 'mean'
        }).round(2)
        segment_summary.columns = ['Students', 'Avg Accuracy', 'Avg Time (s)', 'Avg Attempts']
        
        st.subheader("Student Segments Summary")
        st.dataframe(segment_summary)
    
    def run_dashboard(self):
        """Main dashboard interface"""
        st.title("📊 Educational AI Analytics Dashboard")
        st.markdown("*Transform learning data into actionable insights*")
        
        # Sidebar controls
        st.sidebar.title("Dashboard Controls")
        
        # Initialize demo data button
        if st.sidebar.button("🔄 Generate Demo Data"):
            with st.spinner("Generating realistic student data..."):
                self.generate_demo_data()
            st.sidebar.success("Demo data generated!")
        
        # Date range selector
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=[datetime.now().date() - timedelta(days=30), datetime.now().date()],
            max_value=datetime.now().date()
        )
        
        # Main dashboard
        st.markdown("---")
        
        # KPI Cards
        st.subheader("📈 Key Performance Indicators")
        self.create_kpi_cards()
        
        st.markdown("---")
        
        # Charts and Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Learning Progress")
            self.learning_patterns_chart()
        
        with col2:
            st.subheader("🎯 Topic Performance") 
            self.topic_performance_heatmap()
        
        st.markdown("---")
        
        # Difficulty Analysis
        st.subheader("🔧 AI-Powered Difficulty Adjustment")
        self.difficulty_adjustment_analysis()
        
        st.markdown("---")
        
        # Student Segmentation
        st.subheader("👥 Student Performance Segmentation")
        self.student_segmentation()
        
        # Individual Student Analysis
        st.markdown("---")
        st.subheader("🔍 Individual Student Insights")
        
        conn = sqlite3.connect(self.analytics.db_path)
        students = pd.read_sql_query("SELECT DISTINCT user_id FROM student_sessions LIMIT 20", conn)
        conn.close()
        
        if len(students) > 0:
            selected_student = st.selectbox("Select Student:", students['user_id'].tolist())
            
            if st.button("Generate AI Insights"):
                insights = self.analytics.generate_learning_insights(selected_student)
                
                if 'learning_trend' in insights:
                    trend = insights['learning_trend']
                    st.info(f"📈 **Learning Trend**: Student is {trend['direction']} (confidence: {trend['confidence']:.1%})")
                
                if 'difficulty_recommendation' in insights:
                    opt_diff = insights['difficulty_recommendation']['current_optimal']
                    if opt_diff:
                        st.success(f"🎯 **Recommended Difficulty**: Level {opt_diff} for optimal challenge")
                
                if 'topic_analysis' in insights:
                    analysis = insights['topic_analysis']
                    if analysis['weak_areas']:
                        st.warning(f"⚠️ **Focus Areas**: {', '.join(analysis['weak_areas'])}")
                    if analysis['strong_areas']:
                        st.success(f"✅ **Strengths**: {', '.join(analysis['strong_areas'])}")

if __name__ == "__main__":
    dashboard = EducationalAnalyticsDashboard()
    dashboard.run_dashboard()