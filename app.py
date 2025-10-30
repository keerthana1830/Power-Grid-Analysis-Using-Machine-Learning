"""
Power Grid Analysis Dashboard
Interactive Streamlit application for power grid ML analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from ml_engine import PowerGridMLEngine
from data_generator import generate_power_grid_data
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Power Grid Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-success { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-info { background-color: #17a2b8; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'ml_engine' not in st.session_state:
        st.session_state.ml_engine = PowerGridMLEngine()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

def display_header():
    """Display the main header and description"""
    st.markdown('<h1 class="main-header">⚡ Power Grid Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Intelligent ML-powered analysis for power grid stability, load prediction, and anomaly detection
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_status_indicators():
    """Display system status indicators"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_class = "status-success" if st.session_state.data_uploaded else "status-warning"
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="status-indicator {status_class}"></span>
            <strong>Data Uploaded</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_class = "status-success" if st.session_state.models_trained else "status-warning"
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="status-indicator {status_class}"></span>
            <strong>Models Trained</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_class = "status-success" if st.session_state.analysis_complete else "status-warning"
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="status-indicator {status_class}"></span>
            <strong>Analysis Complete</strong>
        </div>
        """, unsafe_allow_html=True)

def load_data():
    """Handle data loading (upload or sample data)"""
    st.sidebar.header("📊 Data Input")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload CSV File"]
    )
    
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload power grid data (CSV)",
            type=['csv'],
            help="Upload a CSV file with columns: Voltage, Current, Frequency, Power_Factor, Load, Phase_Angle, Stability"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.session_state.data_uploaded = True
                st.sidebar.success("✅ Data uploaded successfully!")
                return data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                return None
    
    else:  # Use sample data
        if st.sidebar.button("🔄 Generate Sample Data"):
            with st.spinner("Generating sample power grid data..."):
                data = generate_power_grid_data(n_samples=2000)
                st.session_state.data = data
                st.session_state.data_uploaded = True
                st.sidebar.success("✅ Sample data generated!")
                return data
    
    return st.session_state.data

def display_data_summary(data):
    """Display data summary and statistics"""
    if data is None:
        return
    
    st.header("📋 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        if 'Stability' in data.columns:
            stability_rate = data['Stability'].mean()
            st.metric("Stability Rate", f"{stability_rate:.1%}")
    with col4:
        if 'Load' in data.columns:
            avg_load = data['Load'].mean()
            st.metric("Avg Load (kW)", f"{avg_load:.1f}")
    
    # Data preview
    st.subheader("📊 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

def train_models(data):
    """Train ML models and display progress"""
    if data is None:
        return None
    
    if st.button("🚀 Train ML Models", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Train models with progress updates
            status_text.text("🔄 Initializing ML engine...")
            progress_bar.progress(20)
            
            status_text.text("🔄 Training models...")
            progress_bar.progress(50)
            
            results = st.session_state.ml_engine.train_all_models(data)
            progress_bar.progress(80)
            
            status_text.text("💾 Saving models...")
            st.session_state.ml_engine.save_models()
            progress_bar.progress(100)
            
            st.session_state.results = results
            st.session_state.models_trained = True
            st.session_state.analysis_complete = True
            
            status_text.text("✅ Training completed successfully!")
            st.success("🎉 All models trained and saved successfully!")
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            return None
    
    return st.session_state.results

def create_load_forecast_plot(results):
    """Create load forecast visualization"""
    if 'load' not in results:
        return None
    
    actual = results['load']['actual']
    predicted = results['load']['predictions']
    
    fig = go.Figure()
    
    # Add actual vs predicted scatter
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Add perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Load Demand: Actual vs Predicted",
        xaxis_title="Actual Load (kW)",
        yaxis_title="Predicted Load (kW)",
        height=400
    )
    
    return fig

def create_stability_distribution_plot(data):
    """Create stability distribution pie chart"""
    if 'Stability' not in data.columns:
        return None
    
    stability_counts = data['Stability'].value_counts()
    labels = ['Unstable', 'Stable']
    values = [stability_counts.get(0, 0), stability_counts.get(1, 0)]
    colors = ['#ff6b6b', '#51cf66']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4
    )])
    
    fig.update_layout(
        title="Grid Stability Distribution",
        height=400
    )
    
    return fig

def create_confusion_matrix_plot(results):
    """Create confusion matrix heatmap"""
    if 'stability' not in results:
        return None
    
    cm = results['stability']['confusion_matrix']
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Unstable', 'Stable'],
        y=['Unstable', 'Stable'],
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        title="Stability Prediction Confusion Matrix",
        height=400
    )
    
    return fig

def create_feature_importance_plot(results):
    """Create feature importance bar chart"""
    if 'stability' not in results:
        return None
    
    feature_imp = results['stability']['feature_importance']
    features = list(feature_imp.keys())
    importance = list(feature_imp.values())
    
    fig = go.Figure([go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='lightblue'
    )])
    
    fig.update_layout(
        title="Feature Importance for Stability Prediction",
        xaxis_title="Importance Score",
        height=400
    )
    
    return fig

def create_anomaly_plot(data, results):
    """Create anomaly detection scatter plot"""
    if 'anomaly' not in results:
        return None
    
    anomaly_labels = results['anomaly']['anomaly_labels']
    
    # Create scatter plot with voltage vs current, colored by anomaly
    colors = ['red' if label == -1 else 'blue' for label in anomaly_labels]
    labels = ['Anomaly' if label == -1 else 'Normal' for label in anomaly_labels]
    
    fig = go.Figure()
    
    # Normal points
    normal_mask = anomaly_labels == 1
    fig.add_trace(go.Scatter(
        x=data['Voltage'][normal_mask],
        y=data['Current'][normal_mask],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Anomaly points
    anomaly_mask = anomaly_labels == -1
    fig.add_trace(go.Scatter(
        x=data['Voltage'][anomaly_mask],
        y=data['Current'][anomaly_mask],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8)
    ))
    
    fig.update_layout(
        title="Anomaly Detection: Voltage vs Current",
        xaxis_title="Voltage (V)",
        yaxis_title="Current (A)",
        height=400
    )
    
    return fig

def display_visualizations(data, results):
    """Display all analysis visualizations"""
    if results is None:
        st.info("👆 Please train the models first to see visualizations")
        return
    
    st.header("📊 Analysis Visualizations")
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Load forecast plot
        load_fig = create_load_forecast_plot(results)
        if load_fig:
            st.plotly_chart(load_fig, use_container_width=True)
        
        # Confusion matrix
        cm_fig = create_confusion_matrix_plot(results)
        if cm_fig:
            st.plotly_chart(cm_fig, use_container_width=True)
    
    with col2:
        # Stability distribution
        stability_fig = create_stability_distribution_plot(data)
        if stability_fig:
            st.plotly_chart(stability_fig, use_container_width=True)
        
        # Feature importance
        feature_fig = create_feature_importance_plot(results)
        if feature_fig:
            st.plotly_chart(feature_fig, use_container_width=True)
    
    # Anomaly detection plot (full width)
    anomaly_fig = create_anomaly_plot(data, results)
    if anomaly_fig:
        st.plotly_chart(anomaly_fig, use_container_width=True)

def display_performance_metrics(results):
    """Display model performance metrics"""
    if results is None:
        return
    
    st.header("📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Stability Classification")
        if 'stability' in results:
            st.metric("Accuracy", f"{results['stability']['accuracy']:.3f}")
            st.metric("F1-Score", f"{results['stability']['f1_score']:.3f}")
    
    with col2:
        st.subheader("📊 Load Prediction")
        if 'load' in results:
            st.metric("RMSE", f"{results['load']['rmse']:.2f} kW")
            st.metric("MAE", f"{results['load']['mae']:.2f} kW")
            st.metric("R² Score", f"{results['load']['r2_score']:.3f}")
    
    with col3:
        st.subheader("🚨 Anomaly Detection")
        if 'anomaly' in results:
            st.metric("Anomaly Rate", f"{results['anomaly']['anomaly_percentage']:.1f}%")
            st.metric("Normal Points", f"{results['anomaly']['normal_count']:,}")
            st.metric("Anomalies", f"{results['anomaly']['anomaly_count']:,}")

def display_insights(data, results):
    """Display automated insights"""
    if results is None:
        return
    
    st.header("💡 Automated Insights")
    
    insights = st.session_state.ml_engine.generate_insights(data)
    
    for insight in insights:
        st.info(insight)

def create_download_report(data, results):
    """Create downloadable analysis report"""
    if results is None:
        return None
    
    # Create a comprehensive report
    report = []
    report.append("# Power Grid Analysis Report\n")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Data summary
    report.append("## Data Summary\n")
    report.append(f"- Total Records: {len(data):,}\n")
    report.append(f"- Features: {len(data.columns)}\n")
    if 'Stability' in data.columns:
        stability_rate = data['Stability'].mean()
        report.append(f"- Stability Rate: {stability_rate:.1%}\n")
    report.append("\n")
    
    # Model performance
    report.append("## Model Performance\n")
    if 'stability' in results:
        report.append(f"- Stability Classification Accuracy: {results['stability']['accuracy']:.3f}\n")
        report.append(f"- Stability Classification F1-Score: {results['stability']['f1_score']:.3f}\n")
    
    if 'load' in results:
        report.append(f"- Load Prediction RMSE: {results['load']['rmse']:.2f} kW\n")
        report.append(f"- Load Prediction R²: {results['load']['r2_score']:.3f}\n")
    
    if 'anomaly' in results:
        report.append(f"- Anomaly Detection Rate: {results['anomaly']['anomaly_percentage']:.1f}%\n")
    
    report.append("\n")
    
    # Insights
    report.append("## Key Insights\n")
    insights = st.session_state.ml_engine.generate_insights(data)
    for insight in insights:
        report.append(f"- {insight}\n")
    
    return "".join(report)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    display_header()
    
    # Status indicators
    display_status_indicators()
    
    st.markdown("---")
    
    # Data loading
    data = load_data()
    
    if data is not None:
        # Data summary
        display_data_summary(data)
        
        st.markdown("---")
        
        # Model training
        st.header("🤖 Machine Learning Models")
        results = train_models(data)
        
        if results is not None:
            st.markdown("---")
            
            # Visualizations
            display_visualizations(data, results)
            
            st.markdown("---")
            
            # Performance metrics
            display_performance_metrics(results)
            
            st.markdown("---")
            
            # Insights
            display_insights(data, results)
            
            st.markdown("---")
            
            # Download report
            st.header("📥 Download Report")
            report_content = create_download_report(data, results)
            if report_content:
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=report_content,
                    file_name=f"power_grid_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    else:
        st.info("👆 Please load data using the sidebar to begin analysis")

if __name__ == "__main__":
    main()