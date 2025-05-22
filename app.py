import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Tanzania Climate Analysis",
    page_icon="🌍",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_climate_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_models():
    temp_model = joblib.load('models/temperature_model.joblib')
    rain_model = joblib.load('models/rainfall_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_order = joblib.load('models/feature_order.joblib')
    return temp_model, rain_model, scaler, feature_order

# Load data and models
df = load_data()
temp_model, rain_model, scaler, feature_order = load_models()

# Title and description
st.title("🌍 Tanzania Climate Analysis and Prediction")
st.markdown("""
This application provides insights into Tanzania's climate patterns and predicts future temperature and rainfall.
Use the sidebar to navigate through different sections of the analysis.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Historical Analysis", "Predictions", "About"])

if page == "Overview":
    st.header("Overview of Tanzania's Climate")
    
    # Display key statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Temperature", 
                 f"{df['Average_Temperature_C'].mean():.1f}°C",
                 f"{df['Average_Temperature_C'].max() - df['Average_Temperature_C'].min():.1f}°C range")
    
    with col2:
        st.metric("Average Rainfall", 
                 f"{df['Total_Rainfall_mm'].mean():.1f} mm",
                 f"{df['Total_Rainfall_mm'].max() - df['Total_Rainfall_mm'].min():.1f} mm range")
    
    with col3:
        st.metric("Data Period", 
                 f"{df['date'].min().year} - {df['date'].max().year}",
                 f"{len(df)} months of data")
    
    # Interactive time series plot
    st.subheader("Climate Trends Over Time")
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add temperature trace
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Average_Temperature_C'],
        name='Temperature',
        line=dict(color='red')
    ))
    
    # Add rainfall trace
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Total_Rainfall_mm'],
        name='Rainfall',
        line=dict(color='blue'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Temperature and Rainfall Trends',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        yaxis2=dict(
            title='Rainfall (mm)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    
    # Create seasonal box plots
    fig = px.box(df, x='season', y='Average_Temperature_C', 
                 title='Temperature Distribution by Season',
                 color='season')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(df, x='season', y='Total_Rainfall_mm',
                 title='Rainfall Distribution by Season',
                 color='season')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Historical Analysis":
    st.header("Historical Climate Analysis")
    
    # Date range selector
    st.subheader("Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['date'].min())
    with col2:
        end_date = st.date_input("End Date", df['date'].max())
    
    # Filter data based on date range
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df[mask]
    
    # Display statistics for selected period
    st.subheader("Statistics for Selected Period")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Temperature", 
                 f"{filtered_df['Average_Temperature_C'].mean():.1f}°C",
                 f"{filtered_df['Average_Temperature_C'].max() - filtered_df['Average_Temperature_C'].min():.1f}°C range")
    
    with col2:
        st.metric("Average Rainfall", 
                 f"{filtered_df['Total_Rainfall_mm'].mean():.1f} mm",
                 f"{filtered_df['Total_Rainfall_mm'].max() - filtered_df['Total_Rainfall_mm'].min():.1f} mm range")
    
    with col3:
        st.metric("Number of Months", 
                 len(filtered_df),
                 f"From {filtered_df['date'].min().strftime('%Y-%m')} to {filtered_df['date'].max().strftime('%Y-%m')}")
    
    # Correlation analysis
    st.subheader("Variable Correlations")
    numeric_cols = ['Average_Temperature_C', 'Total_Rainfall_mm', 
                   'Max_Temperature_C', 'Min_Temperature_C']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols,
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly patterns
    st.subheader("Monthly Patterns")
    monthly_avg = filtered_df.groupby('Month').agg({
        'Average_Temperature_C': 'mean',
        'Total_Rainfall_mm': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_avg['Month'],
        y=monthly_avg['Average_Temperature_C'],
        name='Temperature',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=monthly_avg['Month'],
        y=monthly_avg['Total_Rainfall_mm'],
        name='Rainfall',
        line=dict(color='blue'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Average Monthly Temperature and Rainfall',
        xaxis_title='Month',
        yaxis_title='Temperature (°C)',
        yaxis2=dict(
            title='Rainfall (mm)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Predictions":
    st.header("Climate Predictions")
    
    # Add tabs for different prediction views
    pred_tab1, pred_tab2 = st.tabs(["Next Month Prediction", "Custom Prediction"])
    
    with pred_tab1:
        st.subheader("Next Month's Climate Prediction")
        
        # Get last 12 months of data for lag features
        last_date = df['date'].max()
        last_12_months = df[df['date'] >= last_date - pd.DateOffset(months=12)].copy()
        
        # Calculate any missing features
        if 'temp_range' not in last_12_months.columns:
            last_12_months['temp_range'] = last_12_months['Max_Temperature_C'] - last_12_months['Min_Temperature_C']
        
        if 'rolling_avg_temp' not in last_12_months.columns:
            last_12_months['rolling_avg_temp'] = last_12_months['Average_Temperature_C'].rolling(window=12).mean()
        
        if 'rolling_avg_rainfall' not in last_12_months.columns:
            last_12_months['rolling_avg_rainfall'] = last_12_months['Total_Rainfall_mm'].rolling(window=12).mean()
        
        # Create lag features if they don't exist
        for lag in [1, 2, 3, 6, 12]:
            if f'temp_lag_{lag}' not in last_12_months.columns:
                last_12_months[f'temp_lag_{lag}'] = last_12_months['Average_Temperature_C'].shift(lag)
            if f'rainfall_lag_{lag}' not in last_12_months.columns:
                last_12_months[f'rainfall_lag_{lag}'] = last_12_months['Total_Rainfall_mm'].shift(lag)
        
        # Create seasonal features
        last_12_months['month_sin'] = np.sin(2 * np.pi * last_12_months['Month']/12)
        last_12_months['month_cos'] = np.cos(2 * np.pi * last_12_months['Month']/12)
        
        # Create input features with all required columns in the correct order
        input_features = pd.DataFrame({
            'Max_Temperature_C': [last_12_months['Max_Temperature_C'].iloc[-1]],
            'Min_Temperature_C': [last_12_months['Min_Temperature_C'].iloc[-1]],
            'temp_lag_1': [last_12_months['Average_Temperature_C'].iloc[-1]],
            'temp_lag_2': [last_12_months['Average_Temperature_C'].iloc[-2]],
            'temp_lag_3': [last_12_months['Average_Temperature_C'].iloc[-3]],
            'temp_lag_6': [last_12_months['Average_Temperature_C'].iloc[-6]],
            'temp_lag_12': [last_12_months['Average_Temperature_C'].iloc[-12]],
            'rainfall_lag_1': [last_12_months['Total_Rainfall_mm'].iloc[-1]],
            'rainfall_lag_2': [last_12_months['Total_Rainfall_mm'].iloc[-2]],
            'rainfall_lag_3': [last_12_months['Total_Rainfall_mm'].iloc[-3]],
            'rainfall_lag_6': [last_12_months['Total_Rainfall_mm'].iloc[-6]],
            'rainfall_lag_12': [last_12_months['Total_Rainfall_mm'].iloc[-12]],
            'month_sin': [np.sin(2 * np.pi * (last_date.month)/12)],
            'month_cos': [np.cos(2 * np.pi * (last_date.month)/12)],
            'rolling_avg_temp': [last_12_months['rolling_avg_temp'].iloc[-1]],
            'rolling_avg_rainfall': [last_12_months['rolling_avg_rainfall'].iloc[-1]],
            'temp_range': [last_12_months['temp_range'].iloc[-1]]
        })
        
        # Ensure features are in the correct order
        input_features = input_features[feature_order]
        
        # Scale features while maintaining feature names
        input_scaled = pd.DataFrame(
            scaler.transform(input_features),
            columns=feature_order,
            index=input_features.index
        )
        
        # Pass as numpy array to the model
        temp_pred = temp_model.predict(input_scaled.values)[0]
        rain_pred = rain_model.predict(input_scaled.values)[0]
        
        # Display predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Temperature", 
                     f"{temp_pred:.1f}°C",
                     f"{temp_pred - last_12_months['Average_Temperature_C'].iloc[-1]:.1f}°C from last month")
        
        with col2:
            st.metric("Predicted Rainfall", 
                     f"{rain_pred:.1f} mm",
                     f"{rain_pred - last_12_months['Total_Rainfall_mm'].iloc[-1]:.1f} mm from last month")
        
        # Display historical vs predicted
        st.subheader("Historical vs Predicted")
        
        # Create Plotly figure for temperature
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=last_12_months['date'],
            y=last_12_months['Average_Temperature_C'],
            name='Historical Temperature',
            line=dict(color='blue')
        ))
        fig_temp.add_trace(go.Scatter(
            x=[last_12_months['date'].iloc[-1] + pd.DateOffset(months=1)],
            y=[temp_pred],
            name='Predicted Temperature',
            mode='markers',
            marker=dict(color='red', size=10)
        ))
        
        fig_temp.update_layout(
            title='Temperature: Historical vs Predicted',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Create Plotly figure for rainfall
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Scatter(
            x=last_12_months['date'],
            y=last_12_months['Total_Rainfall_mm'],
            name='Historical Rainfall',
            line=dict(color='green')
        ))
        fig_rain.add_trace(go.Scatter(
            x=[last_12_months['date'].iloc[-1] + pd.DateOffset(months=1)],
            y=[rain_pred],
            name='Predicted Rainfall',
            mode='markers',
            marker=dict(color='orange', size=10)
        ))
        
        fig_rain.update_layout(
            title='Rainfall: Historical vs Predicted',
            xaxis_title='Date',
            yaxis_title='Rainfall (mm)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_rain, use_container_width=True)
    
    with pred_tab2:
        st.subheader("Custom Climate Prediction")
        
        # Add user input controls
        st.write("Adjust the parameters below to see how they affect the predictions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature inputs
            st.write("Temperature Parameters")
            max_temp = st.slider("Maximum Temperature (°C)", 
                               min_value=20.0, max_value=40.0, 
                               value=float(df['Max_Temperature_C'].mean()),
                               step=0.1)
            min_temp = st.slider("Minimum Temperature (°C)", 
                               min_value=15.0, max_value=30.0, 
                               value=float(df['Min_Temperature_C'].mean()),
                               step=0.1)
        
        with col2:
            # Rainfall inputs
            st.write("Rainfall Parameters")
            rainfall = st.slider("Previous Month Rainfall (mm)", 
                               min_value=0.0, max_value=300.0, 
                               value=float(df['Total_Rainfall_mm'].mean()),
                               step=1.0)
            month = st.selectbox("Month", range(1, 13), 
                               index=datetime.now().month - 1)
        
        # Calculate derived features
        temp_range = max_temp - min_temp
        month_sin = np.sin(2 * np.pi * month/12)
        month_cos = np.cos(2 * np.pi * month/12)
        
        # Create custom input features
        custom_input = pd.DataFrame({
            'Max_Temperature_C': [max_temp],
            'Min_Temperature_C': [min_temp],
            'temp_lag_1': [(max_temp + min_temp)/2],
            'temp_lag_2': [(max_temp + min_temp)/2],
            'temp_lag_3': [(max_temp + min_temp)/2],
            'temp_lag_6': [(max_temp + min_temp)/2],
            'temp_lag_12': [(max_temp + min_temp)/2],
            'rainfall_lag_1': [rainfall],
            'rainfall_lag_2': [rainfall],
            'rainfall_lag_3': [rainfall],
            'rainfall_lag_6': [rainfall],
            'rainfall_lag_12': [rainfall],
            'month_sin': [month_sin],
            'month_cos': [month_cos],
            'rolling_avg_temp': [(max_temp + min_temp)/2],
            'rolling_avg_rainfall': [rainfall],
            'temp_range': [temp_range]
        })
        
        # Ensure features are in the correct order
        custom_input = custom_input[feature_order]
        
        # Scale features while maintaining feature names
        custom_scaled = pd.DataFrame(
            scaler.transform(custom_input),
            columns=feature_order,
            index=custom_input.index
        )
        
        # Pass as numpy array to the model
        custom_temp_pred = temp_model.predict(custom_scaled.values)[0]
        custom_rain_pred = rain_model.predict(custom_scaled.values)[0]
        
        # Display predictions
        st.subheader("Custom Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Temperature", 
                     f"{custom_temp_pred:.1f}°C",
                     f"{custom_temp_pred - (max_temp + min_temp)/2:.1f}°C from input average")
        
        with col2:
            st.metric("Predicted Rainfall", 
                     f"{custom_rain_pred:.1f} mm",
                     f"{custom_rain_pred - rainfall:.1f} mm from input")
        
        # Display prediction explanation
        st.subheader("Prediction Explanation")
        st.write("""
        The prediction is based on:
        - Input temperature range: {:.1f}°C to {:.1f}°C
        - Input rainfall: {:.1f} mm
        - Selected month: {}
        - Historical patterns and seasonal trends
        """.format(min_temp, max_temp, rainfall, month))

else:  # About page
    st.header("About This Project")
    
    st.markdown("""
    ### Project Overview
    This application provides climate analysis and predictions for Tanzania, helping to understand and anticipate climate patterns in the region.
    
    ### Features
    - Historical climate data analysis
    - Interactive visualizations
    - Temperature and rainfall predictions
    - Seasonal pattern analysis
    
    ### Data Sources
    The data used in this analysis comes from historical climate records for Tanzania, including:
    - Temperature measurements
    - Rainfall measurements
    - Seasonal variations
    
    ### Methodology
    The predictions are made using machine learning models trained on historical data, taking into account:
    - Seasonal patterns
    - Historical trends
    - Multiple climate variables
    
    ### Contact
    For more information about this project, please contact the development team.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Climate Change Analysis in Tanzania | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True) 