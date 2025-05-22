import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta

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
    return temp_model, rain_model, scaler

# Load data and models
df = load_data()
temp_model, rain_model, scaler = load_models()

# Title and description
st.title("🌍 Tanzania Climate Analysis and Prediction")
st.markdown("""
This application provides insights into Tanzania's climate patterns and predicts future temperature and rainfall.
Use the sidebar to navigate through different sections of the analysis.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Historical Analysis", "Predictions"])

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
    
    # Display recent trends
    st.subheader("Recent Climate Trends")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Temperature trend
    ax1.plot(df['date'], df['Average_Temperature_C'], label='Temperature')
    ax1.plot(df['date'], df['rolling_avg_temp'], label='12-month Average', linewidth=2)
    ax1.set_title('Temperature Trend')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    
    # Rainfall trend
    ax2.plot(df['date'], df['Total_Rainfall_mm'], label='Rainfall')
    ax2.plot(df['date'], df['rolling_avg_rainfall'], label='12-month Average', linewidth=2)
    ax2.set_title('Rainfall Trend')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Historical Analysis":
    st.header("Historical Climate Analysis")
    
    # Seasonal analysis
    st.subheader("Seasonal Patterns")
    season = st.selectbox("Select Season", ['DJF', 'MAM', 'JJA', 'SON'])
    
    season_data = df[df['season'] == season]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Temperature", 
                 f"{season_data['Average_Temperature_C'].mean():.1f}°C")
        st.metric("Temperature Range", 
                 f"{season_data['Average_Temperature_C'].max() - season_data['Average_Temperature_C'].min():.1f}°C")
    
    with col2:
        st.metric("Average Rainfall", 
                 f"{season_data['Total_Rainfall_mm'].mean():.1f} mm")
        st.metric("Rainfall Range", 
                 f"{season_data['Total_Rainfall_mm'].max() - season_data['Total_Rainfall_mm'].min():.1f} mm")
    
    # Correlation analysis
    st.subheader("Variable Correlations")
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = ['Average_Temperature_C', 'Total_Rainfall_mm', 
                   'Max_Temperature_C', 'Min_Temperature_C']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

else:  # Predictions page
    st.header("Climate Predictions")
    
    # Input for prediction
    st.subheader("Make Predictions")
    
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
    feature_order = [
        'Max_Temperature_C', 'Min_Temperature_C', 'rolling_avg_rainfall',
        'rolling_avg_temp', 'temp_range', 'temp_lag_1', 'temp_lag_2',
        'temp_lag_3', 'temp_lag_6', 'temp_lag_12', 'rainfall_lag_1',
        'rainfall_lag_2', 'rainfall_lag_3', 'rainfall_lag_6',
        'rainfall_lag_12', 'month_sin', 'month_cos'
    ]
    
    # Create input features
    input_features = pd.DataFrame({
        'Max_Temperature_C': [last_12_months['Max_Temperature_C'].iloc[-1]],
        'Min_Temperature_C': [last_12_months['Min_Temperature_C'].iloc[-1]],
        'rolling_avg_rainfall': [last_12_months['rolling_avg_rainfall'].iloc[-1]],
        'rolling_avg_temp': [last_12_months['rolling_avg_temp'].iloc[-1]],
        'temp_range': [last_12_months['temp_range'].iloc[-1]],
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
        'month_cos': [np.cos(2 * np.pi * (last_date.month)/12)]
    })
    
    # Ensure features are in the correct order
    input_features = input_features[feature_order]
    
    # Convert to numpy array to avoid feature name issues with scaler
    input_array = input_features.values
    
    # Scale features
    input_scaled = scaler.transform(input_array)
    
    # Make predictions
    temp_pred = temp_model.predict(input_scaled)[0]
    rain_pred = rain_model.predict(input_scaled)[0]
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Temperature plot
    ax1.plot(last_12_months['date'], last_12_months['Average_Temperature_C'], label='Historical')
    ax1.axhline(y=temp_pred, color='r', linestyle='--', label='Predicted')
    ax1.set_title('Temperature: Historical vs Predicted')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    
    # Rainfall plot
    ax2.plot(last_12_months['date'], last_12_months['Total_Rainfall_mm'], label='Historical')
    ax2.axhline(y=rain_pred, color='r', linestyle='--', label='Predicted')
    ax2.set_title('Rainfall: Historical vs Predicted')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Climate Change Analysis in Tanzania | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True) 