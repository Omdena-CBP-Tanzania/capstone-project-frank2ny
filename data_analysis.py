import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def load_processed_data(file_path='data/processed_climate_data.csv'):
    """Load the processed climate data."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_visualizations(df):
    """Create various visualizations for climate data analysis."""
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Temperature Trends Over Time
    plt.figure(figsize=(15, 6))
    plt.plot(df['date'], df['Average_Temperature_C'], label='Average Temperature')
    plt.plot(df['date'], df['rolling_avg_temp'], label='12-month Rolling Average', linewidth=2)
    plt.title('Temperature Trends in Tanzania Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/temperature_trends.png')
    plt.close()
    
    # 2. Rainfall Patterns
    plt.figure(figsize=(15, 6))
    plt.plot(df['date'], df['Total_Rainfall_mm'], label='Monthly Rainfall')
    plt.plot(df['date'], df['rolling_avg_rainfall'], label='12-month Rolling Average', linewidth=2)
    plt.title('Rainfall Patterns in Tanzania Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/rainfall_patterns.png')
    plt.close()
    
    # 3. Seasonal Box Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.boxplot(x='season', y='Average_Temperature_C', data=df, ax=ax1)
    ax1.set_title('Temperature Distribution by Season')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Temperature (°C)')
    
    sns.boxplot(x='season', y='Total_Rainfall_mm', data=df, ax=ax2)
    ax2.set_title('Rainfall Distribution by Season')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Rainfall (mm)')
    
    plt.tight_layout()
    plt.savefig('plots/seasonal_distributions.png')
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['Average_Temperature_C', 'Total_Rainfall_mm', 
                   'Max_Temperature_C', 'Min_Temperature_C', 'temp_range']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Climate Variables')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    # 5. Seasonal Decomposition
    # Temperature decomposition
    temp_decomposition = seasonal_decompose(df['Average_Temperature_C'], period=12)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    temp_decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed Temperature')
    temp_decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    temp_decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    temp_decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    plt.savefig('plots/temperature_decomposition.png')
    plt.close()

def generate_statistical_summary(df):
    """Generate statistical summary of the climate data."""
    summary = {
        'Temperature Statistics': df[['Average_Temperature_C', 'Max_Temperature_C', 'Min_Temperature_C']].describe(),
        'Rainfall Statistics': df['Total_Rainfall_mm'].describe(),
        'Seasonal Averages': df.groupby('season')[['Average_Temperature_C', 'Total_Rainfall_mm']].mean()
    }
    
    # Save summary to text file
    with open('data/statistical_summary.txt', 'w') as f:
        f.write("Statistical Summary of Tanzania Climate Data\n")
        f.write("=" * 50 + "\n\n")
        
        for title, data in summary.items():
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            f.write(str(data))
            f.write("\n\n")

def main():
    print("Starting data analysis...")
    df = load_processed_data()
    if df is not None:
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(df)
        
        # Generate statistical summary
        print("Generating statistical summary...")
        generate_statistical_summary(df)
        
        print("\nAnalysis complete! Check the 'plots' directory for visualizations")
        print("and 'data/statistical_summary.txt' for detailed statistics.")

if __name__ == "__main__":
    main() 