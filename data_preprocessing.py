import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path='data/tanzania_climate_data.csv'):
    """Load the Tanzania climate data from CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the Tanzania climate data:
    - Handle missing values
    - Create date column from Year and Month
    - Add seasonal features
    - Calculate rolling averages
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Create date column
    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    
    # Handle missing values
    numeric_columns = ['Average_Temperature_C', 'Total_Rainfall_mm', 
                      'Max_Temperature_C', 'Min_Temperature_C']
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Add seasonal features
    df['season'] = df['Month'].apply(lambda x: 
        'DJF' if x in [12,1,2] else
        'MAM' if x in [3,4,5] else
        'JJA' if x in [6,7,8] else 'SON'
    )
    
    # Calculate rolling averages (12-month window)
    df['rolling_avg_temp'] = df['Average_Temperature_C'].rolling(window=12).mean()
    df['rolling_avg_rainfall'] = df['Total_Rainfall_mm'].rolling(window=12).mean()
    
    # Add temperature range
    df['temp_range'] = df['Max_Temperature_C'] - df['Min_Temperature_C']
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

def save_processed_data(df, output_path='data/processed_climate_data.csv'):
    """Save the processed data to a CSV file."""
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        print("\nProcessed Data Summary:")
        print(df.describe())
    else:
        print("No data to save")

def main():
    print("Starting data preprocessing...")
    df = load_data()
    processed_df = preprocess_data(df)
    save_processed_data(processed_df)

if __name__ == "__main__":
    main() 