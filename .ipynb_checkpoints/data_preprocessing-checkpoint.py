import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path='data/climate_data.csv'):
    """Load the climate data from CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the climate data:
    - Handle missing values
    - Convert date columns
    - Create additional features
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date columns to datetime
    date_columns = df.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            continue
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Create time-based features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].apply(lambda x: 
            'DJF' if x in [12,1,2] else
            'MAM' if x in [3,4,5] else
            'JJA' if x in [6,7,8] else 'SON'
        )
    
    return df

def save_processed_data(df, output_path='data/processed_climate_data.csv'):
    """Save the processed data to a CSV file."""
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    else:
        print("No data to save")

def main():
    print("Starting data preprocessing...")
    df = load_data()
    processed_df = preprocess_data(df)
    save_processed_data(processed_df)

if __name__ == "__main__":
    main() 