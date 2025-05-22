import pandas as pd
import os

def load_existing_data():
    
    try:
        data_path = 'data/tanzania_climate_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"Successfully loaded data with {len(df)} rows")
            return df
        else:
            print(f"Data file not found at {data_path}")
            return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def main():
    print("Loading existing climate data...")
    df = load_existing_data()
    if df is not None:
        print("\nData Summary:")
        print(df.describe())
        print("\nFirst few rows:")
        print(df.head())

if __name__ == "__main__":
    main() 