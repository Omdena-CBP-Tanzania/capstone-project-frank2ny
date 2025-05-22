import pandas as pd
import requests
import os
from datetime import datetime

def download_climate_data():
    """
    Download climate data for Tanzania from public sources.
    This is a placeholder function - you'll need to replace the URL with actual data source.
    """
    # Example URL (replace with actual data source)
    url = "https://climateknowledgeportal.worldbank.org/api/data/get-download-data"
    
    # Parameters for Tanzania
    params = {
        'country': 'Tanzania',
        'variable': 'temperature,precipitation',
        'start_date': '1960-01-01',
        'end_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Save the data
            os.makedirs('data', exist_ok=True)
            with open('data/climate_data.csv', 'wb') as f:
                f.write(response.content)
            print("Data downloaded successfully!")
        else:
            print(f"Failed to download data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading data: {str(e)}")

def main():
    print("Starting data collection...")
    download_climate_data()

if __name__ == "__main__":
    main() 