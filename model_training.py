import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_processed_data(file_path='data/processed_climate_data.csv'):
    """Load the processed climate data."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_data(df):
    """Prepare data for model training."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'temp_lag_{lag}'] = df['Average_Temperature_C'].shift(lag)
        df[f'rainfall_lag_{lag}'] = df['Total_Rainfall_mm'].shift(lag)
    
    # Create seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    # Drop rows with NaN values (from lag features)
    df = df.dropna()
    
    return df

def train_models(df):
    """Train models for temperature and rainfall prediction."""
    # Features for prediction
    feature_cols = [col for col in df.columns if col not in 
                   ['date', 'Year', 'Month', 'Average_Temperature_C', 
                    'Total_Rainfall_mm', 'season']]
    
    # Prepare data for temperature prediction
    X_temp = df[feature_cols]
    y_temp = df['Average_Temperature_C']
    
    # Prepare data for rainfall prediction
    X_rain = df[feature_cols]
    y_rain = df['Total_Rainfall_mm']
    
    # Split data
    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    X_rain_train, X_rain_test, y_rain_train, y_rain_test = train_test_split(
        X_rain, y_rain, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_temp_train_scaled = scaler.fit_transform(X_temp_train)
    X_temp_test_scaled = scaler.transform(X_temp_test)
    X_rain_train_scaled = scaler.fit_transform(X_rain_train)
    X_rain_test_scaled = scaler.transform(X_rain_test)
    
    # Train temperature model
    temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    temp_model.fit(X_temp_train_scaled, y_temp_train)
    
    # Train rainfall model
    rain_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rain_model.fit(X_rain_train_scaled, y_rain_train)
    
    # Evaluate models
    models = {
        'temperature': {
            'model': temp_model,
            'X_train': X_temp_train_scaled,
            'X_test': X_temp_test_scaled,
            'y_train': y_temp_train,
            'y_test': y_temp_test
        },
        'rainfall': {
            'model': rain_model,
            'X_train': X_rain_train_scaled,
            'X_test': X_rain_test_scaled,
            'y_train': y_rain_train,
            'y_test': y_rain_test
        }
    }
    
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(temp_model, 'models/temperature_model.joblib')
    joblib.dump(rain_model, 'models/rainfall_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return models

def evaluate_models(models):
    """Evaluate model performance and save results."""
    results = {}
    
    for target, model_data in models.items():
        model = model_data['model']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[target] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    # Save results
    with open('data/model_evaluation.txt', 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for target, metrics in results.items():
            f.write(f"{target.title()} Prediction Model\n")
            f.write("-" * len(f"{target.title()} Prediction Model") + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
    
    return results

def main():
    print("Starting model training...")
    df = load_processed_data()
    if df is not None:
        # Prepare data
        print("Preparing data...")
        df = prepare_data(df)
        
        # Train models
        print("Training models...")
        models = train_models(df)
        
        # Evaluate models
        print("Evaluating models...")
        results = evaluate_models(models)
        
        print("\nTraining complete! Check 'data/model_evaluation.txt' for results")
        print("Models have been saved in the 'models' directory")

if __name__ == "__main__":
    main() 