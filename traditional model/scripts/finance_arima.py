import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Handle missing values and infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Take log of volume and standardize
    scaler = StandardScaler()
    df['Volume'] = np.log(df['Volume'])
    df['Volume_Scaled'] = scaler.fit_transform(df[['Volume']])
    
    return df

def find_optimal_sarima_params(data, exog=None):
    """Find optimal SARIMA parameters using grid search"""
    # Define parameter ranges
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    s = [5, 21]  # Consider weekly and monthly seasonality
    
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    print("Starting search for optimal SARIMA parameters...")
    
    for order in itertools.product(p, d, q):
        for seasonal_order in itertools.product(P, D, Q, s):
            try:
                model = SARIMAX(data,
                              exog=exog,
                              order=order,
                              seasonal_order=seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    print(f"Found better model: SARIMA{best_order}x{best_seasonal_order}")
                    print(f"AIC: {best_aic}")
            except:
                continue
    
    return best_order, best_seasonal_order

def check_stationarity(data):
    """Check time series stationarity"""
    data = data.dropna()
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result[1] < 0.05

def plot_original_data(data, save_path):
    """Plot original data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot price
    ax1.plot(data.index, data['Close'])
    ax1.set_title('Original Closing Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot volume
    ax2.plot(data.index, data['Volume'], color='orange')
    ax2.set_title('Trading Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_path, f'Original Data_timeseries_{timestamp}.png'))
    plt.close()

def plot_transformed_data(data, save_path):
    """Plot transformed data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot log price
    ax1.plot(data.index, data['Close'])
    ax1.set_title('Logarithmically Transformed Closing Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log Price')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot standardized volume
    ax2.plot(data.index, data['Volume_Scaled'], color='orange')
    ax2.set_title('Standardized Log Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Standardized Log Volume')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_path, f'Transformed Data_timeseries_{timestamp}.png'))
    plt.close()

def train_sarima_model(train_data, exog=None, order=None, seasonal_order=None):
    """Train SARIMA model"""
    train_data = train_data.dropna()
    
    if order is None or seasonal_order is None:
        # If parameters not specified, find optimal parameters using grid search
        order, seasonal_order = find_optimal_sarima_params(train_data, exog)
    
    print(f"Training SARIMA model with parameters: SARIMA{order}x{seasonal_order}")
    model = SARIMAX(train_data,
                    exog=exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    y_true = y_true.dropna()
    y_pred = y_pred[:len(y_true)]
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

def plot_predictions(train_data, test_data, predictions, train_dates, test_dates, save_path):
    """Plot prediction results"""
    plt.figure(figsize=(15, 7))
    
    # Convert data back to original price scale
    train_data_original = np.exp(train_data)
    test_data_original = np.exp(test_data)
    predictions_original = np.exp(predictions)
    
    plt.plot(train_dates, train_data_original, label='Training Data', color='blue')
    plt.plot(test_dates, test_data_original, label='Actual Test Data', color='orange')
    plt.plot(test_dates, predictions_original, label='Predictions', color='red')
    
    plt.title('SARIMAX Model Predictions vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_path, f'prediction_plot_{timestamp}.png'))
    plt.close()

def save_predictions(test_data, predictions, test_dates, save_path):
    """Save predictions to CSV file"""
    # Convert data back to original price scale
    test_data_original = np.exp(test_data)
    predictions_original = np.exp(predictions)
    
    results_df = pd.DataFrame(index=test_dates)
    results_df['Actual'] = test_data_original
    results_df['Predicted'] = predictions_original
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(os.path.join(save_path, f'predictions_{timestamp}.csv'))

def save_report(metrics, order, seasonal_order, save_path):
    """Save evaluation report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(save_path, f'forecast_report_{timestamp}.txt'), 'w') as f:
        f.write('SARIMAX Model Performance Metrics:\n')
        f.write('--------------------------------\n')
        f.write(f'SARIMA Order (p,d,q): {order}\n')
        f.write(f'Seasonal Order (P,D,Q,s): {seasonal_order}\n')
        f.write('--------------------------------\n')
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')

def main():
    # Create output directory
    output_dir = 'finance_reports'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_and_preprocess_data('GOOGL_standardized.csv')
    
    # Save original data plot
    plot_original_data(df.copy(), output_dir)
    
    # Log transform price
    df['Close'] = np.log(df['Close'])
    plot_transformed_data(df.copy(), output_dir)
    
    # Check stationarity
    is_stationary = check_stationarity(df['Close'])
    print(f"Data is {'stationary' if is_stationary else 'not stationary'}")
    
    # Split training and test sets
    train_size = int(len(df) * 0.8)
    train_data = df['Close'][:train_size]
    test_data = df['Close'][train_size:]
    
    # Prepare exogenous variables
    train_exog = df['Volume_Scaled'][:train_size].values.reshape(-1, 1)
    test_exog = df['Volume_Scaled'][train_size:].values.reshape(-1, 1)
    
    # Train model
    model = train_sarima_model(train_data, exog=train_exog)
    
    # Make predictions
    predictions = model.forecast(steps=len(test_data), exog=test_exog)
    
    # Evaluate model
    metrics = evaluate_model(test_data, predictions)
    
    # Save results
    plot_predictions(train_data, test_data, predictions, df.index[:train_size], df.index[train_size:], output_dir)
    save_predictions(test_data, predictions, df.index[train_size:], output_dir)
    save_report(metrics, model.specification.order, model.specification.seasonal_order, output_dir)

if __name__ == "__main__":
    main()
