import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
import signal
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import itertools
import warnings
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Set Chinese font
try:
    font = FontProperties(fname=r'/System/Library/Fonts/PingFang.ttc')  # macOS
except:
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\SimSun.ttc')  # Windows
    except:
        try:
            font = FontProperties(fname=r'/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')  # Linux
        except:
            print("Warning: Could not find suitable font, characters may not display properly")
            font = None

# Set plotting style
plt.style.use('default')
sns.set_style("whitegrid")
mpl.rcParams['axes.unicode_minus'] = False

def signal_handler(sig, frame):
    """Handle keyboard interrupt signal"""
    print('\nProgram interrupted by user')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def safe_plot(func):
    """Decorator: Safely handle plotting operations"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            plt.close()  # Ensure plot is closed
            return result
        except Exception as e:
            print(f"Error during plotting: {str(e)}")
            plt.close()  # Ensure plot is closed
            return None
    return wrapper

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

@safe_plot
def plot_predictions(actual, predicted, title, save_path=None):
    """Plot prediction results"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Set background style
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Plot actual and predicted values
        ax.plot(actual.index, actual.values, label='Actual Demand', linewidth=2, color='#2878B5')
        ax.plot(predicted.index, predicted.values, label='Predicted Demand', linewidth=2, color='#C82423', linestyle='--')
        
        # Set title and labels
        ax.set_title('Energy Demand Prediction Analysis', fontsize=16, pad=20)
        ax.set_xlabel('Time', fontsize=12, labelpad=10)
        ax.set_ylabel('Demand', fontsize=12, labelpad=10)
        legend = ax.legend(loc='best', fontsize=12, frameon=True, framealpha=0.8)
        
        # Adjust x-axis date display
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        
        # Automatically adjust x-axis label angle and spacing
        plt.xticks(rotation=45, ha='right')
        
        # Set y-axis range with margin
        y_min = min(min(actual.values), min(predicted.values))
        y_max = max(max(actual.values), max(predicted.values))
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved to: {save_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        raise
    finally:
        plt.close()

def optimize_sarimax_params(train_data, exog_data):
    """Grid search for optimal SARIMAX parameters"""
    print("Searching for optimal model parameters...")
    
    # Define parameter ranges
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    # Use subset of data for parameter search
    train_size = min(len(train_data), 5000)
    train_subset = train_data[:train_size]
    exog_subset = exog_data[:train_size] if exog_data is not None else None
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(train_subset,
                              exog=exog_subset,
                              order=param,
                              seasonal_order=seasonal_param,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = model.fit(disp=False, maxiter=50)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = seasonal_param
                    
            except Exception:
                continue
    
    print('Optimal SARIMAX parameters: {}x{}'.format(best_params, best_seasonal_params))
    return best_params, best_seasonal_params

def train_sarimax(data, exog_data, order=(1,1,1), seasonal_order=(1,1,1,24)):
    """Train SARIMAX model"""
    model = SARIMAX(data,
                    exog=exog_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False, maxiter=100)
    return model_fit

def prepare_features(df):
    """Prepare model features"""
    # Create time features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Create lag features
    df['lag_1'] = df['nat_demand'].shift(1)
    df['lag_24'] = df['nat_demand'].shift(24)  # Previous day same hour
    df['lag_168'] = df['nat_demand'].shift(168)  # Previous week same hour
    
    # Create statistical features
    df['rolling_mean_6h'] = df['nat_demand'].rolling(window=6).mean()
    df['rolling_mean_24h'] = df['nat_demand'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['nat_demand'].rolling(window=24).std()
    
    # Handle missing values
    for col in df.columns:
        if col != 'nat_demand':
            df[col] = df[col].fillna(method='bfill')
            df[col] = df[col].fillna(method='ffill')
    
    # Standardize features
    scaler = StandardScaler()
    features = ['hour', 'weekday', 'month', 'is_weekend', 
               'lag_1', 'lag_24', 'lag_168',
               'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h']
    df[features] = scaler.fit_transform(df[features])
    
    return df[features].values

def save_report(predictions, metrics, model_summary, feature_importance, output_dir):
    """Save prediction report"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prediction results
        predictions_df = pd.DataFrame({
            'Time': predictions.index,
            'Predicted Value': predictions.values
        })
        predictions_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        
        # Generate report text
        report_path = os.path.join(output_dir, f'forecast_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Energy Demand Prediction Analysis Report ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Model Evaluation Metrics\n")
            f.write("-----------------\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("2. Feature Importance Analysis\n")
            f.write("-----------------\n")
            for feature, importance in feature_importance.items():
                f.write(f"{feature}: {importance:.4f}\n")
            f.write("\n")
            
            f.write("3. Model Details\n")
            f.write("-----------------\n")
            f.write(str(model_summary))
            f.write("\n\n")
            
            f.write("4. File Information\n")
            f.write("-----------------\n")
            f.write(f"- Prediction Results: predictions_{timestamp}.csv\n")
            f.write(f"- Prediction Plot: prediction_plot_{timestamp}.png\n")
            f.write(f"- Detailed Report: forecast_report_{timestamp}.txt\n")
        
        print(f"\nReport files saved to directory: {output_dir}")
        print(f"- Prediction Results: predictions_{timestamp}.csv")
        print(f"- Prediction Plot: prediction_plot_{timestamp}.png")
        print(f"- Detailed Report: forecast_report_{timestamp}.txt")
        
        return timestamp
        
    except Exception as e:
        print(f"Error saving report: {str(e)}")
        return None

def energy_forecast(data_path):
    """Energy data forecasting"""
    try:
        print("Reading data...")
        df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
        
        # Preprocess data
        print("\nProcessing data...")
        df = df.sort_index()
        df = df.asfreq('h')
        
        # Calculate test set size (three months)
        test_hours = 24 * 90  # 90 days = 3 months
        
        # Use last two years of data
        df = df.last('730D')  # Get last two years of data
        print(f"\nData range: {df.index.min()} to {df.index.max()}")
        print(f"Total hours: {len(df)}")
        
        # Prepare features
        print("\nPreparing features...")
        exog_features = prepare_features(df)
        
        # Split training and test sets (last three months as test set)
        train_data = df['nat_demand'][:-test_hours]
        test_data = df['nat_demand'][-test_hours:]
        train_exog = exog_features[:-test_hours]
        test_exog = exog_features[-test_hours:]
        
        print("\nDataset split:")
        print(f"Training set: {len(train_data)} hours ({len(train_data)/24:.1f} days)")
        print(f"Test set: {len(test_data)} hours ({len(test_data)/24:.1f} days)")
        print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
        print(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")
        
        # Search for optimal parameters
        print("\nSearching for optimal parameters...")
        best_params, best_seasonal_params = optimize_sarimax_params(train_data, train_exog)
        
        # Train model
        print("\nTraining model...")
        model = train_sarimax(train_data, train_exog, 
                            order=best_params, 
                            seasonal_order=best_seasonal_params)
        
        # Make predictions on test set
        print("\nGenerating predictions...")
        test_predictions = model.get_forecast(steps=len(test_data), exog=test_exog)
        predicted_mean = test_predictions.predicted_mean
        
        # Evaluate model
        print("\nEvaluating model performance...")
        metrics = evaluate_model(test_data, predicted_mean)
        
        # Print model summary
        print("\nModel Summary:")
        print(model.summary())
        
        # Analyze feature importance
        print("\nFeature Importance:")
        feature_names = ['hour', 'weekday', 'month', 'is_weekend', 
                         'lag_1', 'lag_24', 'lag_168',
                         'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h']
        coefficients = model.params[1:11]  # Skip intercept
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")
        
        # Create output directory
        output_dir = Path('energy_reports')
        
        # Save prediction results and report
        timestamp = save_report(
            predictions=predicted_mean,
            metrics=metrics,
            model_summary=model.summary(),
            feature_importance=dict(zip(
                ['hour', 'weekday', 'month', 'is_weekend', 
                 'lag_1', 'lag_24', 'lag_168',
                 'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h'],
                model.params[1:11]
            )),
            output_dir=output_dir
        )
        
        # Plot and save prediction results
        if timestamp:
            plot_path = output_dir / f'prediction_plot_{timestamp}.png'
            plot_predictions(test_data, predicted_mean, 'Energy Demand Prediction Results (with features)', save_path=plot_path)
        
        return predicted_mean, metrics
        
    except Exception as e:
        print(f"Error during forecasting: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        print("\n=== Energy Demand Forecasting (SARIMAX Model) ===")
        energy_predictions, energy_metrics = energy_forecast('energy_processed.csv')
        if energy_metrics:
            print("\nEnergy Demand Prediction Evaluation Results:")
            for metric, value in energy_metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print("Energy demand prediction failed, please check data format")
    except KeyboardInterrupt:
        print('\nProgram interrupted by user')
        sys.exit(0)
    except Exception as e:
        print(f"Program error: {str(e)}")
        sys.exit(1)