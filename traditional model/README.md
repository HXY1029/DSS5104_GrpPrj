# SARIMA Prediction Model

This project implements a data prediction system based on the SARIMAX model. The system can process time series data, perform feature engineering, train prediction models, and generate visualization results.

## Features

- Data preprocessing and feature engineering
- SARIMAX model training and parameter optimization
- Prediction result visualization
- Support for both English and Chinese output

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare Data:
   - Ensure data is in CSV format
   - Data should include timestamp column and value columns

2. Run Prediction:
   ```python
   from energy_forecast import train_and_predict
   
   # Train model and generate predictions
   results = train_and_predict('your_data.csv')
   ```

3. View Results:
   - Prediction results will be saved in the `results` directory
   - Visualization charts will be automatically saved in PNG format

## Notes

- Ensure data quality by handling missing values and anomalies
- Adjust model parameters according to actual needs
- Update the model regularly to adapt to new data patterns 