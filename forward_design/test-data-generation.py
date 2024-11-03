import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load Test Data
def load_test_data(file_path, input_cols, output_col):
    """Load the test data and separate input/output columns."""
    data = pd.read_csv(file_path)
    X_test = data[input_cols]
    y_test = data[output_col]
    return X_test, y_test

# Preprocess Test Data
def preprocess_test_data(X_test, scaler):
    """Apply standard scaling to test data."""
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled

# Load and Test Model
def test_model(model_path, X_test_scaled, y_test):
    """Load the model and evaluate it on the test data."""
    # Load model
    model = load_model(model_path)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Evaluate predictions
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test MAE: {mae}")
    print(f"Test R^2 Score: {r2}")
    
    return predictions

# Main function
def main(test_file, model_path, input_cols, output_col):
    # Load test data
    X_test, y_test = load_test_data(test_file, input_cols, output_col)
    
    # Initialize scaler and fit on test data (use the scaler from training in actual use)
    scaler = StandardScaler()
    X_test_scaled = preprocess_test_data(X_test, scaler)
    
    # Test the model
    predictions = test_model(model_path, X_test_scaled, y_test)
    
    # Optional: Save predictions to a CSV file
    output_df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': predictions.flatten()})
    output_df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to test_predictions.csv")

if __name__ == "__main__":
    # Define paths and columns
    test_file = '/path/to/test_data.csv'
    model_path = 'height-regression-fold-5-epochs-100.h5'
    input_cols = ['rad', 'gap', 'n2', 'lambda_val']
    output_col = ['Ts']
    
    main(test_file, model_path, input_cols, output_col)
