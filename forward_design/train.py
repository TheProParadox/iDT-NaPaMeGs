import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Check GPU Availability
def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) available.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available. Using CPU.")

# Load Dataset
def load_data(file_path):
    """Load the dataset from a specified file path."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Data Preprocessing
def preprocess_data(data, input_cols, output_col):
    """Separate input and output columns and apply standard scaling."""
    X = data[input_cols]
    y = data[output_col]
    scaler = StandardScaler()
    return X, y, scaler

# Model Definition
def create_model(input_dim):
    """Build and compile a sequential neural network model."""
    model = Sequential([
        Dense(64, input_shape=(input_dim,)),
        LeakyReLU(alpha=0.1),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dense(64),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse', 'mae'])
    return model

# Training and Evaluation with K-Fold Cross-Validation
def train_kfold(X, y, scaler, n_splits=5, epochs=50, batch_size=2000):
    """Train the model using K-Fold cross-validation."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_no = 1
    loss_per_fold = []
    mae_per_fold = []
    r2_per_fold = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train the model
        model = create_model(input_dim=X_train.shape[1])
        print(f'Training for fold {fold_no} ...')
        history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(X_test_scaled, y_test), verbose=1)
        
        # Evaluate the model
        scores = model.evaluate(X_test_scaled, y_test, verbose=0)
        predictions = model.predict(X_test_scaled)
        r2 = r2_score(y_test, predictions)
        
        print(f'Score for fold {fold_no}: Loss = {scores[0]}; MAE = {scores[2]}; R2 = {r2}')
        loss_per_fold.append(scores[0])
        mae_per_fold.append(scores[2])
        r2_per_fold.append(r2)

        # Plot training and validation loss
        plot_loss(history, fold_no)

        fold_no += 1

    # Save model and print summary statistics
    model.save('height-regression-fold-5-epochs-100.h5')
    print("Model training complete.")
    print("MAE per fold:", mae_per_fold)
    print("Loss per fold:", loss_per_fold)
    print("R^2 per fold:", r2_per_fold)
    print(f'Average Loss: {np.mean(loss_per_fold)}')
    print(f'Average MAE: {np.mean(mae_per_fold)}')
    print(f'Average R^2 Score: {np.mean(r2_per_fold)}')

def plot_loss(history, fold_no):
    """Plot training and validation loss for each fold."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].set_title(f'Fold {fold_no} Train Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title(f'Fold {fold_no} Validation Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"loss_fold_{fold_no}.png")
    plt.close()
    print(f"Saved loss plot for fold {fold_no}")

# Main function
def main(input_file, input_cols, output_col, n_splits=5, epochs=50, batch_size=2000):
    check_gpu()
    data = load_data(input_file)
    X, y, scaler = preprocess_data(data, input_cols, output_col)
    train_kfold(X, y, scaler, n_splits=n_splits, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    # Define paths and columns
    input_file = '/scratch/napamegs/Ag_height0/Ag_3_reduced.csv' 
    input_cols = ['rad', 'gap', 'n2', 'lambda_val']
    output_col = ['Ts']
    
    # Define hyperparameters
    n_splits = 5
    epochs = 50
    batch_size = 2000
    
    main(input_file, input_cols, output_col, n_splits=n_splits, epochs=epochs, batch_size=batch_size)
