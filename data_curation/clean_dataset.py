#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def load_data(input_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Data loaded successfully from {input_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def initial_cleaning(df):
    """Perform initial data cleaning."""
    # Remove rows where 'n1' is '1.58' or 'n1'
    df = df[~df['n1'].isin(['1.58', 'n1'])]
    print("Initial cleaning complete: Removed unwanted 'n1' values")
    return df

def convert_columns_to_float(df, columns):
    """Convert specified columns to float64."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Converted specified columns to float64")
    return df

def clean_radius(df):
    """Clean the 'rad' column."""
    def radius_change(value):
        if pd.notnull(value) and value <= 50 and int(value) % 2 == 1:
            return int(value)
        return None
    
    df['rad'] = df['rad'].apply(radius_change)
    df.dropna(subset=['rad'], inplace=True)
    df['rad'] = df['rad'].astype('int64')
    print("Radius cleaning complete")
    return df

def clean_gap(df):
    """Clean the 'gap' column."""
    def gap_change(value):
        if pd.notnull(value):
            if 20 <= value <= 120 and int(value) % 2 == 1:
                return int(value)
            elif value < 20:
                return int(value)
        return None
    
    df['gap'] = df['gap'].apply(gap_change)
    df.dropna(subset=['gap'], inplace=True)
    df['gap'] = df['gap'].astype('int64')
    print("Gap cleaning complete")
    return df

def clean_lambda_val(df):
    """Clean the 'lambda_val' column."""
    def lambda_val_change(value):
        if pd.notnull(value) and 400 <= value <= 1000:
            return int(value)
        return None
    
    df['lambda_val'] = df['lambda_val'].apply(lambda_val_change)
    df.dropna(subset=['lambda_val'], inplace=True)
    df['lambda_val'] = df['lambda_val'].astype('int64')
    print("Lambda cleaning complete")
    return df

def final_export(df, output_path):
    """Export the cleaned DataFrame to a CSV file."""
    df.reset_index(drop=True, inplace=True)
    try:
        df.to_csv(output_path, index=False)
        print(f"Export to CSV done: {output_path}")
    except Exception as e:
        print(f"Error exporting data: {e}")
        raise

def secondary_lambda_cleaning(input_path, output_path):
    """Perform secondary cleaning on 'lambda_val' to ensure it's even."""
    try:
        data = pd.read_csv(input_path)
        print(f"Data reloaded from {input_path} for secondary cleaning")
    except Exception as e:
        print(f"Error loading data for secondary cleaning: {e}")
        raise

    def lambda_val_even(value):
        if pd.notnull(value) and value % 2 == 0:
            return int(value)
        return None

    data['lambda_val'] = data['lambda_val'].apply(lambda_val_even)
    data.dropna(subset=['lambda_val'], inplace=True)
    data['lambda_val'] = data['lambda_val'].astype('int64')
    print("Secondary lambda cleaning complete")

    data.reset_index(drop=True, inplace=True)
    try:
        data.to_csv(output_path, index=False)
        print(f"Final export to CSV done: {output_path}")
    except Exception as e:
        print(f"Error exporting data: {e}")
        raise

def main():
    # Define input and output paths
    INPUT_PATH = "/scratch/napamegs/Ag_height0/Ag_3_final.csv"
    INTERMEDIATE_OUTPUT_PATH = "/scratch/napamegs/Ag_height0/Ag_3_reduced.csv"
    FINAL_OUTPUT_PATH = "/scratch/napamegs/Ag_height0/Ag_3_reduced_final.csv"

    # Load data
    df = load_data(INPUT_PATH)

    # Initial cleaning
    df = initial_cleaning(df)

    # Convert specified columns to float
    columns_to_convert = ['n1', 'n2', 'height', 'rad', 'gap', 'lambda_val']
    df = convert_columns_to_float(df, columns_to_convert)

    # Clean 'rad' column
    df = clean_radius(df)

    # Clean 'gap' column
    df = clean_gap(df)

    # Clean 'lambda_val' column
    df = clean_lambda_val(df)

    # Display unique values and data overview
    print("Unique values in 'n2':", df['n2'].unique())
    print("Unique values in 'n1':", df['n1'].unique())
    print("Unique values in 'rad':", df['rad'].unique())
    print("Unique values in 'gap':", df['gap'].unique())
    print(df.head())

    # Export intermediate cleaned data
    final_export(df, INTERMEDIATE_OUTPUT_PATH)

    # Perform secondary lambda cleaning
    secondary_lambda_cleaning(INTERMEDIATE_OUTPUT_PATH, FINAL_OUTPUT_PATH)

if __name__ == "__main__":
    main()
