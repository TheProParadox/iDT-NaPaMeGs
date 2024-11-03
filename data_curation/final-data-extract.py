import os
import pandas as pd

# Assume 'main_directory' is the directory containing the 7 subdirectories.
main_directory = '/scratch/napamegs/Ag_height_1to10'

def clean_data(df):
    """
    This function contains all the cleaning steps applied to the dataframe.
    """
    # Apply all your cleaning steps here, just as an example:
    df = df[df['n1'] != '1.58']
    df = df[df['n1'] != 'n1']
    
    # Convert columns to float and other cleaning steps
    for i in ['n1','n2','height','rad','gap','lambda_val']:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    
    # Define your cleaning functions
    def radius_change(value):
        if value <= 50 and value % 2 == 1:
            return int(value)
        return None

    df['rad'] = df['rad'].apply(radius_change)

    def gap_change(value):
        if value >= 20 and value<=120 and value % 2 == 1:
            return int(value)
        elif value <=20:
            return int(value)
        else:
            return None


    df['gap'] = df['gap'].apply(gap_change)
    df.dropna(subset=['gap'], inplace=True)

    df['gap'].unique()
    df = df.astype({'gap': 'int64'})

    def lambda_val_change(value):
        if value >= 400 and value<=1000:
            return int(value)
        else:
            return None

    df['lambda_val'] = df['lambda_val'].apply(lambda_val_change)

    df.dropna(subset=['lambda_val'], inplace=True)

    df['lambda_val'].unique()
    df = df.astype({'lambda_val': 'int64'})
    df.reset_index()

    return df


for subdir, _, files in os.walk(main_directory):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file_path.endswith('.csv') and 'cleaned' not in file_path:
            df = pd.read_csv(file_path, low_memory='false')
            
            df_cleaned = clean_data(df)

            cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
            
            df_cleaned.to_csv(cleaned_file_path, index=False)
            print(f"Cleaned file saved to {cleaned_file_path}")
            
print("Data cleaning complete for all files.")
