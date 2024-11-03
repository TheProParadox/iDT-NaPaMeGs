import os
import gzip
import shutil

# This is the path to the main folder that contains the 7 subfolders
main_folder_path = '/scratch/napamegs/Ag_height_1to10' 

# Function to extract all .csv.gz files in the specified folder
def extract_csv_gz_files(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv.gz'):
                file_path = os.path.join(subdir, file)
                output_file_path = file_path[:-3]
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

extract_csv_gz_files(main_folder_path)
