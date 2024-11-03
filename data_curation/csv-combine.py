import os
import pandas as pd

source_dir = '/scratch/napamegs/Ag_h1to10_cleaned/'
des_dir = '/scratch/napamegs/'

df = pd.DataFrame()
i = 1
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root,file)

            dff = pd.read_csv(file_path,header=None,low_memory=False)
            print(f"{i} file is done")
            df = pd.concat([df, dff], ignore_index=True)
            i += 1

output_file_path = os.path.join(des_dir, 'combined_data.csv')
df.to_csv(output_file_path,index=False)