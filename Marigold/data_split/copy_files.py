import os
import pandas as pd
import shutil

# Load the txt file with whitespace as the separator
df = pd.read_csv('data_split/kitti/eigen_test_files_with_gt.txt', delim_whitespace=True, header=None, names=['column1', 'column2', 'column3'])

# Initialize a variable to store the total size
total_size = 0

source_prefix = "/media/tamerlan/Tamer/thesis/Kitti/data_depth_annotated"
destination_prefix = "/media/tamerlan/Tamer/thesis/Kitti/data_for_colab"

# Loop through each row and check file paths in the first two columns

for _, row in df.iterrows():
    fp1 = str(row.iloc[0])
    fp2 = str(row.iloc[1])

    sfp1 = source_prefix + '/' + fp1
    sfp2 = source_prefix + '/' + fp2

    dfp1 = destination_prefix + '/' + fp1
    dfp2 = destination_prefix + '/' + fp2

    # Check if file path is not None and file exists
    if os.path.exists(sfp1) and os.path.exists(sfp2):
        os.makedirs(os.path.dirname(dfp1), exist_ok=True)
        os.makedirs(os.path.dirname(dfp2), exist_ok=True)

            
        # Copy the file to the destination
        shutil.copy2(sfp1, dfp1)
        shutil.copy2(sfp2, dfp2)

                
