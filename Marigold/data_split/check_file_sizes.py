import os
import pandas as pd

# Load the txt file with whitespace as the separator
df = pd.read_csv('data_split/kitti/colab_eigen_test_files_with_gt.txt', delim_whitespace=True, header=None, names=['column1', 'column2', 'column3'])

# Initialize a variable to store the total size
total_size = 0

prefix = "/media/tamerlan/Tamer/thesis/Kitti/data_for_colab"

counter = 0

# Loop through each row and check file paths in the first two columns
for _, row in df.iterrows():
    fp1 = str(row.iloc[0])
    fp2 = str(row.iloc[1])

    fp1 = prefix + '/' + fp1
    fp2 = prefix + '/' + fp2

    efp1 = os.path.exists(fp1)
    efp2 = os.path.exists(fp2)

    # Check if file path is not None and file exists
    if efp1 and efp2:
        total_size += os.path.getsize(fp1)
        total_size += os.path.getsize(fp2)
        counter += 1
        print("Both exist")
    elif efp1 and not efp2:
        print("only first")
    elif not efp1 and efp2:
        print("only second")
    else:
        print("none")
    
print(df.shape)
print(counter)

print(f"Total size of existing files: {total_size} bytes")