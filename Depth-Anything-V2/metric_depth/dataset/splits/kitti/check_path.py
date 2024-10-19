import os

def check_paths(file_path):
    # Open the text file containing the paths
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate over each line in the file
    for i, line in enumerate(lines):
        # Split the line into two paths (assuming space-separated)
        path1, path2 = line.strip().split()

        # Check if both paths exist
        exists1 = os.path.exists(path1)
        exists2 = os.path.exists(path2)

        # Print the results for each pair of paths
        if not exists1 and not exists2:
            print(f"Line {i+1}: Neither path exists.")
        elif not exists1:
            print(f"Line {i+1}: First path does not exist.")
        elif not exists2:
            print(f"Line {i+1}: Second path does not exist.")

# Specify the path to the text file
txt_file = '/home/tamerlan/Masters/thesis/Depth-Anything-V2/metric_depth/dataset/splits/vkitti2/train.txt'

# Call the function to check paths
check_paths(txt_file)


# /home/tamerlan/Masters/thesis/Depth-Anything/metric_depth/data/Kitti/data_depth_annotated/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png
# /home/tamerlan/Masters/thesis/Depth-Anything/metric_depth/data/Kitti/data_depth_annotated/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png