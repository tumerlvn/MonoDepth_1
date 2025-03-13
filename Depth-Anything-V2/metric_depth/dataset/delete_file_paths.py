import os

def check_paths(file_path):
    # Open the text file containing the paths
    with open(file_path, 'r') as file:
        lines = file.readlines()

    exists_counter = 0

    directory, original_name = os.path.split(file_path)
    new_file_path = os.path.join(directory, f"fixed_{original_name}")

    with open(new_file_path, "w") as file:
    # Iterate over each line in the file
        for i, line in enumerate(lines):
            # Split the line into two paths (assuming space-separated)
            path1, path2 = line.strip().split()

            # Check if both paths exist
            exists1 = os.path.exists(path1)
            exists2 = os.path.exists(path2)

            # Print the results for each pair of paths
            if exists1 and exists2:
                file.write(line)
                exists_counter += 1

    print(f"Number of both existing lines: {exists_counter}")
    print(f"Number of lines: {len(lines)}")


# Specify the path to the text file
txt_file = '/home/tamerlan/Masters/thesis/Depth-Anything-V2/metric_depth/dataset/splits/hypersim/val.txt'

# Call the function to check paths
check_paths(txt_file)

