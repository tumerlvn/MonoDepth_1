import csv
import os

# Input and output file paths
input_txt_file = 'Marigold/data_split/hypersim/val.txt'
output_csv_file = 'Marigold/data_split/hypersim/my_csv_split_val.csv'

# Open the input text file and output CSV file
with open(input_txt_file, 'r') as txt_file, open(output_csv_file, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow([
        'scene_name', 'camera_name', 'frame_id', 
        'included_in_public_release', 'exclude_reason', 'split_partition_name'
    ])
    
    # Process each line in the input file
    for line in txt_file:
        # Split the line into RGB and depth paths
        rgb_path, depth_path = line.strip().split()
        

        # /media/tamerlan/Tamer/thesis/hypersim_uncompressed/ai_004_004/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg
        # Extract scene name, camera name, and frame ID from the RGB path
        parts = rgb_path.split('/')
        scene_name = parts[-4]  # e.g., ai_004_004

        cam_parts = parts[-2].split('_')
        camera_name = f"{cam_parts[1]}_{cam_parts[2]}"
        frame_id = int(parts[-1].split('.')[1])  # e.g., 0000 -> 0
        
        # Add the required columns
        included_in_public_release = True
        exclude_reason = ''
        split_partition_name = 'val'  # Assuming all entries are for validation
        
        # Write the row to the CSV file
        csv_writer.writerow([
            scene_name, camera_name, frame_id, 
            included_in_public_release, exclude_reason, split_partition_name
        ])

print(f"CSV file saved to {output_csv_file}")