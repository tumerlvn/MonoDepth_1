import os
import shutil

path = 'data/KITTI/object/training'
output_file_name = 'cyclist_nums.txt'
output_file = open(output_file_name, 'r+')

nums = set(map(int, output_file.read().split()))


def move_to_new(path, directory):
    path = os.path.join(path, directory)
    res_path = 'new_training'
    res_path = os.path.join(res_path, directory)

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            num = int(filename.split('.')[0])
            if num in nums:
                shutil.copyfile(os.path.join(path, filename), os.path.join(res_path, filename))

move_to_new(path, 'velodyne')
move_to_new(path, 'image_2')
move_to_new(path, 'calib')



output_file.close()