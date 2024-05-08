import os
import shutil

path = 'data/KITTI/object/training'
output_file_name = 'cyclist_nums.txt'
output_file = open(output_file_name, 'r+')

nums = list(map(int, output_file.read().split()))
nums.sort()


def rename(path, directory, end):
    cnt = 0
    path = os.path.join(path, directory)

    for num in nums:
        filename = str(num).rjust(6, '0') + '.' + end
        padded_num = str(cnt).rjust(6, '0')
        new_name = padded_num + '.' + end
        os.rename(os.path.join(path, filename), os.path.join(path, new_name))
        cnt += 1


rename(path, 'velodyne', 'bin')
rename(path, 'image_2', 'png')
rename(path, 'label_2', 'txt')
rename(path, 'calib', 'txt')


output_file.close()