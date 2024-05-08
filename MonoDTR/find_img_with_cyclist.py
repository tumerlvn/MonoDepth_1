import os

# path = 'data/KITTI/object/training/label_2'

path = 'workdirs/MonoDTR/output/validation/data'
# path = 'data/KITTI/object/training/label_2'
output_file_name = 'pedestrian_nums_val.txt'
output_file = open(output_file_name, 'w+')

output = []

for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        for line in open(os.path.join(path, filename)):
            if line.split()[0] == 'Pedestrian':
                output.append(filename)
                break

for s in sorted(output):
    output_file.write(s[:-4] + '\n')
        

output_file.close()

# P2: 2.31265931e+03 -1.76732180e+03 1.89511778e+03 4.71670146e+04 2.84200896e+02 2.58467452e+03 3.08787923e+03 5.24202944e+03 -4.52825094e-01 -2.00279401e-01 8.68813902e-01 2.51200564e+01