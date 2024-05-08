# program for converting chen numbers to KITTI numbers
import os

chen_path = 'visualDet3D/data/kitti/chen_split'

def chen_to_normal(chen_num: int, object_type: str = 'val'): # chen -> line in chen_val -> normal
    file = os.path.join(chen_path, object_type + '.txt')
    with open(file) as fp:
        for i, line in enumerate(fp):
            if i == chen_num:
                return line[:-1]
            
def chen_file_to_normal(input_file, object_type: str = 'val'): # эта функция переносит чен номеры в обычные
    output_file_name = 'pedestrian_nums_val_normal.txt'        # по сути можно даже применить к output/val номерам
    output_file = open(output_file_name, 'w+')                 # и получим изначальные номера до сплита

    output = []

    file = os.path.join(chen_path, object_type + '.txt')
    fp = open(file)
    lines = fp.readlines()
    fp.close()

    with open(input_file) as ifp:
        for line in ifp:
            output.append(lines[int(line)])

    for s in output:
        output_file.write(s)
    
    output_file.close()
        
        
chen_file_to_normal('pedestrian_nums_val.txt', 'val')
