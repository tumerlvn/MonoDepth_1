import pickle
from visualDet3D.data.kitti.kittidata import KittiData, KittiObj, KittiCalib


with open('workdirs/MonoDTR/output/validation/imdb.pkl', 'rb') as f:
    data = pickle.load(f)
    kitti_data = data[2 % len(data)]
    calib = kitti_data.calib
    print(calib.P2)