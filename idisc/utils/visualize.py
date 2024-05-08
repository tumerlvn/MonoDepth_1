import os
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import sys, json
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.resolve() / "idisc"))
from idisc.models.idisc import IDisc
from idisc.utils import (DICT_METRICS_DEPTH, DICT_METRICS_NORMALS, RunningMetric, validate)
from idisc.utils import colorize

# for instance, if you work with indoor domain: (otw look for kitti model/configs)
min_depth, max_depth = 0.1, 3.0
model = IDisc.build(json.load(open('/home/tamerlan/Masters/thesis/idisc/configs/nyu/nyu_r101.json'))) # depth
# model = IDisc.build(json.load(open('/home/tamerlan/Masters/thesis/idisc/configs/nyunorm/nyunorm_swinl.json'))) # normals

model.load_pretrained("/home/tamerlan/Masters/thesis/idisc/idisc/models/checkpoints/nyu_resnet101.pt") # depth
# model.load_pretrained("/home/tamerlan/Masters/thesis/idisc/idisc/models/checkpoints/nyunormals_swinlarge.pt") # normals
model = model.to("cuda")
model.eval()

# read in image
image_path = "/home/tamerlan/Masters/thesis/yolov5/data/images/frame_75.jpg"
image_resized = Image.open(image_path)
image_resized = image_resized.resize([640,480], resample=Image.LANCZOS)
image = np.asarray(image_resized)
image = TF.normalize(TF.to_tensor(image), **{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
image = image.unsqueeze(0).to("cuda")

with torch.inference_mode():
    depth, *_ = model(image)
save_path = "frame_75.png"

print(depth)

# r = np.round((0.5 * depth[0].squeeze().cpu().numpy()[0] + 0.5) * 255)
# g = np.round((0.5 * depth[0].squeeze().cpu().numpy()[1] + 0.5) * 255)
# b = np.round((0.5 * depth[0].squeeze().cpu().numpy()[2] + 0.5) * 255)

# new_tensor = np.stack([r, g, b], axis=0).astype(np.uint8)
# image_array = np.transpose(new_tensor, (1, 2, 0)).astype(np.uint8)
image_array = colorize(depth[0].squeeze().cpu().numpy(), min_depth, max_depth)

Image.fromarray(image_array).save(save_path)