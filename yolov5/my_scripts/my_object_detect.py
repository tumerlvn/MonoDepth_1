import cv2
import numpy as np
import sys
import os
from rectangles import Rect, get_good_rectangles
from PIL import Image
import torch
from torchvision import transforms
from efficientnet.efficientnet.models.efficientnet import EfficientNet, params
from torch import nn
from collections import OrderedDict

checkpoint = torch.load('/home/tamerlan/Masters/intro_deep_learning/efficientnet/experiments/carshumans/best.pth')
model = EfficientNet(1.0, 1.0, 0.2)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2), nn.Softmax(dim=1))

state_dict = checkpoint['model']
model_dict = model.state_dict()
new_state_dict = OrderedDict()
matched_layers, discarded_layers = [], []
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:]

    if k in model_dict and model_dict[k].size() == v.size():
        new_state_dict[k] = v
        matched_layers.append(k)
    else:
        discarded_layers.append(k)

model_dict.update(new_state_dict)
model.load_state_dict(model_dict)

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def get_names_and_res(frame, rectangles):
    names_res = []
    model.eval()

    for rect in rectangles:
        with torch.no_grad():
            img = tfms(Image.fromarray(frame[rect.y:rect.bottom(), rect.x:rect.right()])).unsqueeze(0)
            res = model(img)
            if res[0][0] > 0.5:
                name = 'Car'
                fin_res = res[0][0]
            elif res[0][1] > 0.5:
                name = 'Human'
                fin_res = res[0][1]
            else:
                name = 'Neither'
            names_res.append((name, fin_res))

    return names_res


def init_tracker(type_no):
    tracker_types = ['BOOSTING', 'MIL', 'MOSSE']

    tracker_type = tracker_types[type_no]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy_TrackerBoosting.create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()  
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create() 
    
    return tracker

def drawRectWithName(frame, bbox, name):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    cv2.putText(frame, name, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)

def changeBrightnessAndContrast(frame, bbox):
    x, y, w, h = bbox
    crop = frame[y:y+h, x:x+w]
    gain = 0.3
    crop = (gain * crop.astype(np.float64)).clip(0,255).astype(np.uint8)
    
    matr = np.ones((h, w, 3), dtype='uint8') * 2.0
    
    crop = np.uint8(np.clip(cv2.multiply(np.float64(crop), matr), 0, 255))
    frame[y:y+h, x:x+w] = crop

def filter_rects_size(rectangles, w, h):
    result = []
    for rect in rectangles:
        if rect.w > w and rect.h > h:
            result.append(rect)
    return result

video = cv2.VideoCapture("humans_cars2_slowed.mp4")

ok, frame = video.read()
print(ok)

if not video.isOpened():
    print('Error')
    sys.exit()
else:
    w = 960
    h = 540
    frame = cv2.resize(frame, (w, h))

grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prevFrame = grayFrame

frame_count = 0
frame_freq = 100
trackers = []
rectangles = []
first_flag = True

video_out = cv2.VideoWriter("res_cars_humans.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (w,h))
end = False

while True:
    for i in range(3):
        ok, frame = video.read()
        frame_count += 1
        if not ok:
            end = True
            break
    
    if end:
        break

    frame = cv2.resize(frame, (w, h))
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayFrame, prevFrame)
    tmpCopy = frame.copy()

    if first_flag or frame_count >= frame_freq:
        first_flag = False
        trackers.clear()
        rectangles.clear()
        frame_count = 0

        retval, dstImg = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)    


        dstImg = cv2.blur(dstImg, (7,7))
        retval, dstImg = cv2.threshold(dstImg, 40, 255, cv2.THRESH_BINARY)
        dstImg = cv2.blur(dstImg, (7,7))

        contours, hier = cv2.findContours(dstImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 700:
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
                if w_rect > 5 and h_rect > 5:
                    rectangles.append(Rect(x_rect, y_rect, w_rect, h_rect))

        didMerge = True
        while didMerge:
            rectangles, didMerge = get_good_rectangles(rectangles)

        rectangles = filter_rects_size(rectangles, 25, 25)

        names_res = get_names_and_res(tmpCopy, rectangles)
        
        for i in range(len(rectangles)):
            rect = rectangles[i]
            # drawRectWithName(tmpCopy, rect.bbox(), 'test')
            tmpTracker = init_tracker(1)
            tmpTracker.init(tmpCopy, rect.bbox())
            trackers.append(tmpTracker)

        
    
    
    for i in range(len(trackers)):
        ok, bbox = trackers[i].update(frame)
        if ok:
            drawRectWithName(tmpCopy, bbox, names_res[i][0] + ' ' + str(round(float(names_res[i][1]), 2)))
            changeBrightnessAndContrast(tmpCopy, bbox)

    video_out.write(tmpCopy)
    prevFrame = grayFrame

    # cv2.imshow("img", tmpCopy)
    # if cv2.waitKey(1) == ord('q'):
    #     break

video_out.release()
# cv2.destroyAllWindows()