import os
import time
import csv

import cv2
import torch
import numpy as np

from predictor import Predictor
from yolox.exp.build import get_exp

yolo_model = 'yolox-l'

exp = get_exp(exp_name=yolo_model)

exp.test_conf = 0.25
exp.nmsthre = 0.45
exp.test_size = (640, 640)

model = exp.get_model()
if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()
else:
    device = 'cpu'
model.eval()


ckpt = torch.load(f"/mnt/data/YOLOX/models/{yolo_model.replace('-','_')}.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

predictor = Predictor(
    model=model,
    exp=exp,
    device=device, 
)

if not os.path.isfile('/mnt/data/YOLOX/output/detect_times.csv'):
    with open('/mnt/data/YOLOX/output/detect_times.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'model', 'resolution', 'grids', 'grid_size', 'avg_time_per_frame'])

videos = [
    '/mnt/data/YOLOX/input/4.mp4',
    '/mnt/data/YOLOX/input/5.mp4',
    '/mnt/data/YOLOX/input/6.mp4'
]

for video in videos:
    cap = cv2.VideoCapture(video)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames: {frame_count}")

    writer = cv2.VideoWriter(f"/mnt/data/YOLOX/output/{yolo_model}_{os.path.basename(video)}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))

    n = 0
    detect_time = []
    rows = 4
    cols = 8
    while True:
        ret, frame = cap.read()
        if ret:
            print(f'processing {n}...')
            img = frame

            imgheight=img.shape[0]
            imgwidth=img.shape[1]

            y1 = 0
            M = imgheight // rows
            N = imgwidth // cols

            img_final = img.copy()
            start = time.time()
            for y in range(0,imgheight,M):
                for x in range(0, imgwidth, N):
                    y1 = y + M
                    x1 = x + N
                    tiles = img[y:y+M,x:x+N]

                    outputs, img_info = predictor.inference(tiles)
                    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
                    
                    img_final[y:y+M,x:x+N] = result_image
            detect_time.append(time.time() - start)            
            
            img_final = cv2.resize(img_final, (0,0), fx=0.5, fy=0.5)
            
            writer.write(img_final)
            n += 1
        else:
            break
        
    writer.release()
    
    with open('/mnt/data/YOLOX/output/detect_times.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(video), yolo_model, f'{width}x{height}', f'{cols}x{rows}', f'{int(width/cols)}x{int(height/rows)}', np.mean(detect_time)])