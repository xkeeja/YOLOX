import os
import time

import cv2
import torch

from predictor import Predictor
from yolox.exp.build import get_exp

yolo_model = 'yolox-s'

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


ckpt = torch.load(f"models/{yolo_model.replace('-','_')}.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

predictor = Predictor(
    model=model,
    exp=exp,
    device=device, 
)


cap = cv2.VideoCapture("data/4_trimmed.MP4")
fps = round(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"frames: {frame_count}")

writer = cv2.VideoWriter("data/output/4.MP4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

n = 0
while True:
    ret, frame = cap.read()
    if ret:
        print(f'processing {n}...')
        img = frame

        imgheight=img.shape[0]
        imgwidth=img.shape[1]

        y1 = 0
        M = imgheight // 4
        N = imgwidth // 8

        img_final = img.copy()
        for y in range(0,imgheight,M):
            for x in range(0, imgwidth, N):
                y1 = y + M
                x1 = x + N
                tiles = img[y:y+M,x:x+N]

                outputs, img_info = predictor.inference(tiles)
                result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
                
                img_final[y:y+M,x:x+N] = result_image
                
        writer.write(img_final)
        n += 1
    else:
        break
    
writer.release()