import os
import time

import cv2
import torch

from predictor import Predictor
from yolox.exp.build import get_exp

yolo_model = 'yolox-x'

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


file_name = os.path.join(exp.output_dir, yolo_model)
os.makedirs(file_name, exist_ok=True)

vis_folder = os.path.join(file_name, "vis_res")
os.makedirs(vis_folder, exist_ok=True)

current_time = time.localtime()


images = [
    'data/4_0120.jpg',
    'data/4_0239.jpg',
    'data/4_0343.jpg',
    'data/5_0115.jpg',
    'data/5_0207.jpg',
    'data/5_0304.jpg',
    'data/5_0433.jpg',
    'data/6_0203.jpg',
    'data/6_0231.jpg',
    'data/6_0258.jpg',
    'data/6_0401.jpg',
]

for image in images:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

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
            
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = os.path.join(save_folder, os.path.basename(image))
    print("Saving detection result in {}".format(save_file_name))
    cv2.imwrite(save_file_name, img_final)