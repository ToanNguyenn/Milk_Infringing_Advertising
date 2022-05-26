import torch
import cv2.cv2 as cv2
from c_utils.utils import *
from c_utils import config as cfg

yolo_weight = r'E:\PycharmProjects\Milk_project\yolov5\best.pt'
classify_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\epoch_30_model.pt'
image_path = r'E:\PycharmProjects\Milk_Infringing_Advertising\data_test\Nature One Dairy Premium Goat Milk Formula Toddler000007.png'
save_path = ''
if __name__ == '__main__':
    yolo_model, classify_model = init_model(yolo_weight, classify_weight)
    make_prediction(image_path=image_path, transform=cfg.img_transforms_test,
                    device=cfg.device, yolo_model=yolo_model, classify_model=classify_model, save_path=None)
