import torch
import cv2.cv2 as cv2
from c_utils.utils import *
from yolo import yolo_init
from c_utils import config as cfg
from model.resnet import ResNet101

weight_path = r'E:\PycharmProjects\Milk_project\yolov5\best.pt'
PATH = 'E:\PycharmProjects\Milk_Infringing_Advertising\weight\epoch_30_model.pt'
image_path = r'E:\PycharmProjects\Milk_project\Classification\data_test\SMA Pro First Infant Milk0000051650688199.8493545.jpg'

if __name__ == '__main__':
    img = cv2.imread(image_path)
    list_box = yolo_init(img, weight_path)
    model = resnet_init(ResNet101, weight_path=PATH)
    for img in list_box:
        cv2.imshow("crop", img)
        label, image = make_prediction(model, transform=cfg.img_transforms_test, image=img, device=cfg.device)
        print(label)
        print(image.shape)
        cv2.waitKey(0)