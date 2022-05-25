import torch
from PIL import Image
import cv2.cv2 as cv2





if __name__ == '__main__':
    weight_path = r'E:\PycharmProjects\Milk_project\yolov5\best.pt'
    image = r'E:\PycharmProjects\Milk_project\Classification\data_test\SMA Pro First Infant Milk0000051650688199.8493545.jpg'
    img = cv2.imread(image)
    list_box, list_image = yolo_run(img, weight_path)
    for img in list_image:
        cv2.imshow("crop", img)
        cv2.waitKey(0)