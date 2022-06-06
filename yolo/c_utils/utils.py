import os
import torch
import torchvision
import numpy as np
import cv2.cv2 as cv2

from PIL import Image


def init_model(yolo_weight):
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weight, force_reload=True)
    return yolo_model


def yolo_run(image, model):
    list_box = []
    list_image = []
    output = []
    results = model(image, size=640)  # includes NMS
    label_files = results.pred[0].tolist()
    for num, file in enumerate(label_files):
        boxes = file[:4]

        ymin = round((float(boxes[0])))
        xmin = round((float(boxes[1])))
        ymax = round((float(boxes[2])))
        xmax = round((float(boxes[3])))
        iloc = [xmin, ymin, xmax, ymax]

        cropped_image = image[xmin:xmax, ymin:ymax].copy()
        list_box.append(iloc)
        list_image.append(cropped_image)
        output = [list_box, list_image]

    return output


def crop_image(image_path, transform, device, yolo_model, save_path=None):
    image = cv2.imread(image_path)
    yolo_output = yolo_run(image, yolo_model)
    if len(yolo_output) != 0:
        list_box, list_image = yolo_output
        for num, img in enumerate(list_image):
            image = np.array(image)

            xmin, ymin, xmax, ymax = list_box[num]
            image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
    if save_path is None:
        cv2.imshow("Crop image", image)
    else:
        name = save_path + os.path.basename(image_path)
        cv2.imwrite(name, image)
        print(name)


