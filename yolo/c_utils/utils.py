import os
import torch
import torchvision
import numpy as np
import cv2.cv2 as cv2
from c_utils import config as cfg
from PIL import Image


def init_model(object_detect_weight, rtd_weight):
    yolo_obj_model = torch.hub.load('ultralytics/yolov5', 'custom', path=object_detect_weight, force_reload=True)
    yolo_rtd_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rtd_weight, force_reload=True)
    return yolo_obj_model, yolo_rtd_model


def yolo_run(image, model, type="object_detect"):
    list_box = []
    list_image = []
    list_label = []
    output = []
    if type == 'rtd':
        model.conf = 0.5
    else:
        model.conf = 0.35
        model.classes = [0, 1, 2, 3, 5, 6]
    results = model(image, size=640)  # includes NMS
    label_files = results.pred[0].tolist()
    names = results.names
    for num, file in enumerate(label_files):
        boxes = file[:4]
        label = file[-1]
        ymin = round((float(boxes[0])))
        xmin = round((float(boxes[1])))
        ymax = round((float(boxes[2])))
        xmax = round((float(boxes[3])))
        iloc = [xmin, ymin, xmax, ymax]

        cropped_image = image[xmin:xmax, ymin:ymax].copy()
        list_box.append(iloc)
        list_image.append(cropped_image)
        list_label.append(names[int(label)])
        output = [list_box, list_image, list_label]
    return output


def crop_rtd(image, image_path, yolo_rtd_model,save_path=None):
    # image = cv2.imread(image_path)
    yolo_output = yolo_run(image, yolo_rtd_model, type='rtd')
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
        for num, img in enumerate(list_image):
            xmin, ymin, xmax, ymax = list_box[num]
            image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
            label = list_label[num]
            print(label)
            cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    folder = save_path + '\\rtd\\'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    name = folder + os.path.basename(image_path)
    cv2.imwrite(name, image)
    print(name)


def crop_image(image_path, yolo_obj_model, yolo_rtd_model, save_path=None):
    image = cv2.imread(image_path)
    yolo_output = yolo_run(image, yolo_obj_model)
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
        for num, img in enumerate(list_image):
            xmin, ymin, xmax, ymax = list_box[num]
            image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
            label = list_label[num]
            if label == "sua_binh" or label == "sua_hop":
                crop_rtd(img, image_path=image_path, yolo_rtd_model=yolo_rtd_model, save_path=save_path)
            cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    if save_path is None:
        name = image_path + "_" + "pred" + ".jpg"
        cv2.imwrite(name, image)
        return image
    else:
        name = save_path + '\\' + os.path.basename(image_path)
        cv2.imwrite(name, image)
        print(name)