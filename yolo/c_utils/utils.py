import os
import torch
import cv2
from c_utils import config as cfg


def init_model(object_detect_weight, rtd_weight):
    yolo_obj_model = torch.hub.load('ultralytics/yolov5', 'custom', path=object_detect_weight, force_reload=True)
    yolo_rtd_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rtd_weight, force_reload=True)
    return yolo_obj_model, yolo_rtd_model


def yolo_run(image, model, model_type="object_detect", rtd_conf=0.2, obj_conf=0.3, device=cfg.device):
    list_box = []
    list_image = []
    list_label = []
    output = []
    if model_type == 'rtd':
        model.conf = rtd_conf
        model.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        model.conf = obj_conf
        model.classes = [0, 1, 2, 3, 5, 6]
    model.to(device)
    results = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size=640)  # includes NMS
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


def draw_box(list_box, list_image, list_label, save_path=None, image_path=None, input_image=None,
             count=None, save_result=False, visualize=False, is_rtd=False):
    if is_rtd:
        image = input_image
    else:
        image = cv2.imread(image_path)
    for num, img in enumerate(list_image):
        xmin, ymin, xmax, ymax = list_box[num]
        image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
        label = list_label[num]
        cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    if is_rtd:
        folder = os.path.join(save_path, 'rtd')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        name = os.path.join(folder, str(count) + "_" + os.path.basename(image_path))
        if save_result:
            cv2.imwrite(name, image)
        if visualize:
            try:
                cv2.imshow("Image", image)
                cv2.waitKey(0)
            except:
                from google.colab.patches import cv2_imshow
                cv2_imshow(image)
    else:
        if save_result:
            save_obj(save_path, image_path, image)
        if visualize:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
    return image, list_image, list_label


def save_obj(save_path, image_path, image):
    if save_path is None:
        name = image_path + "_pred.jpg"
        cv2.imwrite(name, image)
        print(name)
    else:
        name = os.path.join(save_path, os.path.basename(image_path))
        cv2.imwrite(name, image)
        print(name)
    print("-------------------")


def object_detect(image_path, yolo_obj_model, obj_conf=0.33):
    image = cv2.imread(image_path)
    list_box = []
    list_image = []
    list_label = []
    yolo_output = yolo_run(image, yolo_obj_model, obj_conf=obj_conf)
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
    return list_box, list_image, list_label


def rtd_detect(image, yolo_rtd_model, rtd_conf=0.6):
    list_box = []
    list_label = []
    list_image = []

    yolo_output = yolo_run(image, yolo_rtd_model, model_type='rtd', rtd_conf=rtd_conf)
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
    return list_box, list_image, list_label
