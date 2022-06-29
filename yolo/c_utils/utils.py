import os
import torch
import cv2
from c_utils import config as cfg
<<<<<<< HEAD
=======
# from google.colab.patches import cv2_imshow
from PIL import Image
>>>>>>> 9411d0cceb65146863c418e1a7f23140a1c44812


def init_model(object_detect_weight, rtd_weight):
    yolo_obj_model = torch.hub.load('ultralytics/yolov5', 'custom', path=object_detect_weight, force_reload=True)
    yolo_rtd_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rtd_weight, force_reload=True)
    return yolo_obj_model, yolo_rtd_model


def yolo_run(image, model, model_type="object_detect", rtd_conf=0.2, obj_conf=0.3):
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


def draw_box(yolo_output, image):
    draw_image = image
    list_label = []
    list_image = []
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
        for num, img in enumerate(list_image):
            xmin, ymin, xmax, ymax = list_box[num]
            draw_image = cv2.rectangle(draw_image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
            label = list_label[num]
<<<<<<< HEAD
            cv2.putText(draw_image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return draw_image, list_image, list_label


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
=======
            cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    folder = os.path.join(save_path, 'rtd/')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    name = folder + str(count) + "_" + os.path.basename(image_path)
    cv2.imwrite(name, image)
>>>>>>> 9411d0cceb65146863c418e1a7f23140a1c44812


def object_detect(image_path, yolo_obj_model, save_path=None, obj_conf=0.3):
    image = cv2.imread(image_path)
    yolo_output = yolo_run(image, yolo_obj_model, obj_conf=obj_conf)
    if len(yolo_output) != 0:
<<<<<<< HEAD
        img, list_image, list_label = draw_box(yolo_output, image)
        save_obj(save_path, image_path, img)
        return image, list_image, list_label
    else:
        list_label = []
        list_image = []
        save_obj(save_path, image_path, image)
        return image, list_image, list_label


def rtd_detect(image, image_path, yolo_rtd_model, save_path=None, count=0, rtd_conf=0.2):
    folder = os.path.join(save_path, 'rtd')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    name = os.path.join(folder, str(count) + "_" + os.path.basename(image_path))

    yolo_output = yolo_run(image, yolo_rtd_model, model_type='rtd', rtd_conf=rtd_conf)
    if len(yolo_output) != 0:
        img, list_image, list_label = draw_box(yolo_output, image)
        cv2.imwrite(name, img)
    else:
        cv2.imwrite(name, image)
=======
        list_box, list_image, list_label = yolo_output
        for num, img in enumerate(list_image):
            xmin, ymin, xmax, ymax = list_box[num]
            image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
            label = list_label[num]
            if label == "sua_binh" or label == "sua_hop":
                crop_rtd(img, image_path=image_path, yolo_rtd_model=yolo_rtd_model, save_path=save_path, count=count, rtd_conf=rtd_conf)
                count += 1
            cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    if save_path is None:
        name = image_path + "_" + "pred" + ".jpg"
        cv2.imwrite(image)
        return image
    else:
        name = os.path.join(save_path, os.path.basename(image_path))
#         cv2.imshow('image', image)
        cv2.imwrite(name, image)
        print(name)
    # print("-------------------")
>>>>>>> 9411d0cceb65146863c418e1a7f23140a1c44812
