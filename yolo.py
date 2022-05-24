import torch
from PIL import Image
import cv2.cv2 as cv2


def yolo_init(image, weight_path):
    list_box = []
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)
    results = model(image, size=640)  # includes NMS
    label_files = results.pred[0].tolist()
    for num, file in enumerate(label_files):
        # print(file)
        boxes = file[:4]
        categories = file[4]

        ymin = round((float(boxes[0])))
        xmin = round((float(boxes[1])))
        ymax = round((float(boxes[2])))
        xmax = round((float(boxes[3])))

        cropped_image = image[xmin:xmax, ymin:ymax].copy()
        list_box.append(cropped_image)
    return list_box


if __name__ == '__main__':
    weight_path = r'E:\PycharmProjects\Milk_project\yolov5\best.pt'
    image = r'E:\PycharmProjects\Milk_project\Classification\SMA Pro First Infant Milk0000051650688199.8493545.jpg'
    img = cv2.imread(image)
    list_box = yolo_init(img,weight_path)
    for img in list_box:
        cv2.imshow("crop", img)
        cv2.waitKey(0)