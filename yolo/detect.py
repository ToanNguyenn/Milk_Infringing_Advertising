from c_utils.utils import *
from c_utils import config as cfg

yolo_weight = r'E:\PycharmProjects\Milk_project\yolov5\best.pt'
image_path = r'E:\PycharmProjects\Milk_Infringing_Advertising\data_test\Nature One Dairy Premium Goat Milk Formula Toddler000007.png'
save_path = ''
if __name__ == '__main__':
    yolo_model = init_model(yolo_weight)
    crop_image(image_path=image_path, transform=cfg.img_transforms_test,
               device=cfg.device, yolo_model=yolo_model, save_path=None)