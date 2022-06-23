from c_utils.utils import *
from c_utils import config as cfg

yolo_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\obj_1.pt'
rtd_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\rtd_best.pt'

# image_path = r'E:\PycharmProjects\Milk_Infringing_Advertising\data_test\Nature One Dairy Premium Goat Milk Formula Toddler000007.png'
save_path = r'E:\data_test\results'
folder = r'E:\data_test\New folder'

if __name__ == '__main__':
    yolo_model, rtd_model = init_model(yolo_weight, rtd_weight)
    for image in os.listdir(folder):
        image_path = os.path.join(folder, image)
        crop_image(image_path=image_path, yolo_obj_model=yolo_model, yolo_rtd_model=rtd_model, save_path=save_path)



