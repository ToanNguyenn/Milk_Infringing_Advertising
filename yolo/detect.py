from c_utils.utils import *
from c_utils import config as cfg
# Step:
# 1.Khai báo weight path
# 2.input image folder path & save folder path
# 3.init model - input weight của object detect và rtd để tạo 2 model object detect (yolo_model) và rtd
# 4.output của object detect là ảnh gốc có bbox, list ảnh của object có trong ảnh gốc và list label có stt tương ứng
# 5.trong list ảnh object nếu có object nào là sữa bình or sữa hộp thì sửa dụng model rtd
# 6.model rtd nhận ảnh object vào để detect brand và lưu ảnh output lại

yolo_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\object_weight.pt'
rtd_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\rtd_best.pt'

save_path = r'E:\data_test\Object Detect classification 23.06\Object Detect classification 23.06\results_Object_Detect_23.06\obj'
folder = r'E:\data_test\Object Detect classification 23.06\Object Detect classification 23.06\RTD'

if __name__ == '__main__':
    yolo_model, rtd_model = init_model(yolo_weight, rtd_weight)
    for image in os.listdir(folder):
        image_path = os.path.join(folder, image)
        img, list_image, list_label = object_detect(image_path=image_path, yolo_obj_model=yolo_model,
                                                    save_path=save_path, obj_conf=0.35)
        count = 0
        for num, label in enumerate(list_label):
            if label == "sua_binh" or label == "sua_hop":
                rtd_detect(list_image[num], image_path=image_path, yolo_rtd_model=rtd_model,
                           save_path=save_path, count=count, rtd_conf=0.6)
                count += 1


