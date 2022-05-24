from Classification.utils.utils import *
from model.resnet import ResNet101
from Classification.utils import utils as cfg

PATH = 'resnet101.pth'
net = ResNet101(img_channel=3, num_classes=50).to(cfg.device)
net.load_state_dict(torch.load(PATH, map_location=torch.device(cfg.device)))


if __name__ == '__main__':
    image_path = r'E:\PycharmProjects\Milk_project\Classification\test\0\a2 Smart Nutrition_15.jpg'
    label, image = make_prediction(net, transform=cfg.img_transforms_test, image_path=image_path, device=cfg.device)
    # print(label)
    # print(image.shape)
    # cv2.imshow('s', image)
    # cv2.waitKey(0)