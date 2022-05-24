from c_utils.utils import *
from model.resnet import ResNet101
from c_utils import config as cfg

PATH = 'E:\PycharmProjects\Milk_Infringing_Advertising\weight\epoch_30_model.pt'
net = ResNet101(img_channel=3, num_classes=cfg.num_classes).to(cfg.device)
net.load_state_dict(torch.load(PATH, map_location=torch.device(cfg.device)))


if __name__ == '__main__':
    image_path = r'E:\PycharmProjects\Milk_project\Classification\data_test\SMA Pro First Infant Milk0000051650688199.8493545.jpg'
    label, image = make_prediction(net, transform=cfg.img_transforms_test, image_path=image_path, device=cfg.device)
    print(label)
    print(image.shape)
    cv2.imshow('s', image)
    cv2.waitKey(0)