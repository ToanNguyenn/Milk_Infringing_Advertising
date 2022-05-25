import torch
import torchvision
import numpy as np
import cv2.cv2 as cv2
import c_utils.config as cfg
import matplotlib.pyplot as plt

from PIL import Image
from model.resnet import ResNet101


def plot_acc(train_accu, eval_acc):
    plt.plot(train_accu, '-o')
    plt.plot(eval_acc, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.show()


def plot_losses(train_losses, eval_losses):
    plt.plot(train_losses, '-o')
    plt.plot(eval_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.show()


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = torch.sigmoid(model(x))
            predictions = (scores>0.5).float()
            num_correct += (predictions == y).sum()
            num_samples += predictions.shape[0]

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


def save_model(model, PATH, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(model.state_dict(), PATH)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def resnet_init(model_type, weight_path):
    # net = model_type(img_channel=3, num_classes=cfg.num_classes).to(cfg.device)
    net = model_type(num_classes=cfg.num_classes).to(cfg.device)
    net.load_state_dict(torch.load(weight_path, map_location=torch.device(cfg.device)))
    return net


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


def make_prediction(image_path, transform, device, yolo_model, resnet_model):
    image = cv2.imread(image_path)
    model = resnet_model
    yolo_output = yolo_run(image, yolo_model)
    list_box, list_image = yolo_output

    for num, img in enumerate(list_image):

        im_pil = Image.fromarray(image)
        img = transform(im_pil).unsqueeze(0).to(device)

        model.eval()
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        image = np.array(image)
        label = cfg.class_name[cfg.classes[predicted]]

        xmin, ymin, xmax, ymax = list_box[num]
        image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
        cv2.putText(image, label, (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    name = image_path + "_" + "pred" + ".jpg"
    cv2.imwrite(name, image)
    print(name)

def init_model(yolo_weight, resnet_weight):
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weight, force_reload=True)
    resnet_model = resnet_init(ResNet101, weight_path=resnet_weight)
    return yolo_model, resnet_model


def pred_test_loader(test_loader, model, path):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # print images
    cv2.imshow('Batch Image', torchvision.utils.make_grid(images))
    print('Label')
    print(' '.join(f'{cfg.class_name[cfg.classes[labels[j]]]:.100s}' + ' : ' + f'{cfg.classes[labels[j]]:.0f}' + '\n' for j in range(4)))

    net = model.to(cfg.device)
    net.load_state_dict(torch.load(path))
    net.eval()
    images = images.to(cfg.device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print("Predicted")
    print(' '.join(f'{cfg.class_name[cfg.classes[predicted[j]]]:.100s}' + ' : ' + f'{cfg.classes[predicted[j]]:.0f}' + '\n' for j in range(4)))