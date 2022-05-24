import torch
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import cv2.cv2 as cv2
import c_utils.config as cfg


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
    net = model_type(img_channel=3, num_classes=cfg.num_classes).to(cfg.device)
    net.load_state_dict(torch.load(weight_path, map_location=torch.device(cfg.device)))
    return net


def make_prediction(model, image, transform, device):
    # image = Image.open(image_path)
    # net = model.to(cfg.device)
    im_pil = Image.fromarray(image)
    net = model
    img = transform(im_pil).unsqueeze(0).to(device)
    net.eval()
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    image = np.array(image)
    # print(f'{cfg.class_name[cfg.classes[predicted]]:.100s}')
    label = cfg.class_name[cfg.classes[predicted]]
    return label, image


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