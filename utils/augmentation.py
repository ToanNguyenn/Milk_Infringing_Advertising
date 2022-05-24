import albumentations as A
import pandas as pd
import csv
import numpy as np
import cv2
import copy
import random
import matplotlib.pyplot as plt
import os

from albumentations.augmentations.transforms import HueSaturationValue
from albumentations.augmentations.geometric.transforms import Perspective, Affine
import albumentations as A
import numpy as np

def getTransform(loop):
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.GaussNoise (p=0.5)
        ])
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
            A.CoarseDropout(p=1),
        ])
    elif loop == 2:
        transform = A.Compose([
            A.MotionBlur(blur_limit=(4,6), p=1),
            A.ImageCompression(p=0.5)
        ])
    elif loop == 3:
        transform = A.Compose([
            A.CoarseDropout(p=0.8),
            A.RGBShift(p=1),
            A.RandomRotate90(p=0.2)
        ])
    elif loop == 4:
        transform = A.Compose([
            A.AdvancedBlur(p=0.4),
            A.MedianBlur(p=0.4),
            A.ToGray(p=0.4),
        ])
    elif loop == 5:
        transform = A.Compose([
            A.Downscale(scale_min=0.6, scale_max=0.6, p=1),
            A.RandomGamma(p=0.3)
        ])

    return transform


def readImage(filename):
    img = cv2.imread(filename)
    return img


def showImage(image):
    cv2.imshow('Image', image)

root = '/content/drive/MyDrive/AT/Data/Data_Crawl/sua_crop_rename_for_classification'
save_root = '/content/drive/MyDrive/AT/Data/Data_Crawl/Data crawl server UK/augmentation'
fail_img = []
count = 0


if __name__ == '__main__':

    for num in os.listdir(root):
      folder = os.path.join(root, num)
      for name in os.listdir(folder):
        image = os.path.join(folder, name)

        # read img
        image = readImage(image)
        img = copy.deepcopy(image)
        showImage(image)
        print('============')

        # augmentation
        for i in range (0, 6):

          transform = getTransform(i)

          try:
            transformed = transform(image=image)   # transformed
            transformed_image = transformed['image']
          except:
            print("Fail \n")
            print(name)
            continue

        # show img
          showImage(transformed_image)

        # save img, make folder if it not exist
          path = os.path.join(os.path.join(save_root,num))
          isExist = os.path.exists(path)
          if not isExist:
            os.makedirs(path)

          name_save = os.path.join(path,f'{count}_' + name)

          cv2.imwrite(name_save, transformed_image)
          print("name: ", name_save)
          print(os.path.isfile(name_save))
          count += 1
          print('----------------------------')
    print('Done !!!!')
