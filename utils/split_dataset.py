import os
import splitfolders
import random

random.seed(10)
root = r'E:\PycharmProjects\Milk_project\DataTextSpotting'
save = r'E:\PycharmProjects\Milk_project\Classification\data_train'
for folder in os.listdir(root):
    root_folder = os.path.join(root, folder)
    splitfolders.ratio(str(root), output=save,seed=42, ratio=(.7, .2, .1), )
