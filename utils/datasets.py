import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MilkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=transforms.ToTensor()):
        self.csv = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join((str(self.csv.iloc[idx, 1])), str(self.csv.iloc[idx, 0]))
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        annotations = np.array(self.csv.iloc[idx, 1])
        # annotations = torch.tensor(annotations.astype('float')).view(1).to(torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, annotations
