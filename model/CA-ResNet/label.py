
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


class Waterlevel(Dataset):
    def __init__(self, root, train=True, transforms=None):
        imgs = []
        for path in os.listdir(root):
            if path == "level 0":
                label = 0.0
            elif path == "level 1":
                label = 1.0
            elif path == "level 2":
                label = 2.0
            elif path == "level 3":
                label = 3.0
            elif path == "level 4":
                label = 4.0
            elif path == "level 5":
                label = 5.0
            elif path == "level 6":
                label = 6.0
            elif path == "level 7":
                label = 7.0
            elif path == "level 8":
                label = 8.0
            elif path == "level 9":
                label = 9.0
            # elif path == "level 10":
            #     label = 10.0
            else:
                print("data label error")

            childpath = os.path.join(root, path)
            for imgpath in os.listdir(childpath):
                imgs.append((os.path.join(childpath, imgpath), label))

        self.imgs = imgs
        self.train = train

        if transforms is None:
            if self.train:  # 对训练集进行数据增强
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行归一化处理
                self.transforms = T.Compose([
                    T.Resize((224, 224)),  # 将输入的图片调整为224*224*3
                    T.RandomHorizontalFlip(),  # 随机水平翻转，数据增强
                    T.CenterCrop((224, 224)),
                    T.RandomRotation(10),  # 随机角度旋转
                    T.ToTensor(),
                    normalize
                ])
            else:  # 而验证集不需要进行数据增强
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.CenterCrop((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        label = self.imgs[index][1]

        data = Image.open(img_path)
        if data.mode != "RGB":
            data = data.convert("RGB")
        data = self.transforms(data)
        return data, label, img_path

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    # root = "/home/elvis/workfile/dataset/car/train"
    root = r'D:\pycharm\project\water depth prediction\dataset\train-2'
    train_dataset = Waterlevel(root)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for data, label in train_dataset:
        # print(data.shape)
        print(label)