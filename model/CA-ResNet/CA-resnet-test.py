import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import Linear
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
# from main import VGGNet_Transfer
from label import Waterlevel
import numpy as np
torch.set_printoptions(threshold=np.inf)

log = open('result.txt', mode='a', encoding='utf-8')
# 测试所保存的模型
path = 'resnet50-CA-best.pt'
# path = 'val_model_5_images.pth'

# vgg16_true = torchvision.models.vgg16(pretrained=True)
# vgg16_true.classifier[6] = Linear(4096, 1)         # 直接对最后一层全连接层进行改动
# model = vgg16_true

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class ResNet50WithCoordAtt(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithCoordAtt, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()
        # Add SE blocks after each residual block
        self.ca_blocks = nn.ModuleList([
            CoordAtt(256, 256),
            CoordAtt(512, 512),
            CoordAtt(1024, 1024),
            CoordAtt(2048, 2048)
        ])
        # Add a new fully connected layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.ca_blocks[0](x)

        x = self.resnet.layer2(x)
        x = self.ca_blocks[1](x)

        x = self.resnet.layer3(x)
        x = self.ca_blocks[2](x)

        x = self.resnet.layer4(x)
        x = self.ca_blocks[3](x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNet50WithCoordAtt(num_classes=10)

print(model)
if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(path))      # 再加载模型参数

# 读取测试集中的数据
root_test = r''
test_dataset = Waterlevel(root_test)
test_dataloader = DataLoader(test_dataset, batch_size=1132, shuffle=True)
with torch.no_grad():
    for batch_idx, (img, target, paths) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        test_output = model(img)
        test_output = test_output.to(torch.float32)  # 将对入参和出参做一个类型转换便于计算MSE的值
        test_target = target.to(torch.float32)
        _, predicted = torch.max(test_output.data, dim=1)
        # print(paths)
        # print(test_output)
        print(predicted, file=log)
        print(test_target, file=log)
        break
