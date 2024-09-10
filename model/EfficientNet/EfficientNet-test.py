import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import Linear
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
# from main import VGGNet_Transfer
from label import Waterlevel
from efficientnet_pytorch import EfficientNet

# 测试所保存的模型
path = 'EfficientNet_best.pt'
# path = 'val_model_5_images.pth'

eifficientnet_true = EfficientNet.from_name('efficientnet-b0')
# 修改最后一层以进行十分类
num_ftrs = eifficientnet_true._fc.in_features
eifficientnet_true._fc = nn.Linear(num_ftrs, 10)
model = eifficientnet_true

if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(path))      # 再加载模型参数

# 读取测试集中的数据
root_test = r''
test_dataset = Waterlevel(root_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
with torch.no_grad():
    for batch_idx, (img, target) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        test_output = model(img)
        test_output = test_output.to(torch.float32)  # 将对入参和出参做一个类型转换便于计算MSE的值
        test_target = target.to(torch.float32)
        _, predicted = torch.max(test_output.data, dim=1)
        # print(paths)
        print(test_output)
        print(predicted)
        print(test_target)
        break
