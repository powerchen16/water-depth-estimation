import torch
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import Linear
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
# from main import VGGNet_Transfer
from label import Waterlevel
from transformers import ViTForImageClassification, ViTFeatureExtractor
torch.set_printoptions(threshold=np.inf)   # 将所有数据显示完整！！！！


# 测试所保存的模型
path = 'ViT-class10-best.pt'
state_dict = torch.load('ViT-class10-best.pt')
for key in list(state_dict.keys()):
    if key.startswith('module.'):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

model = ViTForImageClassification.from_pretrained('ViT_weights')
model.classifier = nn.Linear(model.config.hidden_size, 10) 
model.load_state_dict(state_dict)   # 加载预训练权重
feature_extractor = ViTFeatureExtractor.from_pretrained('ViT_weights')
print(model)

if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(path))      # 再加载模型参数


# 读取测试集中的数据
root_test = 'test-2-10'
test_dataset = Waterlevel(root_test)
test_dataloader = DataLoader(test_dataset, batch_size=1200, shuffle=True)
with torch.no_grad():
    for batch_idx, (img, target) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        test_output = model(img)
        test_output = test_output.logits
        test_output = test_output.to(torch.float32)  # 将对入参和出参做一个类型转换便于计算MSE的值
        test_target = target.to(torch.float32)
        _, predicted = torch.max(test_output.data, dim=1)
        # print(paths)
        # print(test_output)
        print(predicted)
        print(test_target)
