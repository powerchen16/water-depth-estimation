import torch
import numpy as np
import torchvision.models
from matplotlib import pyplot as plt
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tqdm import tqdm
from torch import nn
from label import Waterlevel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. prepare data
root = 'train-2-10'
train_dataset = Waterlevel(root)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

root_val = 'test-2-10'
val_dataset = Waterlevel(root_val)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)


model = ViTForImageClassification.from_pretrained('ViT_weights')
model.load_state_dict(torch.load('ViT_weights/pytorch_model.bin'))   # 加载预训练权重
model.classifier = nn.Linear(model.config.hidden_size, 10)   # 将
feature_extractor = ViTFeatureExtractor.from_pretrained('ViT_weights')
print(model)
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)
    
criterion = nn.CrossEntropyLoss()   # 做分类使用交叉熵损失函数
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)


train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
epoch_list = []
precision_list = []
recall_list = []
f1_list = []
for epoch in range(100):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    for batch_idx, (data, target, path) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        # data = feature_extractor(images=data, return_tensor='pt')
        output = model(data)
        #logits = output.logits
        # 将概率张量转换为张量
        #output = torch.tensor(logits)

        # 将目标标签转换为适当的数据类型
        #target = target.long()
        
        #output = output.to(torch.float32)       # 将对入参和出参做一个类型转换
        #target = target.to(torch.float32)
        
        loss = criterion(output.logits, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(output.logits, dim=1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

    acc_train = train_correct / train_total
    train_acc_list.append(acc_train)
    train_loss_list.append(train_loss)
    # print('epoch %d, loss: %f' % (epoch, loss.item()))

    # val
    model.eval()
    correct = 0
    total = 0
    val_loss_all = 0.0
    y_true = []   # 真实值以及预测值
    y_pred = []
    with torch.no_grad():
        for batch_idx, (data, target, path) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)

            # data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
            # data = feature_extractor(images=data, return_tensor='pt')
            output = model(data)

            val_loss = criterion(output.logits, target.long())             # 记录一下验证集的损失
            val_loss_all += val_loss.item()

            _, predicted = torch.max(output.logits, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc_val = correct / total

    val_acc_list.append(acc_val)
    val_loss_list.append(val_loss_all)
    epoch_list.append(epoch)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')   # macro表示平均
    precision_list.append(precision)
    recall = recall_score(y_true, y_pred, average='macro')
    recall_list.append(recall)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_list.append(f1)


    # save model
    torch.save(model.state_dict(), "ViT-class10-last.pt")
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "ViT-class10-best.pt")
        print("save epoch {} model".format(epoch))
    print("epoch = {},loss = {},acc = {},val_loss = {},acc_val = {}".format(epoch, train_loss, acc_train, val_loss_all, acc_val))
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score = {}".format(accuracy, precision, recall, f1))

print(epoch_list)
print(train_acc_list)
print(train_loss_list)
print(val_loss_list)
print(val_acc_list)
print("accuracy的平均值为：", np.mean(val_acc_list),)
print("precision的平均值为：", np.mean(precision_list))
print("recall的平均值为：", np.mean(recall_list))
print("f1的平均值为：", np.mean(f1_list),)

# 绘制每个epoch的变化图
fig, ax = plt.subplots()
line1 = ax.plot(epoch_list, train_loss_list, color='green', label="train_loss")
line3 = ax.plot(epoch_list, val_loss_list, color='blue', label='val_loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
plt.legend()
ax.spines['right'].set_visible(False)       # ax右轴隐藏

z_ax = ax.twinx() # 创建与轴群ax共享x轴的轴群z_ax
line2 = z_ax.plot(epoch_list, val_acc_list, color='red', label="val_acc")
line4 = z_ax.plot(epoch_list, train_acc_list, color='black', label="train_acc")
z_ax.set_ylabel('acc')

lns = line1+line2+line3+line4
# lns = line1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.savefig('ViT-512-1e-4-数增-1e-4-class10.png')
plt.show()
print("运行完毕")

