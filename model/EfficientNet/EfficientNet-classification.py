import torch
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
# from main import VGGNet_Transfer
from label import Waterlevel
import torchvision
from torch.nn import Linear
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. prepare data
root = r''
train_dataset = Waterlevel(root, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

root_val = r''
val_dataset = Waterlevel(root_val, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

# 2. load model
# eifficientnet_true = torchvision.models.efficientnet_b0(pretrained=True)    # 调用efficientnet网络模型，两种方式都可结果都一样
eifficientnet_true = EfficientNet.from_pretrained('efficientnet-b0')
# 修改最后一层以进行十分类
num_ftrs = eifficientnet_true._fc.in_features
eifficientnet_true._fc = nn.Linear(num_ftrs, 10)
model = eifficientnet_true

if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# print(model)

# 3. prepare super parameters
# class LabelSmoothingLoss(nn.Module):        # 使用标签平滑来抑制过拟合
#     def __init__(self, smoothing=0.1, num_classes=10):
#         super(LabelSmoothingLoss, self).__init__()
#         self.smoothing = smoothing
#         self.num_classes = num_classes
#
#     def forward(self, inputs, targets):
#         epsilon = self.smoothing / self.num_classes
#         smoothed_targets = torch.full_like(inputs, epsilon)
#         smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing + epsilon)
#         loss = torch.sum(-smoothed_targets * torch.log_softmax(inputs, dim=1), dim=1)
#         return loss.mean()
# criterion = LabelSmoothingLoss(smoothing=0.1, num_classes=10)
criterion = nn.CrossEntropyLoss()  # 做分类使用交叉熵损失函数
# criterion = nn.MSELoss()    #做回归使用MSE(均方误差)、以及均方根误差（RMSE）
learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)     # 使用L2正则化（权重衰减）来缓解过拟合
# learning_rate = 1e-2
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# print(optimizer)

# 4. train
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
epoch_list = []
for epoch in range(200):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # output = output.to(torch.float32)       # 将对入参和出参做一个类型转换
        # target = target.to(torch.float32)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(output.data, dim=1)
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
    precision_list = []
    recall_list = []
    f1_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_loss = criterion(output, target.long())             # 记录一下验证集的损失
            val_loss_all += val_loss.item()

            _, predicted = torch.max(output.data, dim=1)
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
    torch.save(model.state_dict(), "last.pt")
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "best.pt")
        print("save epoch {} model".format(epoch))
    print("epoch = {},loss = {},acc = {},val_loss = {},acc_val = {}".format(epoch, train_loss, acc_train, val_loss_all, acc_val))
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score = {}".format(accuracy, precision, recall, f1))

print(epoch_list)
print(train_acc_list)
print(train_loss_list)
print(val_loss_list)
print(val_acc_list)
print("acc的平均值为：", np.mean(val_acc_list))
print("precision的平均值为：", np.mean(precision_list))
print("recall的平均值为：", np.mean(recall_list))
print("f1的平均值为：", np.mean(f1_list))

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
plt.show()
